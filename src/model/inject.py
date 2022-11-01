import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.bert.modeling_bert import BertAttention, BertLayer, BertIntermediate, \
    BertOutput, BertPooler, BertModel, BertConfig, BertSelfOutput


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(val, divisor):
    return (val / divisor).is_integer()

def cast_tuple(val, num = 1):
    return val if isinstance(val, tuple) else ((val,) * num)


class InjectCrossAttention(nn.Module):
    def __init__(self, config, k_v_dim, q_dim, position_embedding_type=None):
        super().__init__()

        #### START: BERT SELF-ATTENTION ####
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(q_dim, self.all_head_size)
        self.key = nn.Linear(k_v_dim, self.all_head_size)
        self.value = nn.Linear(k_v_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        #### BERT SELF OUTPUT ####
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
        self.k = config.k

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                injected_hidden_states=None,
                injected_attention_mask=None,
                past_key_value=None,
                output_attentions=False,
                ):
        #### START: BERT SELF ATTENTION ####

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = injected_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = injected_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(injected_hidden_states))
            value_layer = self.transpose_for_scores(self.value(injected_hidden_states))
            attention_mask = injected_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        setting = ""

        # todo: put it into the BertLayerCA
        ### query == context and key == input
        if query_layer.shape[0] > key_layer.shape[0]:
            key_layer = key_layer.repeat_interleave(self.k, dim=0)
            value_layer = value_layer.repeat_interleave(self.k, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.k, dim=0)

            setting = "ENCODER"

        ### query == input and key == context
        elif query_layer.shape[0] < key_layer.shape[0]:
            query_layer = query_layer.repeat_interleave(self.k, dim=0)
            setting = "DECODER"


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask


        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)


        if setting == "DECODER":
            #: todo doing as a function
            avg_attention_probs = rearrange(attention_probs, '(b k) n i j-> b k n i j', k = self.k).mean(dim=1)
            avg_context_layer = rearrange(context_layer, '(b k) n d-> b k n d', k = self.k).mean(dim=1)
        else:
            avg_attention_probs = attention_probs
            avg_context_layer = context_layer

        self_outputs = (avg_context_layer, attention_probs) if output_attentions else (avg_context_layer,)


        #### START: BERT SELF OUTPUT
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertLayerCA(nn.Module):
    def __init__(self, config, k_v_dim, q_dim):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        #TODO: check if we need to randomly initialize
        self.crossattention = InjectCrossAttention(config, k_v_dim, q_dim, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            injected_hidden_states=None,
            injected_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[0]

        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            head_mask,
            injected_hidden_states,
            injected_attention_mask,
            cross_attn_past_key_value,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        # add cross-attn cache to positions 3,4 of present_key_value tuple
        cross_attn_present_key_value = cross_attention_outputs[0]
        present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        # if self.is_decoder:
        #     outputs = outputs + (present_key_value,)
        outputs = outputs + (present_key_value,)

        if output_attentions:
            outputs = outputs + (self_attention_outputs[1], cross_attention_outputs[1])

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

def copy_bert_layer(layer, config, k_v_dim, q_dim):
    assert type(layer) == BertLayer or type(layer) == BertLayerCA
    new_layer = BertLayerCA(config, k_v_dim, q_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for n, mod in layer.named_children():
        if n != "crossattention":
            setattr(new_layer, n, mod)

    for param in new_layer.parameters():
        param.requires_grad = True

    return new_layer

def create_encoder_with_crossattention(encoder, enc_cross_attn_layers, config, k_v_dim, q_dim):
    for i in enc_cross_attn_layers:
        layer_self_att = getattr(encoder, 'layer')[i]
        layer_self_cross_att = copy_bert_layer(layer_self_att, config, k_v_dim, q_dim)
        setattr(encoder.layer, str(i), layer_self_cross_att)
    return encoder


class Inject(nn.Module):

    def __init__(self,
                 model_name,
                 retrieval_model_name,
                 input_encoder_cross_attn_layers = [],
                 context_encoder_cross_attn_layers = [],
                 k = 2,
                 setting = "INJECT",
                 num_labels=3,
                 shared_model=False,
                 frozen_model=False,
                 synchronised_model=False
                 ):
        super().__init__()
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model_name = model_name
        self.config = config
        self.context_encoder_cross_attn_layers = context_encoder_cross_attn_layers
        self.pad_id = config.pad_token_id
        self.num_labels = num_labels
        self.k = k
        self.config.k = k
        self.setting = setting
        self.config.setting = setting
        self.shared_model = shared_model
        self.frozen_model = frozen_model
        self.synchronised_model = synchronised_model
        self.retrieval_model_name = retrieval_model_name if retrieval_model_name else model_name

        # initialize pretrained BERT

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if shared_model:
            base_bert = BertModel.from_pretrained(self.model_name).to(self.device)

            input_encoder = base_bert
            context_encoder = base_bert
        else:
            input_encoder = BertModel.from_pretrained(self.retrieval_model_name).to(self.device)
            context_encoder = BertModel.from_pretrained(self.model_name).to(self.device)

        if frozen_model:
            for param in input_encoder.parameters():
                param.requires_grad = False
            for param in context_encoder.parameters():
                param.requires_grad = False


        # embedding part
        self.input_encoder_embeddings = input_encoder.embeddings
        self.context_encoder_embeddings = context_encoder.embeddings

        if retrieval_model_name != model_name:
            self.input_encoder_hidden_dim = input_encoder.config.hidden_size
            self.context_encoder_hidden_dim = context_encoder.config.hidden_size

        # encoder part
        self.context_encoder = create_encoder_with_crossattention(context_encoder.encoder,
                                                                  context_encoder_cross_attn_layers, config,
                                                                  k_v_dim=self.input_encoder_hidden_dim,
                                                                  q_dim=self.context_encoder_hidden_dim)


        input_encoder_layers = []

        for layer_num in range(config.num_hidden_layers):
            if layer_num in input_encoder_cross_attn_layers:
                bert_layer = getattr(input_encoder.encoder, 'layer')[layer_num]
                bert_layer_with_crossattention = copy_bert_layer(bert_layer, config,
                                                                 k_v_dim=self.context_encoder_hidden_dim,
                                                                 q_dim=self.input_encoder_hidden_dim)
                input_encoder_layers.append(bert_layer_with_crossattention)
            else:
                bert_layer = getattr(input_encoder.encoder, 'layer')[layer_num]
                input_encoder_layers.append(bert_layer)
                
        self.input_encoder_layers = nn.ModuleList(input_encoder_layers)

        # classifier part
        self.pooler = BertPooler(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def update_k(self, new_k):
        self.k = new_k
        for encoder_layer in self.context_encoder.input_encoder_layers:
            if type(encoder_layer) == BertLayerCA:
                encoder_layer.crossattention.k = new_k

        for layer in self.input_encoder_layers:
            if type(layer) == BertLayerCA:
                layer.crossattention.k = new_k

    def async_forward(self, input_embeddings, context_embeddings, output_hidden_states, output_attentions,
                      extended_context_attention_mask, extended_input_attention_mask):

        hidden_states = input_embeddings
        past_key_values = None
        head_mask = None

        all_hidden_states = () if output_hidden_states else None
        all_input_self_attentions = [] if output_attentions else None
        all_context_self_attentions = [] if output_attentions else None
        all_input_cross_attentions = [] if output_attentions else None
        all_context_cross_attentions = [] if output_attentions else None

        conditioned_encoder_hidden_states = None

        for i, layer_module in enumerate(self.input_encoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if type(layer_module) == BertLayerCA:
                # retrieve encoding for context
                context = self.context_encoder(context_embeddings, attention_mask = extended_context_attention_mask,
                                               encoder_hidden_states = hidden_states,
                                               encoder_attention_mask = extended_input_attention_mask)
                conditioned_encoder_hidden_states = context.last_hidden_state
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if i in self.context_encoder_cross_attn_layers:
                encoder_hidden_states = conditioned_encoder_hidden_states
            else:
                encoder_hidden_states = None

            layer_outputs = layer_module(
                hidden_states,
                extended_input_attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                extended_context_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            # if use_cache:
            #     next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)


        return hidden_states, all_hidden_states, all_input_self_attentions, all_context_self_attentions, \
               all_input_cross_attentions, all_context_cross_attentions

    def sync_forward(self, input_embeddings, context_embeddings, output_hidden_states, output_attentions,
                     extended_context_attention_mask, extended_input_attention_mask):

        input_hidden_states = input_embeddings
        context_hidden_states = context_embeddings

        past_key_values = None
        head_mask = None

        all_hidden_states = () if output_hidden_states else None
        all_input_self_attentions = [] if output_attentions else None
        all_context_self_attentions = [] if output_attentions else None
        all_input_cross_attentions = [] if output_attentions else None
        all_context_cross_attentions = [] if output_attentions else None

        last_context_ca_layer = max([i for i, layer in enumerate(self.context_encoder.layer) if type(layer) == BertLayerCA])

        for i, (input_layer, context_layer) in enumerate(zip(self.input_encoder_layers, self.context_encoder.layer)):

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if type(input_layer) == BertLayerCA:

                context_layer_outputs = context_layer(
                    context_hidden_states,
                    extended_context_attention_mask,
                    layer_head_mask,
                    input_hidden_states,
                    extended_input_attention_mask,
                    past_key_value,
                    output_attentions,
                )
                context_hidden_states = context_layer_outputs[0]

                input_layer_outputs = input_layer(
                    input_hidden_states,
                    extended_input_attention_mask,
                    layer_head_mask,
                    context_layer_outputs[0],
                    extended_context_attention_mask,
                    past_key_value,
                    output_attentions,
                )
                input_hidden_states = input_layer_outputs[0]

                if output_attentions:
                    context_self_attentions, context_cross_attentions = context_layer_outputs[2:]
                    input_self_attentions, input_cross_attentions = input_layer_outputs[2:]

                    all_input_self_attentions.append(input_self_attentions)
                    all_context_self_attentions.append(context_self_attentions)

                    all_input_cross_attentions.append(input_cross_attentions)
                    all_context_cross_attentions.append(context_cross_attentions)

            else:
                input_layer_outputs = input_layer(
                    input_hidden_states,
                    extended_input_attention_mask,
                    layer_head_mask,
                    None,
                    None,
                    past_key_value,
                    output_attentions,
                )

                if i <= last_context_ca_layer:

                    context_layer_outputs = context_layer(
                        context_hidden_states,
                        extended_context_attention_mask,
                        layer_head_mask,
                        None,
                        None,
                        past_key_value,
                        output_attentions,
                    )
                    context_hidden_states = context_layer_outputs[0]

                input_hidden_states = input_layer_outputs[0]

                if output_attentions:
                    input_attentions = input_layer_outputs[1]
                    context_attentions = context_layer_outputs[1]

                    all_input_self_attentions.append(input_attentions)
                    all_context_self_attentions.append(context_attentions)

        return input_hidden_states, all_hidden_states, all_input_self_attentions, all_context_self_attentions, \
               all_input_cross_attentions, all_context_cross_attentions


    def forward(
            self,
            input_ids,
            attention_mask,
            labels=None,
            context_input_ids=None,
            context_attention_mask=None,
            output_attentions=False,
            # return_dict = False
    ):
        """
        b - batch
        n - sequence length / chunk length
        k - number of chunks
        d - feature dimension
        r - num retrieved neighbors
        """

        # if not exists(retrieved):
        #     return self.forward_without_retrieval(seq)

        input_token_type_ids = torch.zeros(input_ids.shape).long().to(self.device)
        context_token_type_ids = torch.zeros(context_input_ids.shape).long().to(self.device)

        context_input_ids = rearrange(context_input_ids, 'b k n -> (b k) n').to(self.device)
        context_token_type_ids = rearrange(context_token_type_ids, 'b k n -> (b k) n')

        context_attention_mask = rearrange(context_attention_mask, 'b k n -> (b k) n').to(self.device)

        extended_input_attention_mask = attention_mask[:, None, None, :]
        extended_context_attention_mask = context_attention_mask[:, None, None, :]

        input_embeddings = self.input_encoder_embeddings(input_ids, input_token_type_ids)

        context_embeddings = self.context_encoder_embeddings(context_input_ids, context_token_type_ids)

        output_hidden_states = False

        if self.synchronised_model:
            hidden_states, all_hidden_states, all_input_self_attentions, all_context_self_attentions, \
            all_input_cross_attentions, all_context_cross_attentions = self.sync_forward(input_embeddings,
                                                                                         context_embeddings,
                                                                                         output_hidden_states,
                                                                                         output_attentions,
                                                                                         extended_context_attention_mask,
                                                                                         extended_input_attention_mask)
        else:
            hidden_states, all_hidden_states, all_input_self_attentions, all_context_self_attentions, \
            all_input_cross_attentions, all_context_cross_attentions = self.async_forward(input_embeddings,
                                                                                          context_embeddings,
                                                                                          output_hidden_states,
                                                                                          output_attentions,
                                                                                          extended_context_attention_mask,
                                                                                          extended_input_attention_mask)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.detach().cpu(),)

        sequence_output = hidden_states
        pooled_output = self.pooler(sequence_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # if not return_dict:
        #     return (sequence_output, pooled_output) + encoder_outputs[1:]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        outputs = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
            # attentions=attentions,
        )
        if output_attentions:
            return (outputs, all_input_self_attentions, all_input_cross_attentions, all_context_self_attentions,
                    all_context_cross_attentions)
        else:
            return outputs


    def forward_with_embeddings(
            self,
            input_embeddings,
            attention_mask,
            labels=None,
            context_embeddings=None,
            context_attention_mask=None,
            output_attentions=False,
            # return_dict = False
    ):
        """
        b - batch
        n - sequence length / chunk length
        k - number of chunks
        d - feature dimension
        r - num retrieved neighbors
        """

        # if not exists(retrieved):
        #     return self.forward_without_retrieval(seq)

        #input_token_type_ids = torch.zeros(input_ids.shape).long().to(self.device)

        #input_embeddings = self.input_encoder_embeddings(input_ids, input_token_type_ids)

        context_attention_mask = rearrange(context_attention_mask, 'b k n -> (b k) n').to(self.device)

        extended_input_attention_mask = attention_mask[:, None, None, :]
        extended_context_attention_mask = context_attention_mask[:, None, None, :]

        output_hidden_states = False

        if self.synchronised_model:
            hidden_states, all_hidden_states, all_input_self_attentions, all_context_self_attentions, \
            all_input_cross_attentions, all_context_cross_attentions = self.sync_forward(input_embeddings,
                                                                                         context_embeddings,
                                                                                         output_hidden_states,
                                                                                         output_attentions,
                                                                                         extended_context_attention_mask,
                                                                                         extended_input_attention_mask)
        else:
            hidden_states, all_hidden_states, all_input_self_attentions, all_context_self_attentions, \
            all_input_cross_attentions, all_context_cross_attentions = self.async_forward(input_embeddings,
                                                                                          context_embeddings,
                                                                                          output_hidden_states,
                                                                                          output_attentions,
                                                                                          extended_context_attention_mask,
                                                                                          extended_input_attention_mask)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.detach().cpu(),)

        sequence_output = hidden_states
        pooled_output = self.pooler(sequence_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # if not return_dict:
        #     return (sequence_output, pooled_output) + encoder_outputs[1:]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        outputs = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
            # attentions=attentions,
        )
        if output_attentions:
            return (outputs, all_input_self_attentions, all_input_cross_attentions, all_context_self_attentions,
                    all_context_cross_attentions)
        else:
            return outputs