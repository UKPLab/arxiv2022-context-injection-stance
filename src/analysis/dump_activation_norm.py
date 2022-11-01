import glob
import itertools
import json
import os
from collections import defaultdict

import dropbox
import pandas
import torch
from mlflow.tracking import MlflowClient
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification

from analysis.captum.layer_activation import LayerActivation
from model.retro_bert import ArgRetro, BertLayerCA
from run import load_samples
from train.encoding import load_tokenized_dataset, encoding_functions
from train.utils import RetroDataCollatorWithPadding, truncate_sentence
from einops import rearrange


def check_dropbox_file_exists(path, base_folder="retro-cl-models"):
    dbx = dropbox.Dropbox(app_key="rsx45nmlpp77mdy", app_secret="yopki6xvye8g81n",
                          oauth2_refresh_token="O7naVr6RvusAAAAAAAAAAZUEMKXevXbcZDzDnDXtIw4U-Q0FeAcywwCKAnIiRvt5",
                          timeout=30000)
    #dbx.refresh_access_token()
    try:
        metadata = dbx.files_get_metadata("/" + base_folder + "/" + path)
    except:
        return False
    return True


def download_model(dump_path: str):
    dbx = dropbox.Dropbox(app_key="rsx45nmlpp77mdy", app_secret="yopki6xvye8g81n",
                          oauth2_refresh_token="O7naVr6RvusAAAAAAAAAAZUEMKXevXbcZDzDnDXtIw4U-Q0FeAcywwCKAnIiRvt5",
                          timeout=30000)
    # dbx.refresh_access_token()

    local_path = "tmp/" + dump_path

    os.system("mkdir -p " + local_path)
    dbx.files_download_to_file(local_path + "/model.tar.gz", "/retro-cl-models/" + dump_path + ".tar.gz")

    os.system("mkdir -p " + local_path + "/model")
    os.system("tar -xf " + local_path + "/model.tar.gz -C " + local_path + "/model")
    os.system("rm " + local_path + "/model.tar.gz")

    return "tmp/" + dump_path + "/model/"


def download_predictions(run_id: str):
    dbx = dropbox.Dropbox(app_key="rsx45nmlpp77mdy", app_secret="yopki6xvye8g81n",
                          oauth2_refresh_token="O7naVr6RvusAAAAAAAAAAZUEMKXevXbcZDzDnDXtIw4U-Q0FeAcywwCKAnIiRvt5",
                          timeout=30000)
    # dbx.refresh_access_token()

    local_path = "tmp/" + run_id

    try:
        os.system("mkdir -p " + local_path)
        dbx.files_download_to_file(local_path + "/predictions.csv",
                                   "/retro-cl-preds/output-" + run_id + "/predictions.csv")
    except:
        return pandas.DataFrame()

    predictions = pandas.read_csv(local_path + "/predictions.csv")

    return predictions


mlflow_url = "http://10.176.133.1:8000/" if os.system(
    "ping -c 1 10.176.133.1") is 0 else "http://6.tcp.eu.ngrok.io:12311"
client = MlflowClient(mlflow_url)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="argmin")
    parser.add_argument("--setting", type=str, default="RETRO_JOINED_TOPIC")
    parser.add_argument("--variation", type=str, default="relevant_final")

    args = parser.parse_args()

    task = args.task
    setting = args.setting
    variation = args.variation

    data_folder = "../../data/"

    experiment = client.get_experiment_by_name("bench_100-" + task + "-retro")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.seed='0' and params.setting='" + setting + "' and params.variation='" + variation + "'"
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    for run in tqdm(runs):

        dump_path = run.data.params["dump_path"]

        if not check_dropbox_file_exists(dump_path + ".tar.gz"):
            continue

        seed = int(run.data.params["seed"])
        fold = int(run.data.params["fold"])
        variation = run.data.params["variation"]
        setting = run.data.params["setting"]
        shared_model = run.data.params["shared_model"]
        frozen_model = run.data.params["frozen_model"]
        synchronised_model = run.data.params["synchronised_model"]
        model_name = run.data.params["model_name_str"]
        retrieval_model_name = run.data.params["retrieval_model_name"]
        input_encoder_cross_attn_layers = run.data.params.get("input_encoder_cross_attn_layers", None)
        context_encoder_cross_attn_layers = run.data.params.get("context_encoder_cross_attn_layers", None)
        is_cross_topic = False
        k = 2
        samples = load_samples(task, variation, setting, k, 3, fold, is_cross_topic, data_folder)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        samples["sentence"] = samples["sentence"].apply(lambda sentence: truncate_sentence(sentence, 100, tokenizer))
        samples["context"] = samples["context"].apply(lambda context: [truncate_sentence(ele, 500, tokenizer) for ele in context])

        num_labels = len(set(samples["label"]))

        if variation in ["conceptnet_pairs"] or setting in ["RETRO_CONCAT_JOINED_TOPIC", "RETRO_JOINED_DOUBLE_TOPIC"]:
            continue

        if "np" in setting and "key" not in setting:
            continue

        if "RETRO" in setting:
            model = ArgRetro(
                model_name,
                retrieval_model_name=None if retrieval_model_name == 'None' else retrieval_model_name,
                input_encoder_cross_attn_layers=eval(input_encoder_cross_attn_layers),
                context_encoder_cross_attn_layers=eval(context_encoder_cross_attn_layers),
                k=k,
                setting=setting,
                num_labels=num_labels,
                shared_model=eval(shared_model),
                frozen_model=eval(frozen_model),
                synchronised_model=eval(synchronised_model)
            ).to(device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

        encoding_function = encoding_functions[setting]

        samples = samples[samples["split"].isin(["test", "train", "val", "dev"])]

        test_samples = samples[samples["split"] == "test"]
        train_samples = samples[samples["split"] == "train"]
        dev_samples = samples[samples["split"].isin(["dev", "val"])]

        test_dataset = load_tokenized_dataset(test_samples, setting, tokenizer, encoding_function)
        dev_dataset = load_tokenized_dataset(dev_samples, setting, tokenizer, encoding_function)
        train_dataset = load_tokenized_dataset(train_samples, setting, tokenizer, encoding_function)

        model_path = download_model(dump_path)
        predictions = download_predictions(dump_path)

        if "RETRO" in setting:
            data_collator = RetroDataCollatorWithPadding(tokenizer=tokenizer)
        else:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        model_files = glob.glob("tmp/" + dump_path + "/**/**/**/pytorch_model.bin")

        if len(model_files) == 0:
            continue
        else:
            model_file = model_files[0]

        model_state = torch.load(model_file, map_location=device)

        #model_state = torch.load(
        #    "tmp/183fa060-0225-11ed-b2a1-0242ac130005/model/output-183fa060-0225-11ed-b2a1-0242ac130005/checkpoint-2140/pytorch_model.bin",
        #    map_location=device)
        model.load_state_dict(model_state)

        os.system("rm -rf tmp/" + dump_path )

        @torch.enable_grad()
        def predict(input_ids, attention_mask, context_input_ids, context_attention_mask, labels):
            preds = model(
                input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask
            )

            logits = torch.nn.functional.softmax(preds["logits"], dim=1)
            return logits

        @torch.enable_grad()
        def predict_baseline(input_ids, attention_mask):
            preds = model(
                input_ids=input_ids, attention_mask=attention_mask,
            )
            logits = torch.nn.functional.softmax(preds["logits"], dim=1)
            return logits

        for split, dataset in [("train", train_dataset), ("test", test_dataset), ("dev", dev_dataset)]:

            task_activations = defaultdict(list)

            elements = list(dataset)

            activations = []

            def get_retro_normalized_activation(layers, dense_activations, input_ids, attention_mask, context_input_ids,
                                                context_attention_mask, labels):
                all_head_size = 768
                head_size = 64
                num_heads = 12

                #input_embeddings = interpretable_input_embedding.indices_to_embeddings(input_ids.to(device))
                #context_input_ids = rearrange(context_input_ids, "b k l -> (b k) l", k=2)
                #context_embeddings = interpretable_context_embedding.indices_to_embeddings(context_input_ids.to(device))

                layer_activation = LayerActivation(predict, layers)

                value_layer_activations = layer_activation.attribute(
                    inputs=(input_ids.to(device), attention_mask.to(device), context_input_ids.to(device), context_attention_mask.to(device)),
                    additional_forward_args=(labels))
                value_layer_activations = torch.stack(value_layer_activations)
                new_x_shape = value_layer_activations.size()[:-1] + (num_heads, head_size)
                value_layer_activations = value_layer_activations.view(*new_x_shape)
                value_layer_activations = value_layer_activations.permute(0, 1, 3, 2, 4)
                value_layer_activations = value_layer_activations.permute(0, 1, 3, 2, 4).contiguous()
                value_layer_activations_shape = value_layer_activations.size()

                value_layer_activations = value_layer_activations.view(
                    value_layer_activations_shape[:-1] + (1, value_layer_activations_shape[-1],))

                dense_activations = dense_activations.view(len(layers), all_head_size, num_heads, head_size)
                dense_activations = dense_activations.permute(0, 2, 3, 1).contiguous()
                f_x = torch.stack([value_layer_activations_i.matmul(dense_activations_i) for
                                   value_layer_activations_i, dense_activations_i in
                                   zip(value_layer_activations, dense_activations)])
                f_x_shape = f_x.size()
                f_x = f_x.view(f_x_shape[:-2] + (f_x_shape[-1],))
                f_x = f_x.permute(0, 1, 3, 2, 4).contiguous()
                f_x_norm = torch.linalg.norm(f_x, dim=-1)
                return f_x_norm.mean(dim=[2]).transpose(0, 1)

            def get_baseline_normalized_activation(layers, dense_activations, input_ids, attention_mask):
                all_head_size = 768
                head_size = 64
                num_heads = 12

                layer_activation = LayerActivation(predict_baseline, layers)

                value_layer_activations = layer_activation.attribute(inputs=(input_ids.to(device), attention_mask.to(device)))
                value_layer_activations = torch.stack(value_layer_activations)
                new_x_shape = value_layer_activations.size()[:-1] + (num_heads, head_size)
                value_layer_activations = value_layer_activations.view(*new_x_shape)
                value_layer_activations = value_layer_activations.permute(0, 1, 3, 2, 4)
                value_layer_activations = value_layer_activations.permute(0, 1, 3, 2, 4).contiguous()
                value_layer_activations_shape = value_layer_activations.size()

                value_layer_activations = value_layer_activations.view(
                    value_layer_activations_shape[:-1] + (1, value_layer_activations_shape[-1],))

                dense_activations = dense_activations.view(len(layers), all_head_size, num_heads, head_size)
                dense_activations = dense_activations.permute(0, 2, 3, 1).contiguous()
                f_x = torch.stack([value_layer_activations_i.matmul(dense_activations_i) for
                                   value_layer_activations_i, dense_activations_i in
                                   zip(value_layer_activations, dense_activations)])
                f_x_shape = f_x.size()
                f_x = f_x.view(f_x_shape[:-2] + (f_x_shape[-1],))
                f_x = f_x.permute(0, 1, 3, 2, 4).contiguous()
                f_x_norm = torch.linalg.norm(f_x, dim=-1)
                return f_x_norm.mean(dim=[2]).transpose(0, 1)

            def process_token_activations(token_ids, token_activations):
                token_activations = token_activations[:, token_ids != 0].detach().cpu().numpy().tolist()
                token_ids = token_ids[token_ids != 0]
                tokens = tokenizer.batch_decode(token_ids.unsqueeze(dim=-1))

                return tokens, token_activations

            def get_retro_batch_activations(batch_samples, preds, labels):
                batch = data_collator(batch_samples)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                context_attention_mask = batch["context_attention_mask"]
                context_input_ids = batch["context_input_ids"]
                labels = batch["labels"].to(device)

                #model.context_encoder.layer = model.context_encoder.layer[:6]

                input_self_attention_layers = [layer.attention.self.value for layer in model.input_encoder_layers]
                input_self_attention_dense_activation = torch.stack(
                    [layer.attention.output.dense.weight for layer in model.input_encoder_layers])
                context_self_attention_layers = [layer.attention.self.value for layer in model.context_encoder.layer]
                context_self_attention_dense_activation = torch.stack(
                    [layer.attention.output.dense.weight for layer in model.context_encoder.layer])

                input_cross_attention_layers = [layer.crossattention.value for layer in model.input_encoder_layers if
                                                type(layer) == BertLayerCA]
                input_cross_attention_dense_activation = torch.stack(
                    [layer.crossattention.output.dense.weight for layer in model.input_encoder_layers if
                     type(layer) == BertLayerCA])
                context_cross_attention_layers = [layer.crossattention.value for layer in model.context_encoder.layer if
                                                  type(layer) == BertLayerCA]
                context_cross_attention_dense_activation = torch.stack(
                    [layer.crossattention.output.dense.weight for layer in model.context_encoder.layer if
                     type(layer) == BertLayerCA])

                input_self_attention_normalized_activations = get_retro_normalized_activation(input_self_attention_layers,
                                                                                              input_self_attention_dense_activation,
                                                                                              input_ids, attention_mask,
                                                                                              context_input_ids,
                                                                                              context_attention_mask,
                                                                                              labels)
                input_self_attention_outputs = [process_token_activations(token_ids, token_activations) for
                                                token_ids, token_activations in
                                                zip(input_ids, input_self_attention_normalized_activations)]

                ### layer wise activation

                context_self_attention_normalized_activations = get_retro_normalized_activation(
                    context_self_attention_layers, context_self_attention_dense_activation,
                    input_ids, attention_mask, context_input_ids, context_attention_mask, labels
                )

                context_self_attention_normalized_activations = rearrange(context_self_attention_normalized_activations,
                                                                          '(b k) l n -> b k l n', k=2)
                context_self_attention_outputs = [
                    [
                        process_token_activations(token_ids, token_activations)
                        for token_ids, token_activations in zip(context_ids, context_activations)
                    ]

                    for context_ids, context_activations in
                    zip(context_input_ids, context_self_attention_normalized_activations)
                ]

                context_cross_attention_normalized_activations = get_retro_normalized_activation(
                    context_cross_attention_layers, context_cross_attention_dense_activation,
                    input_ids, attention_mask, context_input_ids, context_attention_mask, labels
                )
                input_cross_attention_outputs = [process_token_activations(token_ids, token_activations) for
                                                 token_ids, token_activations in
                                                 zip(input_ids, context_cross_attention_normalized_activations)]

                input_cross_attention_normalized_activations = get_retro_normalized_activation(
                    input_cross_attention_layers,
                    input_cross_attention_dense_activation, input_ids, attention_mask,
                    context_input_ids,context_attention_mask, labels
                )
                input_cross_attention_normalized_activations = rearrange(input_cross_attention_normalized_activations,
                                                                         '(b k) l n -> b k l n', k=2)
                context_cross_attention_outputs = [
                    [
                        process_token_activations(token_ids, token_activations)
                        for token_ids, token_activations in zip(context_ids, context_activations)
                    ]

                    for context_ids, context_activations in
                    zip(context_input_ids, input_cross_attention_normalized_activations)
                ]

                sentences = [batch_sample["sentence"] for batch_sample in batch_samples]
                contexts = [batch_sample["context"] for batch_sample in batch_samples]

                batch_activations = [
                    {
                        "sentence": sentence,
                        "context": context,
                        "label": int(label),
                        "pred": int(pred),
                        "input_tokens": input_self_attention_output[0],
                        "context_tokens": [ele[0] for ele in context_self_attention_output],
                        "input_self_activations": input_self_attention_output[1],
                        "input_cross_activations": input_cross_attention_output[1],
                        "context_self_activations": [ele[1] for ele in context_self_attention_output],
                        "context_cross_activations": [ele[1] for ele in context_cross_attention_output],
                    }
                    for i, (sentence, context, pred, label, input_self_attention_output, context_self_attention_output,
                            input_cross_attention_output, context_cross_attention_output) in enumerate(
                        zip(sentences, contexts, preds, labels, input_self_attention_outputs,
                            context_self_attention_outputs, input_cross_attention_outputs, context_cross_attention_outputs))
                ]

                return batch_activations

            def get_baseline_batch_activations(batch_samples, preds, labels):

                sentences = [ele["sentence"] for ele in batch_samples]
                contexts = [batch_sample["context"] for batch_sample in batch_samples]
                max_length = max([len(batch_sample["input_ids"]) for batch_sample in batch_samples])
                input_ids = torch.stack(
                    [torch.tensor(batch_sample["input_ids"] + [0] * (max_length - len(batch_sample["input_ids"]))) for
                     batch_sample in batch_samples])
                attention_mask = (input_ids > 0).int()

                input_self_attention_layers = [layer.attention.self.value for layer in model.bert.encoder.layer]
                input_self_attention_dense_activation = torch.stack(
                    [layer.attention.output.dense.weight for layer in model.bert.encoder.layer])

                input_self_attention_normalized_activations = get_baseline_normalized_activation(
                    input_self_attention_layers, input_self_attention_dense_activation, input_ids, attention_mask)
                input_self_attention_outputs = [process_token_activations(token_ids, token_activations) for
                                                token_ids, token_activations in
                                                zip(input_ids, input_self_attention_normalized_activations)]

                batch_activations = [
                    {
                        "sentence": sentence,
                        "context": context,
                        "label": int(label),
                        "pred": int(pred),
                        "input_tokens": input_self_attention_output[0],
                        "input_self_activations": input_self_attention_output[1],
                    }
                    for i, (sentence, context, pred, label, input_self_attention_output) in
                    enumerate(zip(sentences, contexts, preds, labels, input_self_attention_outputs))
                ]

                return batch_activations



            for start in tqdm(range(0, len(elements), 3)):
                model.zero_grad()
                batch_samples = elements[start:start + 3]
                preds = list(predictions.iloc[start:start + 3]["preds"])
                labels = list(predictions.iloc[start:start + 3]["label"])

                if "RETRO" in setting:
                    batch_activations = get_retro_batch_activations(batch_samples, preds, labels)
                else:
                    batch_activations = get_baseline_batch_activations(batch_samples, preds, labels)

                activations += batch_activations

            task_activations[(
            setting, variation, seed, shared_model, frozen_model, model_name, input_encoder_cross_attn_layers,
            context_encoder_cross_attn_layers)].append(activations)

            for (setting, variation, seed, shared_model, frozen_model, model_name, input_encoder_cross_attn_layers,
                 context_encoder_cross_attn_layers), activations_list in task_activations.items():

                file_name = "__".join([
                    task, split, setting, variation, str(seed), str(shared_model), str(frozen_model),
                    model_name, str(input_encoder_cross_attn_layers),
                    str(context_encoder_cross_attn_layers)]
                )

                with open('normalized_activations/' + file_name + "_" + '.json', 'w') as file:
                    json.dump(activations_list, file)



if __name__ == '__main__':
    main()
