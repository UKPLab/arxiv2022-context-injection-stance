import random
import lorem
from datasets import Dataset
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def encode_sentence_topic(tokenizer, sentence, topic):
    if len(topic) == 0:
        return tokenizer.encode(text=sentence)
    else:
        return tokenizer.encode(text=sentence)[:-1] + tokenizer.encode(text="[SEP] " + topic)[1:]


def truncate(input_ids):
    if len(input_ids) < 512:
        return input_ids
    else:
        return input_ids[:511] + [102]

def baseline_topic_encoding_function(samples, tokenizer):
    return samples.apply(lambda row: encode_sentence_topic(tokenizer, row["sentence"], row["topic"]), axis=1)


def random_input_encoding_function(samples, tokenizer):
    return samples.apply(lambda row: encode_sentence_topic(tokenizer, row["sentence"], lorem.sentence()), axis=1)


def other_topic_encoding_function(samples, tokenizer):
    topics = samples["topic"].unique()

    random.seed(0)

    return samples.apply(lambda row:
                         tokenizer.encode(text=row["sentence"])[:-1] +
                         tokenizer.encode(
                             text=random.choice([" [SEP] " + topic for topic in topics if topic != row["topic"]]))[1:]
                         , axis=1)


def baseline_context_encoding_function(samples, tokenizer):
    return samples.apply(lambda row:
                         truncate(
                            tokenizer.encode(text=row["sentence"])[:-1] +
                            tokenizer.encode(text="[SEP] " + " ".join(row["context"]))[1:]
                         )
                         , axis=1)


def baseline_joined_topic_encoding_function(samples, tokenizer):
    return samples.apply(lambda row:
                         truncate(
                            tokenizer.encode(text=row["sentence"])[:-1] +
                            tokenizer.encode(text=" [SEP] " + row["topic"] + " " + " ".join(row["context"]))[1:]
                         )
                         , axis=1)


def retro_joined_topic_encoding_function(samples, tokenizer):
    input_ids = samples.apply(lambda row: encode_sentence_topic(tokenizer, row["sentence"], row["topic"]), axis=1)
    context_ids = samples.apply(
        lambda row: [tokenizer.encode(text=ele) for ele in row["context"]],
        axis=1)

    return input_ids, context_ids


def retro_joined_double_topic_encoding_function(samples, tokenizer):
    input_ids = samples.apply(lambda row: encode_sentence_topic(tokenizer, row["sentence"], row["topic"]), axis=1)
    context_ids = samples.apply(
        lambda row: [tokenizer.encode(text=row["topic"] + " [SEP] " + ele) for ele in row["context"]],
        axis=1)

    return input_ids, context_ids
def retro_encoding_function(samples, tokenizer):
    input_ids = samples.apply(lambda row: tokenizer.encode(text=row["sentence"]), axis=1)
    context_ids = samples.apply(
        lambda row: [tokenizer.encode(text=ele) for ele in row["context"]],
        axis=1)

    return input_ids, context_ids


def chunked_retro_encoding_function(samples, tokenizer):
    input_ids = samples.apply(lambda row: tokenizer.encode(text=row["sentence"]), axis=1)

    context_ids = samples.apply(lambda row: [
        [tokenizer.encode(text=ele) for ele in context] for context in row["context"]
    ], axis=1)

    return input_ids, context_ids


def baseline_encoding_function(samples, tokenizer):
    return samples.apply(lambda row: tokenizer.encode(text=row["sentence"]), axis=1)


def load_tokenized_dataset(samples, setting, tokenizer, baseline_enocding_function):
    if setting == "RETRO" or setting == "RETRO_POOLING" or setting == "RETRO_SHORTEST" or setting == "RETRO_JOINED_TOPIC" or setting == "RETRO_JOINED_DOUBLE_TOPIC":
        samples["input_ids"], samples["context_ids"] = baseline_enocding_function(samples, tokenizer)
        dataset = Dataset.from_pandas(samples)
    elif setting == "RETRO_TOPIC":
        samples["context"] = samples["topic"].apply(lambda topic: [topic])
        samples["input_ids"], samples["context_ids"] = baseline_enocding_function(samples, tokenizer)
        dataset = Dataset.from_pandas(samples)
    elif setting == "RETRO_CONCAT" or setting == "RETRO_CONCAT_JOINED_TOPIC":
        samples["context"] = samples["context"].apply(lambda context: [" ".join(context)])
        samples["input_ids"], samples["context_ids"] = baseline_enocding_function(samples, tokenizer)
        dataset = Dataset.from_pandas(samples)
    elif setting == "RETRO_RANDOM_INPUT":
        samples["context"] = samples["topic"].apply(lambda topic: [lorem.sentence()])
        samples["input_ids"], samples["context_ids"] = baseline_enocding_function(samples, tokenizer)
        dataset = Dataset.from_pandas(samples)
    elif setting in ["MULTI_CLS_CONCAT_JOINED_TOPIC", "MULTI_CLS_JOINED_TOPIC_MTL"]:
        samples["context"] = samples["context"].apply(lambda context: [" ".join(context)])
        samples["input_ids"], samples["context_ids"] = baseline_enocding_function(samples, tokenizer)
        dataset = Dataset.from_pandas(samples)
    else:
        samples["input_ids"] = baseline_enocding_function(samples, tokenizer)
        dataset = Dataset.from_pandas(samples)
    return dataset


def select_context_elements(context, setting, k=None):
    try:
        context = eval(context)
    except:
        context = [context]

    if "SHORTEST" in setting:
        context = sorted(context, key=lambda ele: len(ele))

    if k is None:
        return context
    else:
        if len(context) < k:
            context += [""] * (k - len(context))
        return context[:k]


encoding_functions = {
    "BASELINE": baseline_encoding_function,
    "BASELINE_NEIGHBOUR": baseline_context_encoding_function,
    "BASELINE_NEIGHBOUR_SHORTEST": baseline_context_encoding_function,
    "BASELINE_TOPIC": baseline_topic_encoding_function,
    "BASELINE_JOINED_TOPIC": baseline_joined_topic_encoding_function,
    "OTHER_TOPIC": other_topic_encoding_function,
    "RANDOM_INPUT": random_input_encoding_function,
    "RETRO": retro_encoding_function,
    "RETRO_CONCAT": retro_encoding_function,
    "RETRO_POOLING": retro_encoding_function,
    "RETRO_TOPIC": retro_encoding_function,
    "RETRO_JOINED_TOPIC": retro_joined_topic_encoding_function,
    "RETRO_JOINED_DOUBLE_TOPIC": retro_joined_double_topic_encoding_function,
    "RETRO_CONCAT_JOINED_TOPIC": retro_joined_topic_encoding_function,
    "RETRO_SHORTEST": retro_encoding_function,
    "RETRO_RANDOM_INPUT": retro_encoding_function,
    "MULTI_CLS_CONCAT_JOINED_TOPIC": retro_joined_topic_encoding_function,
    "MULTI_CLS_JOINED_TOPIC_MTL": retro_joined_topic_encoding_function
}
