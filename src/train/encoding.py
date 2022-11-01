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


def inject_joined_topic_encoding_function(samples, tokenizer):
    input_ids = samples.apply(lambda row: encode_sentence_topic(tokenizer, row["sentence"], row["topic"]), axis=1)
    context_ids = samples.apply(
        lambda row: [tokenizer.encode(text=ele) for ele in row["context"]],
        axis=1)

    return input_ids, context_ids


def inject_joined_double_topic_encoding_function(samples, tokenizer):
    input_ids = samples.apply(lambda row: encode_sentence_topic(tokenizer, row["sentence"], row["topic"]), axis=1)
    context_ids = samples.apply(
        lambda row: [tokenizer.encode(text=row["topic"] + " [SEP] " + ele) for ele in row["context"]],
        axis=1)

    return input_ids, context_ids
def inject_encoding_function(samples, tokenizer):
    input_ids = samples.apply(lambda row: tokenizer.encode(text=row["sentence"]), axis=1)
    context_ids = samples.apply(
        lambda row: [tokenizer.encode(text=ele) for ele in row["context"]],
        axis=1)

    return input_ids, context_ids


def chunked_inject_encoding_function(samples, tokenizer):
    input_ids = samples.apply(lambda row: tokenizer.encode(text=row["sentence"]), axis=1)

    context_ids = samples.apply(lambda row: [
        [tokenizer.encode(text=ele) for ele in context] for context in row["context"]
    ], axis=1)

    return input_ids, context_ids


def baseline_encoding_function(samples, tokenizer):
    return samples.apply(lambda row: tokenizer.encode(text=row["sentence"]), axis=1)


def load_tokenized_dataset(samples, setting, tokenizer, baseline_enocding_function):
    if setting == "INJECT" or setting == "INJECT_POOLING" or setting == "INJECT_SHORTEST" or setting == "INJECT_JOINED_TOPIC" or setting == "INJECT_JOINED_DOUBLE_TOPIC":
        samples["input_ids"], samples["context_ids"] = baseline_enocding_function(samples, tokenizer)
        dataset = Dataset.from_pandas(samples)
    elif setting == "INJECT_TOPIC":
        samples["context"] = samples["topic"].apply(lambda topic: [topic])
        samples["input_ids"], samples["context_ids"] = baseline_enocding_function(samples, tokenizer)
        dataset = Dataset.from_pandas(samples)
    elif setting == "INJECT_CONCAT" or setting == "INJECT_CONCAT_JOINED_TOPIC":
        samples["context"] = samples["context"].apply(lambda context: [" ".join(context)])
        samples["input_ids"], samples["context_ids"] = baseline_enocding_function(samples, tokenizer)
        dataset = Dataset.from_pandas(samples)
    elif setting == "INJECT_RANDOM_INPUT":
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
    "INJECT": inject_encoding_function,
    "INJECT_CONCAT": inject_encoding_function,
    "INJECT_POOLING": inject_encoding_function,
    "INJECT_TOPIC": inject_encoding_function,
    "INJECT_JOINED_TOPIC": inject_joined_topic_encoding_function,
    "INJECT_JOINED_DOUBLE_TOPIC": inject_joined_double_topic_encoding_function,
    "INJECT_CONCAT_JOINED_TOPIC": inject_joined_topic_encoding_function,
    "INJECT_SHORTEST": inject_encoding_function,
    "INJECT_RANDOM_INPUT": inject_encoding_function,
    "MULTI_CLS_CONCAT_JOINED_TOPIC": inject_joined_topic_encoding_function,
    "MULTI_CLS_JOINED_TOPIC_MTL": inject_joined_topic_encoding_function
}
