import glob
import itertools
import json
from collections import defaultdict, Counter, OrderedDict
from multiprocessing import Pool

import plotly.express as px

import numpy
import pandas
from tqdm import tqdm
from transformers import AutoTokenizer

from run import load_samples
from train.utils import truncate_sentence


def get_run_params(task_file):
    task_file = task_file.replace("normalized_activations/", "")
    task_file = task_file.replace(".json", "")
    task, split, setting, variation, seed, _, _, model_name, injection_layer, _ = task_file.split("__")
    injection_layer = eval(injection_layer)
    if injection_layer != None:
        injection_layer = injection_layer[0]
    return (task, split, setting, variation, seed, model_name, injection_layer)


def normalize_activations(activations):
    norms = numpy.linalg.norm(activations)
    return numpy.divide(activations, norms)

def per_token_results(activation_frame):
    if "cross_activation" in activation_frame:
        return [
            {
                "token": token,
                "occurrence": token_frame.shape[0],
                "topic": token_frame["topic"].to_numpy().tolist(),
                "result": token_frame["result"].to_numpy().tolist(),
                "label": token_frame["label"].to_numpy().tolist(),
                "prediction": token_frame["prediction"].to_numpy().tolist(),
                "odds-ratio_topic": token_frame["odds-ratio_topic"].to_numpy().tolist(),
                "odds-ratio_label": token_frame["odds-ratio_label"].to_numpy().tolist(),
                #"odds-ratio_topic-label": token_frame["odds-ratio_topic-label"].to_numpy().tolist(),
                "self_activations": token_frame["self_activation"].to_numpy().tolist(),
                "cross_activations": token_frame["cross_activation"].to_numpy().tolist(),
                "diff_activations": token_frame["diff_activation"].to_numpy().tolist(),
            }
            for token, token_frame in activation_frame.groupby("token")
        ]
    else:
        return [
            {
                "token": token,
                "topic": token_frame["topic"].to_numpy().tolist(),
                "result": token_frame["result"].to_numpy().tolist(),
                "label": token_frame["label"].to_numpy().tolist(),
                "odds-ratio_topic": token_frame["odds-ratio_topic"].to_numpy().tolist(),
                "odds-ratio_label": token_frame["odds-ratio_label"].to_numpy().tolist(),
                #"odds-ratio_topic-label": token_frame["odds-ratio_topic-label"].to_numpy().tolist(),
                "prediction": token_frame["prediction"].to_numpy().tolist(),
                "occurrence": token_frame.shape[0],
                "self_activations": token_frame["self_activation"].to_numpy().tolist(),
            }
            for token, token_frame in activation_frame.groupby("token")
        ]

def per_token_average_results(activation_frame):
    if "cross_activation" in activation_frame:
        return [
            {
                "token": token,
                "occurrence": token_frame.shape[0],
                "average_self_activation": token_frame["self_activation"].mean(),
                "std_self_activation": token_frame["self_activation"].std(),
                "average_cross_activation": token_frame["cross_activation"].mean(),
                "std_cross_activation": token_frame["cross_activation"].std(),
                "average_diff_activation": token_frame["diff_activation"].mean(),
                "std_diff_activation": token_frame["diff_activation"].std(),
                "average_odds-ratio_topic": token_frame["odds-ratio_topic"].mean(),
                "std_odds-ratio_topic": token_frame["odds-ratio_topic"].std(),
            }
            for token, token_frame in activation_frame.groupby("token")
        ]
    else:
        return [
            {
                "token": token,
                "occurrence": token_frame.shape[0],
                "average_self_activation": token_frame["self_activation"].mean(),
                "std_self_activation": token_frame["self_activation"].std(),
                "average_odds-ratio_topic": token_frame["odds-ratio_topic"].mean(),
                "std_odds-ratio_topic": token_frame["odds-ratio_topic"].std(),
            }
            for token, token_frame in activation_frame.groupby("token")
        ]

def dump_average_token_activations(activation_frame, source, task, split,  setting, variation, injection_layer):
    finger_print = "__".join([task, split, source, setting, variation, injection_layer])

    token_frame = pandas.DataFrame(per_token_average_results(activation_frame))
    token_frame.to_csv("average_token_activations/overall__" + finger_print + ".csv")

    topic_results = []
    for topic, topic_frame in activation_frame.groupby("topic"):
        topic_token_frame = pandas.DataFrame(per_token_average_results(topic_frame))
        topic_token_frame["topic"] = topic
        topic_results.append(topic_token_frame)

    if len(topic_results) != 0:
        topic_result_frame = pandas.concat(topic_results)
        topic_result_frame.to_csv("average_token_activations/topic__" + finger_print + ".csv")

    accuracy_token_results = []
    for accuracy, accuracy_frame in activation_frame.groupby("result"):
        accuracy_token_frame = pandas.DataFrame(per_token_average_results(topic_frame))
        accuracy_token_frame["accuracy"] = accuracy
        accuracy_token_results.append(accuracy_token_frame)

    if len(accuracy_token_results) != 0:
        accuracy_token_result_frame = pandas.concat(accuracy_token_results)
        accuracy_token_result_frame.to_csv("average_token_activations/accuracy__" + finger_print + ".csv")

    accuracy_detail_token_results = []
    for (accuracy, label, pred), topic_frame in activation_frame.groupby(["result", "label", "prediction"]):
        accuracy_frame = pandas.DataFrame(per_token_average_results(topic_frame))
        accuracy_frame["accuracy"] = accuracy
        accuracy_frame["label"] = label
        accuracy_frame["pred"] = pred
        accuracy_detail_token_results.append(accuracy_frame)

    if len(accuracy_detail_token_results) != 0:
        accuracy_detail_token_result_frame = pandas.concat(accuracy_detail_token_results)
        accuracy_detail_token_result_frame.to_csv("average_token_activations/accuracy-detail__" + finger_print + ".csv")


def dump_token_activations(activation_frame, source, task, split, setting, variation, injection_layer):
    finger_print = "__".join([task, split, source, setting, variation, injection_layer])

    token_frame = pandas.DataFrame(per_token_results(activation_frame))
    token_frame.to_csv("token_activations/overall__" + finger_print + ".csv")

    topic_results = []
    for topic, topic_frame in activation_frame.groupby("topic"):
        topic_token_frame = pandas.DataFrame(per_token_results(topic_frame))
        topic_token_frame["topic"] = topic
        topic_results.append(topic_token_frame)

    if len(topic_results) != 0:
        topic_result_frame = pandas.concat(topic_results)
        topic_result_frame.to_csv("token_activations/topic__" + finger_print + ".csv")

    accuracy_token_results = []
    for accuracy, accuracy_frame in activation_frame.groupby("result"):
        accuracy_token_frame = pandas.DataFrame(per_token_results(topic_frame))
        accuracy_token_frame["accuracy"] = accuracy
        accuracy_token_results.append(accuracy_token_frame)

    if len(topic_results) != 0:
        accuracy_token_result_frame = pandas.concat(accuracy_token_results)
        accuracy_token_result_frame.to_csv("token_activations/accuracy__" + finger_print + ".csv")

    accuracy_detail_token_results = []
    for (accuracy, label, pred), topic_frame in activation_frame.groupby(["result", "label", "prediction"]):
        accuracy_frame = pandas.DataFrame(per_token_results(topic_frame))
        accuracy_frame["accuracy"] = accuracy
        accuracy_frame["label"] = label
        accuracy_frame["pred"] = pred
        accuracy_detail_token_results.append(accuracy_frame)

    if len(topic_results) != 0:
        accuracy_detail_token_result_frame = pandas.concat(accuracy_detail_token_results)
        accuracy_detail_token_result_frame.to_csv("token_activations/accuracy-detail__" + finger_print + ".csv")


def get_upper_tail_tokens(elements):
    sorted_elements = sorted(elements.items(), key=lambda ele: ele[1], reverse=True)

    return sorted_elements[:30]


def dump_correlations(correlations, task, split, setting, variation, injection_layer):
    finger_print = "__".join([task, split, setting, variation, str(injection_layer)])

    pandas.DataFrame(correlations).to_csv("correlations/" + finger_print + ".csv")


def dump_correlation_matrix(activation_frame, source, task, split, setting, variation, injection_layer):
    relevant_columns = activation_frame.select_dtypes(include=[numpy.float, numpy.int]).columns

    for color_column in ["topic", "label", "result"]:
        finger_print = "__".join([color_column, task, split, source, setting, variation, str(injection_layer)])
        correlation_matrix_fig = px.scatter_matrix(
            activation_frame, dimensions=relevant_columns, color=color_column, hover_data=["token", "topic", "label", "result"]
        )
        correlation_matrix_fig = correlation_matrix_fig.update_traces(marker=dict(size=2,line=dict(width=0)),opacity=0.3)
        correlation_matrix_fig.write_html("correlation_matrices/matrix__" + finger_print + ".html")

def dump_most_relevant_context_tokens(token_co_occurrences, source, task, split, setting, variation, injection_layer):
    finger_print = "__".join([task, split, source, setting, variation, injection_layer])

    averaged_token_co_occurrences = {
        input_token:{
            context_token: numpy.average(values, axis=0)
            for context_token, values in context_tokens.items()
        }
        for input_token, context_tokens in token_co_occurrences.items()
    }

    averaged_token_co_occurrences_tails = {
        input_token:get_upper_tail_tokens({
            context_token: context_cross_activation
            for context_token, (_, context_cross_activation) in context_tokens.items()
        })
        for input_token, context_tokens in averaged_token_co_occurrences.items()
    }

    average_input_diff_activations = {
        input_token: numpy.average([
            input_diff_activation
            for context_token, (input_diff_activation, _) in context_tokens.items()
        ])
        for input_token, context_tokens in averaged_token_co_occurrences.items()
    }
    co_occurrence_elements = [
        {
            "token": token,
            "diff_activation": diff_activation,
            "relevant_tokens": ", ".join([ele[0] for ele in averaged_token_co_occurrences_tails[token]]),
            "relevant_token_scores": ", ".join([str(round(ele[1], 5)) for ele in averaged_token_co_occurrences_tails[token]]),
        }
        for token, diff_activation in average_input_diff_activations.items()
    ]

    pandas.DataFrame(co_occurrence_elements).to_csv("relevant_tokens/" + finger_print + ".csv")





def get_full_activations(tokens, self_activations, cross_activations, topic, result, row):
    activations = []
    for token, self_activation, cross_activation in zip(tokens, self_activations, cross_activations):
        activations.append({
            "token": token,
            "result": result,
            "label": row["label"],
            "prediction": row["pred"],
            "topic": topic,
            "self_activation": self_activation,
            "cross_activation": cross_activation,
            "diff_activation": cross_activation - self_activation
        })
    return activations



def get_self_activations(tokens, self_activations, topic, result, row):
    activations = []
    for token, self_activation in zip(tokens, self_activations):
        activations.append({
            "token": token,
            "result": result,
            "label": row["label"],
            "prediction": row["pred"],
            "topic": topic,
            "self_activation": self_activation,
        })
    return activations

def flatten_tokens(tokens):
    while type(tokens[0]) != str:
        tokens = list(itertools.chain.from_iterable(tokens))

    return tokens

def get_odds_ratio(inputs, target_cols, tokenizer, token_col = ["input_tokens"]):

    token_occurrences = []

    for targets, target_frame in inputs.groupby(target_cols):
        if type(targets) != tuple:
            targets = tuple([targets])

        tokens = flatten_tokens(list(target_frame[token_col].apply(tokenizer.encode).apply(lambda tokens: tokenizer.convert_ids_to_tokens(tokens[1:-1]))))

        for token, occurrence in Counter(tokens).items():
            token_count = {
                "token": token,
                "occurrence": occurrence + 1, # correction to handle zero occurrences
            }
            for target, name in zip(targets, target_cols):
                token_count[name] = target

            token_occurrences.append(token_count)

    token_occurrence_frame = pandas.DataFrame(token_occurrences)
    odds_ratios = defaultdict(lambda: -10000)

    token_ratios = defaultdict(list)
    for targets, target_occurrence_frame in token_occurrence_frame.groupby(target_cols):
        if type(targets) != tuple:
            targets = tuple([targets])

        total_occurrence = target_occurrence_frame["occurrence"].sum()
        inverse_occurrence_frame = token_occurrence_frame.copy()
        for target, name in zip(targets, target_cols):
            inverse_occurrence_frame = inverse_occurrence_frame[inverse_occurrence_frame[name] != target]

        total_inverse_occurrence = inverse_occurrence_frame["occurrence"].sum()

        inverse_occurrences = dict(inverse_occurrence_frame[["token", "occurrence"]].values)


        for index, row in target_occurrence_frame.iterrows():
            token = row["token"]
            token_occurrence = row["occurrence"]
            inverse_token_occurrence = inverse_occurrences.get(token, 1)

            odds_token = token_occurrence/(total_occurrence - token_occurrence)
            inverse_odds_token = inverse_token_occurrence/(total_inverse_occurrence - inverse_token_occurrence)
            token_ratios[token].append(numpy.log(odds_token/inverse_odds_token))
            #odds_ratios[targets + tuple([token])] = numpy.log(odds_token/inverse_odds_token)
        #odds_ratios[token] = max(target_ratios)
    return {
        token: max(ratios)
        for token, ratios in token_ratios.items()
    }


def get_correlation(task, variation, split, setting, injection_layer, input_activation_frame, context_activation_frame=None, result="all"):

    input_correlations = input_activation_frame.corr()

    correlation_entry = {
        "task": task,
        "result": result,
        "variation": variation,
        "split": split,
        "setting": setting,
        "injection_layer": injection_layer,
        "input/self-activation/odds-ratio_topic": input_correlations["self_activation"]["odds-ratio_topic"],
        "input/self-activation/odds-ratio_label": input_correlations["self_activation"]["odds-ratio_label"],
        "input/self-activation/odds-ratio_split": input_correlations["self_activation"]["odds-ratio_split"],
        "input/self-activation/odds-ratio_topic-label": input_correlations["self_activation"]["odds-ratio_topic-label"],
    }

    for bin, desc in [(0, "low"), (1, "medium"), (2, "high")]:
        for odds_ratio in ["odds-ratio_topic", "odds-ratio_label", "odds-ratio_topic-label"]:
            input_activation_frame_bin = input_activation_frame[input_activation_frame[odds_ratio + "_bin"] == bin]
            input_correlations_bin = input_activation_frame_bin.corr()
            correlation_entry["input/self-activation/" + odds_ratio + "_" + desc] = input_correlations_bin["self_activation"][odds_ratio]


    dump_correlation_matrix(input_activation_frame, "input", task, split, setting, variation, injection_layer)

    if "RETRO" in setting:

        context_correlations = context_activation_frame.corr()

        correlation_entry["input/self-activation/cross-attention"] = input_correlations["self_activation"]["cross_activation"]
        correlation_entry["input/self-activation/diff-attention"] = input_correlations["self_activation"]["diff_activation"]
        correlation_entry["input/cross-activation/diff-attention"] = input_correlations["cross_activation"]["diff_activation"]
        correlation_entry["input/cross-activation/odds-ratio_topic"] = input_correlations["cross_activation"]["odds-ratio_topic"]
        correlation_entry["input/cross-activation/odds-ratio_label"] = input_correlations["cross_activation"]["odds-ratio_label"]
        correlation_entry["input/cross-activation/odds-ratio_split"] = input_correlations["cross_activation"]["odds-ratio_split"]
        correlation_entry["input/cross-activation/odds-ratio_topic-label"] = input_correlations["cross_activation"]["odds-ratio_topic-label"]
        correlation_entry["input/diff-activation/odds-ratio_topic"] = input_correlations["diff_activation"]["odds-ratio_topic"]
        correlation_entry["input/diff-activation/odds-ratio_label"] = input_correlations["diff_activation"]["odds-ratio_label"]
        correlation_entry["input/diff-activation/odds-ratio_split"] = input_correlations["diff_activation"]["odds-ratio_split"]
        correlation_entry["input/diff-activation/odds-ratio_topic-label"] = input_correlations["diff_activation"]["odds-ratio_topic-label"]

        for bin, desc in [(0, "low"), (1, "medium"), (2, "high")]:
            for odds_ratio in ["odds-ratio_topic", "odds-ratio_label", "odds-ratio_topic-label"]:
                input_activation_frame_bin = input_activation_frame[input_activation_frame[odds_ratio + "_bin"] == bin]
                input_correlations_bin = input_activation_frame_bin.corr()
                correlation_entry["input/cross-activation/" + odds_ratio + "_" + desc] = input_correlations_bin["cross_activation"][odds_ratio]
                correlation_entry["input/diff-activation/" + odds_ratio + "_" + desc] = input_correlations_bin["diff_activation"][odds_ratio]

                context_activation_frame_bin = context_activation_frame[context_activation_frame[odds_ratio + "_bin"] == bin]
                context_correlations_bin = context_activation_frame_bin.corr()
                correlation_entry["context/self-activation/" + odds_ratio + "_" + desc] = context_correlations_bin["self_activation"][odds_ratio]
                correlation_entry["context/cross-activation/" + odds_ratio + "_" + desc] = context_correlations_bin["cross_activation"][odds_ratio]
                correlation_entry["context/diff-activation/" + odds_ratio + "_" + desc] = context_correlations_bin["diff_activation"][odds_ratio]


        correlation_entry["context/self-activation/cross-attention"] = context_correlations["self_activation"]["cross_activation"]
        correlation_entry["context/self-activation/diff-attention"] = context_correlations["self_activation"]["diff_activation"]
        correlation_entry["context/self-activation/odds-ratio_topic"] = context_correlations["self_activation"]["odds-ratio_topic"]
        correlation_entry["context/self-activation/odds-ratio_label"] = context_correlations["self_activation"]["odds-ratio_label"]
        correlation_entry["context/self-activation/odds-ratio_split"] = context_correlations["self_activation"]["odds-ratio_split"]
        correlation_entry["context/self-activation/odds-ratio_topic-label"] = context_correlations["self_activation"]["odds-ratio_topic-label"]
        correlation_entry["context/cross-activation/odds-ratio_topic"] = context_correlations["cross_activation"]["odds-ratio_topic"]
        correlation_entry["context/cross-activation/odds-ratio_label"] = context_correlations["cross_activation"]["odds-ratio_label"]
        correlation_entry["context/cross-activation/odds-ratio_split"] = context_correlations["cross_activation"]["odds-ratio_split"]
        correlation_entry["context/cross-activation/odds-ratio_topic-label"] = context_correlations["cross_activation"]["odds-ratio_topic-label"]
        correlation_entry["context/diff-activation/odds-ratio_topic"] = context_correlations["diff_activation"]["odds-ratio_topic"]
        correlation_entry["context/diff-activation/odds-ratio_label"] = context_correlations["diff_activation"]["odds-ratio_label"]
        correlation_entry["context/diff-activation/odds-ratio_split"] = context_correlations["diff_activation"]["odds-ratio_split"]
        correlation_entry["context/diff-activation/odds-ratio_topic-label"] = context_correlations["diff_activation"]["odds-ratio_topic-label"]

    return correlation_entry

def process_task_file(task_file):
    task, split, setting, variation, seed, model_name, param_injection_layer = get_run_params(task_file)

    correlations = []

    input_topic_odds_ratios = {}
    input_label_odds_ratios = {}
    input_split_odds_ratios = {}
    input_topic_label_odds_ratios = {}

    context_topic_odds_ratios = {}
    context_label_odds_ratios = {}
    context_split_odds_ratios = {}
    context_topic_label_odds_ratios = {}

    if param_injection_layer != None:
        injection_layers = [param_injection_layer]
    else:
        injection_layers = [2,5,8,11]

    for injection_layer in injection_layers:

        samples = load_samples(task, variation, setting, 2, 3, 0, True, "../../data/")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        samples["sentence"] = samples["sentence"].apply(lambda sentence: truncate_sentence(sentence, 100, tokenizer))
        samples["context"] = samples["context"].apply(lambda context: [truncate_sentence(ele, 500, tokenizer) for ele in context])

        task_activations = pandas.DataFrame(json.load(open(task_file, "r"))[0])
        task_activations["topic"] = task_activations["sentence"].apply(lambda sentence: "".join(samples[samples["sentence"] == sentence]["topic"].values))

        if (task, variation) not in input_topic_odds_ratios:
            input_topic_odds_ratios[(task, variation)] = get_odds_ratio(samples, tokenizer=tokenizer, token_col="sentence", target_cols=["topic"])
            input_label_odds_ratios[(task, variation)] = get_odds_ratio(samples, tokenizer=tokenizer, token_col="sentence", target_cols=["label"])
            input_split_odds_ratios[(task, variation)] = get_odds_ratio(samples, tokenizer=tokenizer, token_col="sentence", target_cols=["split"])
            input_topic_label_odds_ratios[(task, variation)] = get_odds_ratio(samples, tokenizer=tokenizer, token_col="sentence", target_cols=["topic", "label"])

            if "RETRO" in setting:
                context_topic_odds_ratios[(task, variation)] = get_odds_ratio(samples, tokenizer=tokenizer, token_col="context", target_cols=["topic"])
                context_label_odds_ratios[(task, variation)] = get_odds_ratio(samples, tokenizer=tokenizer, token_col="context", target_cols=["label"])
                context_split_odds_ratios[(task, variation)] = get_odds_ratio(samples, tokenizer=tokenizer, token_col="context", target_cols=["split"])
                context_topic_label_odds_ratios[(task, variation)] = get_odds_ratio(samples, tokenizer=tokenizer, token_col="context", target_cols=["topic", "label"])

        input_activations = []
        context_activations = []

        input_context_token_co_occurrences = defaultdict(lambda: defaultdict(list))
        context_input_token_co_occurrences = defaultdict(lambda: defaultdict(list))

        for index, row in tqdm(task_activations.iterrows()):

            if row["label"] == row["pred"]:
                result = "correct"
            else:
                result = "wrong"

            topic = samples[samples["sentence"] == row["sentence"]]["topic"].values[0]

            input_tokens = row["input_tokens"]
            input_sep_token_index = input_tokens.index("[SEP]")
            input_tokens = input_tokens[1:input_sep_token_index]

            input_self_activations = normalize_activations(
                row["input_self_activations"][injection_layer][1:input_sep_token_index]
            )

            if "input_cross_activations" in row:
                input_cross_activations = normalize_activations(
                    row["input_cross_activations"][0]
                )
                input_activation = get_full_activations(input_tokens, input_self_activations, input_cross_activations, topic, result, row)
            else:
                input_activation = get_self_activations(input_tokens, input_self_activations, topic, result, row)

            input_activations.extend(input_activation)

            if "context_tokens" in row:

                context_tokens = row["context_tokens"]
                context_sep_token_indices = [tokens.index("[SEP]") for tokens in context_tokens]
                context_tokens = [tokens[1:sep_index] for tokens, sep_index in zip(context_tokens, context_sep_token_indices)]

                context_self_activations = list(itertools.chain.from_iterable(
                    [normalize_activations(ele[injection_layer][1:sep_index]) for ele, sep_index in zip(row["context_self_activations"], context_sep_token_indices)]
                ))

                context_tokens =  list(itertools.chain.from_iterable(context_tokens))

                if "context_cross_activations" in row:
                    context_cross_activations = list(itertools.chain.from_iterable(
                        [normalize_activations(ele[0]) for ele in row["context_cross_activations"]]
                    ))
                    context_activation = get_full_activations(context_tokens, context_self_activations, context_cross_activations, topic, result, row)
                else:
                    context_activation = get_self_activations(context_tokens, context_self_activations, topic, result, row)


                context_activations.extend(context_activation)

                for input_ele, context_ele in itertools.product(input_activation, context_activation):
                    input_token = input_ele["token"]
                    context_token = context_ele["token"]

                    input_diff_activation = input_ele["diff_activation"]
                    context_cross_activation = context_ele["cross_activation"]
                    input_self_activation = input_ele["self_activation"]
                    context_diff_activation = context_ele["diff_activation"]

                    input_context_token_co_occurrences[input_token][context_token].append(numpy.array([input_diff_activation, context_cross_activation]))
                    context_input_token_co_occurrences[context_token][input_token].append(numpy.array([context_diff_activation, input_self_activation]))


        input_activation_frame = pandas.DataFrame(input_activations)
        input_activation_frame["odds-ratio_topic"] = input_activation_frame.apply(lambda row: input_topic_odds_ratios[(task, variation)][row["token"]], axis=1)
        input_activation_frame["odds-ratio_label"] = input_activation_frame.apply(lambda row: input_label_odds_ratios[(task, variation)][row["token"]], axis=1)
        input_activation_frame["odds-ratio_split"] = input_activation_frame.apply(lambda row: input_split_odds_ratios[(task, variation)][row["token"]], axis=1)
        input_activation_frame["odds-ratio_topic-label"] = input_activation_frame.apply(lambda row: input_topic_label_odds_ratios[(task, variation)][row["token"]], axis=1)

        input_activation_frame = input_activation_frame[input_activation_frame["odds-ratio_topic"] != -10000]
        input_activation_frame = input_activation_frame[input_activation_frame["odds-ratio_label"] != -10000]
        input_activation_frame = input_activation_frame[input_activation_frame["odds-ratio_split"] != -10000]
        input_activation_frame = input_activation_frame[input_activation_frame["odds-ratio_topic-label"] != -10000]

        input_activation_frame["odds-ratio_topic_bin"] = pandas.qcut(input_activation_frame["odds-ratio_topic"], q=3, labels=False)
        input_activation_frame["odds-ratio_label_bin"] = pandas.qcut(input_activation_frame["odds-ratio_label"], q=3, labels=False)
        input_activation_frame["odds-ratio_split_bin"] = pandas.qcut(input_activation_frame["odds-ratio_label"], q=3, labels=False)
        input_activation_frame["odds-ratio_topic-label_bin"] = pandas.qcut(input_activation_frame["odds-ratio_topic-label"], q=3, labels=False)

        dump_token_activations(input_activation_frame, source="input", task=task, split=split, variation=variation, injection_layer=str(injection_layer), setting=setting)
        dump_average_token_activations(input_activation_frame, source="input", task=task, split=split, variation=variation, injection_layer=str(injection_layer), setting=setting)

        dump_correlation_matrix(input_activation_frame, "input", task, split, setting, variation, injection_layer)

        if "RETRO" in setting:

            dump_most_relevant_context_tokens(input_context_token_co_occurrences, "input", task, split, setting, variation, str(injection_layer))
            dump_most_relevant_context_tokens(context_input_token_co_occurrences, "context", task, split, setting, variation, str(injection_layer))

            context_activation_frame = pandas.DataFrame(context_activations)

            context_activation_frame["odds-ratio_topic"] = context_activation_frame.apply(lambda row: context_topic_odds_ratios[(task, variation)][row["token"]], axis=1)
            context_activation_frame["odds-ratio_label"] = context_activation_frame.apply(lambda row: context_label_odds_ratios[(task, variation)][row["token"]], axis=1)
            context_activation_frame["odds-ratio_split"] = context_activation_frame.apply(lambda row: context_split_odds_ratios[(task, variation)][row["token"]], axis=1)
            context_activation_frame["odds-ratio_topic-label"] = context_activation_frame.apply(lambda row: context_topic_label_odds_ratios[(task, variation)][row["token"]], axis=1)

            context_activation_frame = context_activation_frame[context_activation_frame["odds-ratio_topic"] != -10000]
            context_activation_frame = context_activation_frame[context_activation_frame["odds-ratio_label"] != -10000]
            context_activation_frame = context_activation_frame[context_activation_frame["odds-ratio_split"] != -10000]
            context_activation_frame = context_activation_frame[context_activation_frame["odds-ratio_topic-label"] != -10000]

            context_activation_frame["odds-ratio_topic_bin"] = pandas.qcut(context_activation_frame["odds-ratio_topic"], q=3, labels=False)
            context_activation_frame["odds-ratio_label_bin"] = pandas.qcut(context_activation_frame["odds-ratio_label"], q=3, labels=False)
            context_activation_frame["odds-ratio_split_bin"] = pandas.qcut(context_activation_frame["odds-ratio_split"], q=3, labels=False)
            context_activation_frame["odds-ratio_topic-label_bin"] = pandas.qcut(context_activation_frame["odds-ratio_topic-label"], q=3, labels=False)

            dump_token_activations(context_activation_frame, source="context", task=task, variation=variation, injection_layer=str(injection_layer), setting=setting, split=split)
            dump_average_token_activations(context_activation_frame, source="context", task=task, variation=variation, injection_layer=str(injection_layer), setting=setting, split=split)
            dump_correlation_matrix(context_activation_frame, "context", task, split, setting, variation, injection_layer)

            correlation_entry = get_correlation(
                task=task, variation=variation, split=split, setting=setting, injection_layer=injection_layer,
                input_activation_frame=input_activation_frame, context_activation_frame=context_activation_frame
            )

            correct_correlation_entry = get_correlation(
                task=task, variation=variation, split=split, setting=setting, result="correct",injection_layer=injection_layer,
                input_activation_frame=input_activation_frame[input_activation_frame["prediction"] == input_activation_frame["label"]],
                context_activation_frame=context_activation_frame[context_activation_frame["prediction"] == context_activation_frame["label"]],
            )

            wrong_correlation_entry = get_correlation(
                task=task, variation=variation, split=split, setting=setting, result="wrong",injection_layer=injection_layer,
                input_activation_frame=input_activation_frame[input_activation_frame["prediction"] != input_activation_frame["label"]],
                context_activation_frame=context_activation_frame[context_activation_frame["prediction"] != context_activation_frame["label"]]
            )
        else:
            correlation_entry = get_correlation(
                task=task, variation=variation, split=split, setting=setting,injection_layer=injection_layer,
                input_activation_frame=input_activation_frame,
            )
            correct_correlation_entry = get_correlation(
                task=task, variation=variation, split=split, setting=setting, result="correct",injection_layer=injection_layer,
                input_activation_frame=input_activation_frame[input_activation_frame["prediction"] == input_activation_frame["label"]],
            )
            wrong_correlation_entry = get_correlation(
                task=task, variation=variation, split=split, setting=setting, result="wrong",injection_layer=injection_layer,
                input_activation_frame=input_activation_frame[input_activation_frame["prediction"] != input_activation_frame["label"]],
            )

        correlations.append(correlation_entry)
        correlations.append(correct_correlation_entry)
        correlations.append(wrong_correlation_entry)

    dump_correlations(correlations, task, split, setting, variation, injection_layer)

    return "Done"


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="argmin")

    args = parser.parse_args()

    task = args.task

    pool = Pool(10)
    task_files = glob.glob("normalized_activations/" + task + "*__*.json")
    pool.map(process_task_file, task_files)

if __name__ == '__main__':
    main()