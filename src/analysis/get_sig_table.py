import glob
import os

import numpy
import pandas

results = []

ks = ["2", "None", "1", 2, 1]

tasks =  [
    "arc", "iac1", "perspectrum", "poldeb", "scd",
    "emergent", "fnc1",   "snopes", "mtsd", "rumor", "semeval2016task6",
    "semeval2019t7", "wtwt", "argmin", "ibmcs", "vast"
]

for task in tasks:
    result_file = "p-values_bench_100-" + task + "_full.csv"

    if not os.path.exists(result_file):
        continue


    result_frame = pandas.read_csv(result_file)
    #result_frame = result_frame[result_frame["training"] == "CROSS-TOPIC"]
    result_frame["k"] = result_frame["k"].apply(str)
    result_frame = result_frame[result_frame["k"].isin(ks)]

    result_frame = result_frame[result_frame["model_name"] == "bert-base-uncased"]
    result_frame["task"] = task
    result_frame["input_encoder_layers"] = result_frame["input_encoder_layers"].fillna("")
    result_frame["context_encoder_layers"] = result_frame["context_encoder_layers"].fillna("")
    results.append(result_frame)

results_frame = pandas.concat(results)
base_results = results_frame[(results_frame["setting"].isin(["BASELINE", "BASELINE_TOPIC", "RANDOM_INPUT", "RETRO_TOPIC", "RETRO_RANDOM_INPUT"]))]
k_results = results_frame[results_frame["k"] != "None"]

def get_result_row(setting, model, training, input_encoder_layers, context_encoder_layers, frozen_model, shared_model, synchronised_model, frame, variation="",):

    results = {
        "setting": setting,
        "model": model,
        "training": training,
        "variation": variation,
        "input_encoder_layers": input_encoder_layers,
        "context_encoder_layers": context_encoder_layers,
        "frozen_model": frozen_model,
        "shared_model": shared_model,
        "synchronised_model": synchronised_model,
    }

    mean_results = []


    for task, task_results in frame.groupby("task"):

        mean_results.append(task_results.iloc[0]["p_value"])
        mean_result_str = round(task_results.iloc[0]["p_value"] * 100, 1)
        #std_result_str = str(round(task_results.iloc[0]["metric_f1-macro std"] * 100, 1))

        results[task] = mean_result_str

    results["avg"] = round(numpy.mean(mean_results) * 100, 1)

    return results


def get_average_results(setting, model, training, input_encoder_layers, context_encoder_layers, frozen_model, shared_model, synchronised_model, setting_frame, variation=""):

    average_result = {
        "setting": setting,
        "variation": variation,
        "model": model,
        "training": training,
        "input_encoder_layers": input_encoder_layers,
        "context_encoder_layers": context_encoder_layers,
        "frozen_model": frozen_model,
        "shared_model": shared_model,
        "synchronised_model": synchronised_model
    }

    stat_frame = pandas.DataFrame([setting_frame.mean(), setting_frame.std()]).T

    for metric, row in stat_frame.iterrows():
        if metric in average_result or metric == "seed":
            continue
            
        average = round(row[0], 1)
        std = round(row[1], 1)
        average_result[metric] = average
        average_result[metric + " std"] = std

        if metric == "avg":
            average_result[metric + " pretty"] = str(average) + "$\pm$" + str(std)
        else:
            average_result[metric + " pretty"] = str(average)

    return average_result

for k, k_result_frame in k_results.groupby("k"):

    aggregated_base_results = []
    aggregated_k_results = []

    for (setting, model, training, input_encoder_layers, context_encoder_layers, frozen_model, shared_model, synchronised_model), setting_frame in base_results.groupby(["setting", "model_name", "training", "input_encoder_layers", "context_encoder_layers", "frozen_model", "shared_model", "synchronised_model"], dropna=False):
        aggregated_base_results.append(get_result_row(setting, model, training, input_encoder_layers, context_encoder_layers, frozen_model, shared_model, synchronised_model, setting_frame))

    for (setting, model, training, input_encoder_layers, context_encoder_layers, frozen_model, shared_model, synchronised_model, variation), setting_frame in k_result_frame.groupby(["setting", "model_name", "training", "input_encoder_layers", "context_encoder_layers", "frozen_model", "shared_model", "synchronised_model","variation"], dropna=False):
        aggregated_k_results.append(get_result_row(setting, model, training, input_encoder_layers, context_encoder_layers, frozen_model, shared_model, synchronised_model, setting_frame, variation=variation))

    aggregated_base_frame = pandas.DataFrame(aggregated_base_results)
    aggregated_k_frame = pandas.DataFrame(aggregated_k_results)

    average_results = []

    for (setting, model, training, input_encoder_layers, context_encoder_layers, frozen_model, shared_model, synchronised_model), setting_frame in aggregated_base_frame.groupby(["setting", "model", "training", "input_encoder_layers", "context_encoder_layers", "frozen_model", "shared_model", "synchronised_model"], dropna=False):
        average_results.append(get_average_results(setting, model, training, input_encoder_layers, context_encoder_layers, frozen_model, shared_model, synchronised_model, setting_frame))
    for (setting, model, training, input_encoder_layers, context_encoder_layers, frozen_model, shared_model, synchronised_model, variation), setting_frame in aggregated_k_frame.groupby(["setting", "model", "training", "input_encoder_layers", "context_encoder_layers", "frozen_model", "shared_model", "synchronised_model", "variation"], dropna=False):
        average_results.append(get_average_results(setting, model, training, input_encoder_layers, context_encoder_layers, frozen_model, shared_model, synchronised_model, setting_frame, variation=variation))

    #average_results = [ele for ele in average_results if len(ele) > 1]

    result_frame = pandas.DataFrame(average_results)
    result_frame.to_csv("table_sig_" + str(k) + ".csv")

    metric_cols = ["avg pretty"] + [task + " pretty" for task in tasks if task + " pretty" in result_frame.columns]

    other_cols = ["setting", "training", "variation", "input_encoder_layers"]

    result_frame[other_cols + metric_cols].to_latex("table_sig _" + str(k) + ".txt", escape=False)

