import os.path

import pandas
import plotly.express as px

tasks = [
    "bench_100-argmin", "bench_100-ibmcs","bench_100-emergent", "bench_100-perspectrum",  "bench_100-vast", "bench_100-poldeb", "bench_100-mtsd", "bench_100-rumor", "bench_100-wtwt",
    "bench_100-iac1", "bench_100-scd", "bench_100-semeval2019t7", "bench_100-semeval2016task6", "bench_100-arc", "bench_100-fnc1", "bench_100-snopes",
]


def read_task_results(task):
    task_results_file = "results_" + task + "_full.csv"

    if os.path.exists(task_results_file):
        task_results = pandas.read_csv(task_results_file)
        task_results = task_results[(task_results["setting"].isin(["BASELINE", "BASELINE_TOPIC", "RETRO_JOINED_TOPIC", "MULTI_CLS_CONCAT_JOINED_TOPIC"]))]

        task_results["task"] = task
        #task_results["input_encoder_layers"] = task_results["input_encoder_layers"].apply(lambda ele: str(ele) i)
        task_results["input_encoder_layers"] = task_results.apply(lambda row: row["setting"] if str(row["input_encoder_layers"]) == "nan" else "Injected at " +  (row["input_encoder_layers"]) , axis=1)
        task_results["variation"] = task_results["variation"].apply(lambda variation: "none" if variation == "pure" else variation)

        #task_results["color"] = task_results.apply(lambda row: row["setting"] + "+" + row["variation"] if "BASELINE" in row["setting"] else row["input_encoder_layers"], axis=1)
        #task_results["grouping"] = task_results.apply(lambda row: row["variation"] if "RETRO" in row["setting"] else row["input_encoder_layers"], axis=1)

        #task_results["setting"] = task_results.apply(lambda row: row["setting"] + "+" + row["variation"], axis=1)
        return task_results
    else:
        return pandas.DataFrame()

results = pandas.concat([
    read_task_results(task)
    for task in tasks
])

filtered_results = results[
   (results["variation"].isin(["none", "relevant_final", "conceptnet", "t0_target"]))
]

fig = px.strip(filtered_results, x='variation', y='full f1-macro ', facet_col="task", facet_col_wrap=3, color="input_encoder_layers")
fig.update_yaxes(matches=None)
fig.show()