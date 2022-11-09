import itertools
import os
from collections import defaultdict
from multiprocessing import Pool

import dropbox as dropbox
import pandas
import pingouin
from mlflow.tracking import MlflowClient
from scipy.stats import wilcoxon, binom_test, chi2
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import SquareTable
from tqdm import tqdm

mlflow_url = "ml_flwow_url"
client = MlflowClient(mlflow_url)

def check_dropbox_file_exists(path, base_folder = "retro-cl-models"):
    dbx = dropbox.Dropbox(app_key="app_key", app_secret="app_secret", oauth2_refresh_token="oauth2_refresh_token", timeout=30000)
    dbx.refresh_access_token()
    try:
        metadata = dbx.files_get_metadata("/" + base_folder + "/"+path)
    except:
        return False
    return True

def McNemarPostHoc(data, field1, field2, exact='25'):
    myCross = pandas.crosstab(data[field1], data[field2])

    C = list(myCross.sum())
    R = list(myCross.sum(axis=1))
    n = sum(C)
    k = len(C)
    McNemarResults = pandas.DataFrame(columns=['chi2', 'df', 'p-appr.','p-appr.adj','n for bin.', 'p-exact','p-exact.adj'])
    df = 1

    # loop for each category
    for i in range(k):

        # create the 2x2 table
        a = myCross.iloc[i,i]
        b = R[i] - a
        c = C[i] - a
        d = n - a - b - c

        # the chi-square approximation
        chiVal = (b - c)**2 / (b + c)
        pVal = chi2.sf(chiVal, df)

        # adding the bonferroni adjustment
        pAdj = pVal * k
        if pAdj > 1:
            pAdj = 1

        if b+c < int(exact):

            # the exact binomial test
            pExact = binom_test(min(b,c), b+c, 1/2, alternative='two-sided')
            pExAdj = pExact * k

            # adding the bonferroni adjustment
            if pExAdj > 1:
                pExAdj = 1

        else:
            pExact = 'n.a.'
            pExAdj = 'n.a.'

        # show results in nice dataframe
        newRow = pandas.Series(data={'chi2': chiVal, 'df': df, 'p-appr.': pVal,'p-appr.adj': pAdj,'n for bin.': b+c,'p-exact': pExact,'p-exact.adj': pExAdj}, name=myCross.index[i])
        McNemarResults = McNemarResults.append(newRow, ignore_index=False)

    return McNemarResults

def download_predictions(run_id: str):
    dbx = dropbox.Dropbox(app_key="app_key", app_secret="app_secret", oauth2_refresh_token="oauth2_refresh_token", timeout=30000)
    #dbx.refresh_access_token()

    local_path = "tmp/" + run_id

    os.system("mkdir -p " + local_path)

    if check_dropbox_file_exists(run_id, base_folder="retro-cl-preds"):
        dbx.files_download_to_file(local_path + "/predictions.csv", "/retro-cl-preds/" + run_id + "/predictions.csv")
    elif check_dropbox_file_exists(run_id, base_folder="multicls-cl-preds"):
        dbx.files_download_to_file(local_path + "/predictions.csv", "/multicls-cl-preds/" + run_id + "/predictions.csv")
    else:
        return pandas.DataFrame()

    predictions = pandas.read_csv(local_path + "/predictions.csv")
    os.system("rm -rf " + local_path)

    return predictions

def convert_string_to_bool(input_object):
    if type(input_object) == bool:
        return input_object
    elif type(input_object) == str:
        if input_object.lower() == "true":
            return True
        elif input_object.lower() == "false":
            return False

def get_results(task):
    experiment = client.get_experiment_by_name(task + "-retro")

    if not experiment:
        return

    runs = []

    for fold in range(3):
        for seed in range(3):
            runs.append(client.search_runs(
                experiment.experiment_id, max_results=5000, filter_string="params.model_name_str='bert-large-uncased' and params.seed='" + str(seed) + "' params.fold='" + str(fold) + "'"
            ))

    results = []
    runs = list(itertools.chain.from_iterable(runs))
    for run in tqdm(runs):

        if "dump_path" in run.data.params:
            dump_path = run.data.params["dump_path"].replace(".//", "")
            if "output" not in dump_path:
                dump_path = "output-" + dump_path
        else:
            continue

        predictions = download_predictions(dump_path)

        if predictions.shape[0] == 0:
            #client.delete_run(run_id=run.info.run_id)
            continue

        synchronised_model = convert_string_to_bool(run.data.params.get("synchronised_model", False))
        shared_model = convert_string_to_bool(run.data.params.get("shared_model", False))
        frozen_model = convert_string_to_bool(run.data.params.get("frozen_model", False))

        eval_f1_macro = max([
                step.value
                for step in client.get_metric_history(run.info.run_id, "eval_f1-macro")
        ])

        result_entry = {
            "setting": run.data.params["setting"],
            "training": run.data.params["training"],
            "input_encoder_layers": run.data.params["input_encoder_cross_attn_layers"] if "RETRO" in run.data.params["setting"] else "",
            "context_encoder_layers": run.data.params["context_encoder_cross_attn_layers"] if "RETRO" in run.data.params["setting"] else "",
            "shared_model": shared_model,
            "frozen_model": frozen_model,
            "synchronised_model": synchronised_model,
            "fold": run.data.params["fold"] if "fold" in run.data.params else 0,
            "is_sorted": run.data.params["sorted"] if "sorted" in run.data.params else False,
            "k": run.data.params["k"],
            "seed": run.data.params["seed"],
            "metric_f1-macro": run.data.metrics.get("test_f1-macro", 0),
            "metric_eval_f1-macro": eval_f1_macro,
            "metric_accuracy": run.data.metrics.get("test_accuracy", 0),
            "variation":run.data.params["variation"] if "variation" in run.data.params else "",
            "model_name":run.data.params["model_name_str"] if "model_name_str" in run.data.params else "bert-base-uncased",
            "predictions": predictions
        }

        for metric, value in run.data.metrics.items():
            if "label" in metric and "test" in metric:
                result_entry["metric_" + metric] = value


        results.append(result_entry)

    result_frame = pandas.DataFrame(results)
    result_frame = result_frame.drop_duplicates(["setting", "seed", "input_encoder_layers", "variation", "fold", "shared_model", "frozen_model", "training"])

    predictions = defaultdict(list)

    if result_frame.shape[0] == 0:
        return None

    p_value_entries = []

    for (training, model_name), grouped_frame in result_frame.groupby(["training", "model_name"]):
        baseline_prediction_frames = grouped_frame[grouped_frame["setting"] == "BASELINE_TOPIC"]["predictions"].values

        # Continue if empty
        if len(baseline_prediction_frames) == 0:
            continue

        baseline_predictions = pandas.concat(baseline_prediction_frames).apply(lambda row: 1 if row["preds"] == row["label"] else 0, axis=1)


        for (setting, is_sorted, k, input_encoder_layers, context_encoder_layers, shared_model, frozen_model, synchronised_model, variation), params_frame in grouped_frame.groupby(["setting", "is_sorted", "k", "input_encoder_layers", "context_encoder_layers", "shared_model", "frozen_model", "synchronised_model", "variation"]):
            params_prediction_frames = params_frame["predictions"].values

            # Continue if empty
            if len(params_prediction_frames) == 0:
                continue

            params_predictions = pandas.concat(params_prediction_frames).apply(lambda row: 1 if row["preds"] == row["label"] else 0, axis=1)


            # Check for exceptions when testing would work
            if len(params_predictions) == 0:
                continue


            testing_frame = pandas.DataFrame([list(baseline_predictions), list(params_predictions)]).T

            table = SquareTable.from_data(testing_frame, shift_zeros=0)
            p_value = table.homogeneity(method="bhapkar").pvalue
            #table = McNemarPostHoc(testing_frame, 0, 1)
            #p_value = table.symmetry().pvalue

            p_value = round(p_value, 4)

            p_value_entries.append({
                "setting": setting,
                "model_name": model_name,
                "frozen_model": frozen_model,
                "variation": variation,
                "training": training,
                "is_sorted":is_sorted,
                "input_encoder_layers": input_encoder_layers,
                "context_encoder_layers": context_encoder_layers,
                "shared_model": shared_model,
                "synchronised_model": synchronised_model,
                "k": k,
                "p_value": p_value,
                "sig": 1 if p_value < 0.05 else 0
            })

    pandas.DataFrame(p_value_entries).to_csv("p-values_" + task + "_full.csv")




if __name__ == '__main__':
    tasks = [
         "bench_100-iac1", "bench_100-semeval2019t7", "bench_100-snopes", "bench_100-argmin", "bench_100-rumor","bench_100-ibmcs","bench_100-emergent", "bench_100-perspectrum",  "bench_100-vast", "bench_100-poldeb", "bench_100-mtsd", "bench_100-wtwt",
         "bench_100-scd", "bench_100-semeval2016task6", "bench_100-arc", "bench_100-fnc1",
    ]


    #for task in tasks:
    #    get_results(task)

    pool = Pool()

    pool.map(get_results, tasks)
