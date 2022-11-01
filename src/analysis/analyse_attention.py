import os

from mlflow.tracking import MlflowClient
from tqdm import tqdm

mlflow_url = "http://10.176.133.1:8000/" if os.system("ping -c 1 10.176.133.1") is 0 else "http://6.tcp.eu.ngrok.io:12311"
client = MlflowClient(mlflow_url)

experiment = client.get_experiment_by_name("ukp-sent")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.model='bert-base-uncased' and tags.AVERAGE='True'"
)

for run in tqdm(runs):

    seed = run.data.params["seed"]
    fold = run.data.params["fold"]

    if "SINGLE_TOPIC" in run.data.tags:
        run_path = "./attributions-ukp-sent-topic/fold-" + str(fold) + "/seed-" + str(seed) + "/"
    else:
        run_path = "./attributions-ukp-sent/fold-" + str(fold) + "/seed-" + str(seed) + "/"


    os.system("mkdir -p " + run_path)

    artifacts = client.list_artifacts(run_id=run.info.run_id, path="predictions")
    relevant_artifacts = [
        artifact for artifact in artifacts
        if ("test_" in artifact.path or "train_" in artifact.path) and "attribution" in artifact.path
    ]

    for relevant_artifact in relevant_artifacts:
        artifact_set = "train" if "train" in relevant_artifact.path else "test"
        local_path = client.download_artifacts(run_id=run.info.run_id, path=relevant_artifact.path, dst_path=run_path)
