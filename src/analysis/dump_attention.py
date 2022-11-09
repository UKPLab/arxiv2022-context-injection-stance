import glob
import os

import dropbox
import numpy
import pandas
import torch
from captum.attr import LayerIntegratedGradients
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from model.retro_bert import ArgRetro
from run import load_samples
from train.encoding import load_tokenized_dataset, encoding_functions
from train.utils import RetroDataCollatorWithPadding


def download_model(run_id: str):
    dbx = dropbox.Dropbox(app_key="app_key", app_secret="app_secret", oauth2_refresh_token="oauth2_refresh_token", timeout=30000)
    #dbx.refresh_access_token()

    local_path = "tmp/" + dump_path

    os.system("mkdir -p " + local_path)
    dbx.files_download_to_file(local_path + "/model.tar.gz", "/retro-cl-models/" + run_id + ".tar.gz")

    os.system("mkdir -p " + local_path + "/model")
    os.system("tar -xf " + local_path + "/model.tar.gz -C " + local_path + "/model")
    os.system("rm " + local_path + "/model.tar.gz")

    return "tmp/" + dump_path + "/model/"


def download_predictions(run_id: str):
    dbx = dropbox.Dropbox(app_key="app_key", app_secret="app_secret", oauth2_refresh_token="oauth2_refresh_token", timeout=30000)
    #dbx.refresh_access_token()

    local_path = "tmp/" + run_id

    try:
        os.system("mkdir -p " + local_path)
        dbx.files_download_to_file(local_path + "/predictions.csv", "/retro-cl-preds/" + run_id + "/predictions.csv")
    except:
        return pandas.DataFrame()

    predictions = pandas.read_csv(local_path + "/predictions.csv")

    return predictions


mlflow_url = "url_to_mlflow
client = MlflowClient(mlflow_url)

task = "ukp-sent"
data_folder = "../../data/"

experiment = client.get_experiment_by_name("dump-" + task + "-retro")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.seed='0'"
)




for run in tqdm(runs):

    dump_path = run.data.params["dump_path"]

    seed = int(run.data.params["seed"])
    fold = int(run.data.params["fold"])
    variation = run.data.params["variation"]
    setting = run.data.params["setting"]
    shared_model = run.data.params["shared_model"]
    frozen_model = run.data.params["frozen_model"]
    synchronised_model = run.data.params["synchronised_model"]
    model_name = run.data.params["model_name_str"]
    retrieval_model_name = run.data.params["retrieval_model_name"]
    input_encoder_cross_attn_layers = run.data.params["input_encoder_cross_attn_layers"]
    context_encoder_cross_attn_layers = run.data.params["context_encoder_cross_attn_layers"]
    k = int(run.data.params["k"])
    is_cross_topic = False

    samples = load_samples(task, variation, setting, k, 3, fold, is_cross_topic, data_folder)
    num_labels = len(set(samples["label"]))


    model = ArgRetro(
        model_name,
        retrieval_model_name = None if retrieval_model_name == 'None' else retrieval_model_name,
        input_encoder_cross_attn_layers=eval(input_encoder_cross_attn_layers),
        context_encoder_cross_attn_layers=eval(context_encoder_cross_attn_layers),
        k = k,
        setting = setting,
        num_labels = num_labels,
        shared_model = bool(shared_model),
        frozen_model = bool(frozen_model),
        synchronised_model = bool(synchronised_model)
    )

    encoding_function = encoding_functions[setting]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    samples = samples[samples["split"].isin(["train", "val", "test"])]

    train_samples = samples[samples["split"] == "train"]
    val_samples = samples[samples["split"] == "val"]
    test_samples = samples[samples["split"] == "test"]

    train_dataset = load_tokenized_dataset(train_samples, setting, tokenizer, encoding_function)
    val_dataset = load_tokenized_dataset(val_samples, setting, tokenizer, encoding_function)
    test_dataset = load_tokenized_dataset(test_samples, setting, tokenizer, encoding_function)

    #model_path = download_model(dump_path)
    predictions = download_predictions(dump_path)

    if "RETRO" in setting:
        data_collator = RetroDataCollatorWithPadding(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #model_file = glob.glob("tmp/" + dump_path + "/**/**/**/pytorch_model.bin")[0]
    #model_state = torch.load(model_file, map_location=torch.device('cpu'))

    model_state = torch.load("./tmp/608a8c3a-f23b-11ec-a868-0242ac130002/model/output-608a8c3a-f23b-11ec-a868-0242ac130002/checkpoint-799/pytorch_model.bin", map_location=torch.device('cpu'))

    test_elements = list(test_dataset)

    for start in range(0, len(test_elements), 10):
        indices = torch.tensor(range(start, start + 10))
        batch_samples = test_elements[start:start+10]
        batch = data_collator(batch_samples)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        context_input_ids = batch["context_input_ids"]
        context_attention_mask = batch["context_attention_mask"]

        preds, all_input_self_attentions, all_input_cross_attentions, all_context_self_attentions, all_context_cross_attentions = model(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True,
            labels=labels, context_input_ids=context_input_ids, context_attention_mask=context_attention_mask
        )

        stacked_input_self_attentions = torch.stack(all_input_self_attentions)
        stacked_context_self_attentions = torch.stack(all_context_self_attentions)
        stacked_input_cross_attentions = torch.stack(all_input_cross_attentions)
        stacked_context_cross_attentions = torch.stack(all_context_cross_attentions)

        print()
