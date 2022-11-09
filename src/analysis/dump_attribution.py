import glob
import os
from collections import defaultdict

import dropbox
import numpy
import pandas
import torch
from captum.attr import LayerIntegratedGradients
from einops import rearrange
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


mlflow_url = "http://10.176.133.1:8000/" if os.system("ping -c 1 10.176.133.1") is 0 else "http://6.tcp.eu.ngrok.io:12311"
client = MlflowClient(mlflow_url)

task = "ukp-sent"
data_folder = "../../data/"

experiment = client.get_experiment_by_name("dump-" + task + "-retro")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.seed='0'"
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        shared_model = eval(shared_model),
        frozen_model = eval(frozen_model),
        synchronised_model = eval(synchronised_model)
    ).to(device)

    encoding_function = encoding_functions[setting]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    samples = samples[samples["split"].isin(["train", "val", "test"])]

    train_samples = samples[samples["split"] == "train"]
    val_samples = samples[samples["split"] == "val"]
    test_samples = samples[samples["split"] == "test"]

    train_dataset = load_tokenized_dataset(train_samples, setting, tokenizer, encoding_function)
    val_dataset = load_tokenized_dataset(val_samples, setting, tokenizer, encoding_function)
    test_dataset = load_tokenized_dataset(test_samples, setting, tokenizer, encoding_function)

    model_path = download_model(dump_path)
    predictions = download_predictions(dump_path)

    if "RETRO" in setting:
        data_collator = RetroDataCollatorWithPadding(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    model_file = glob.glob("tmp/" + dump_path + "/**/**/**/pytorch_model.bin")[0]
    model_state = torch.load(model_file, map_location=device)

    #model_state = torch.load("./tmp/608a8c3a-f23b-11ec-a868-0242ac130002/model/output-608a8c3a-f23b-11ec-a868-0242ac130002/checkpoint-799/pytorch_model.bin", map_location=device)
    #model.load_state_dict(model_state)


    @torch.enable_grad()
    def predict(input_ids, attention_mask, context_input_ids, context_attention_mask, labels):
        model.train()

        preds = model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, context_input_ids=context_input_ids, context_attention_mask=context_attention_mask
        )
        logits = torch.nn.functional.softmax(preds["logits"], dim=1)
        return logits



    test_elements = list(test_dataset)

    token_attributions = defaultdict(lambda: list)
    input_attributions = []

    for start in tqdm(range(0, len(test_elements), 10)):
        indices = torch.tensor(range(start, start + 10))
        batch_samples = test_elements[start:start+10]
        batch = data_collator(batch_samples)

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        context_input_ids = batch["context_input_ids"].to(device)
        context_attention_mask = batch["context_attention_mask"].to(device)

        ref_input_ids = input_ids.clone()
        ref_input_ids[(ref_input_ids != 102) & (ref_input_ids != 101)] = 0

        ref_context_input_ids = context_input_ids.clone()
        ref_context_input_ids[(ref_context_input_ids != 102) & (ref_context_input_ids != 101)] = 0

        model.eval()

        def normalize_attribution(attributions):
            attributions = attributions.sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            attributions = attributions.cpu().detach().numpy()

            return attributions

        def get_attribution_entries(input_ids, attributions, layer):
            entries = []

            for i, (input_id, attribution) in enumerate(zip(input_ids.cpu().detach().numpy(), attributions)):

                input_attribution = [
                    (tokenizer.decode(int(token_id)), token_attribution)
                    for token_id, token_attribution in zip(input_id, attribution)
                    if token_attribution != 0 and tokenizer.decode(int(token_id)) != "[PAD]"
                ]

                entries.append({
                    "layer": layer,
                    "sentence": batch_samples[i]["sentence"],
                    "input_attribution": input_attribution
                })

            return entries


        for i, layer in enumerate(model.input_encoder_layers):

            lig = LayerIntegratedGradients(predict, layer.output)
            attributions, delta = lig.attribute(
                inputs=(input_ids, attention_mask, context_input_ids, context_attention_mask), additional_forward_args=(labels), internal_batch_size=40,
                target=labels, return_convergence_delta=True, baselines=(ref_input_ids, attention_mask, ref_context_input_ids, context_attention_mask)
            )
            attributions = normalize_attribution(attributions)
            attribution_entries = get_attribution_entries(input_ids, attributions, str(i))
            input_attributions += attribution_entries

        lig = LayerIntegratedGradients(predict, model.input_encoder_embeddings)

        attributions, delta = lig.attribute(
            inputs=(input_ids, attention_mask, context_input_ids, context_attention_mask), additional_forward_args=(labels),
            target=labels, return_convergence_delta=True, baselines=(ref_input_ids, attention_mask, ref_context_input_ids, context_attention_mask)
        )

        attributions = normalize_attribution(attributions)
        attribution_entries = get_attribution_entries(input_ids, attributions, "")
        input_attributions += attribution_entries


    pandas.DataFrame(input_attributions).to_csv(
        "attributions_" + task + "__" + setting + "__" + variation + "__" + str(fold) + "__" + str(seed)
    )
