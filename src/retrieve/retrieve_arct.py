import glob
import csv
import ast

import pandas
import pandas as pd
from sentence_transformers import SentenceTransformer

data_path = "../../data/arct/"



def read_csv_from_split(base_path):

    train_path = base_path + "train.tsv"
    dev_path = base_path + "dev.tsv"
    test_path = base_path + "test.tsv"

    frames = []

    for split in ["train", "dev", "test"]:
        path_frame = pandas.read_csv(base_path + split + ".tsv", sep="\t")

        if "dev" in split:
            path_frame["split"] = "val"
        else:
            path_frame["split"] = split

        frames.append(path_frame)

    return pandas.concat(frames).reset_index()



parsed_samples = read_csv_from_split(data_path)
parsed_samples = parsed_samples[["warrant0", "warrant1", "correctLabelW0orW1", "reason", "claim", "debateTitle"]]

pure_samples = []
reason_samples = []
claim_samples = []

for index, row in parsed_samples.iterrows():
    row_label = row["correctLabelW0orW1"]

    pure_samples.append({
        "sentence": row["warrant0"],
        "topic": row["debateTitle"],
        "candidates": [row["claim"] + ", because " + row["reason"]],
        "label": 1 if row_label == 0 else 0
    })
    pure_samples.append({
        "sentence": row["warrant1"],
        "topic": row["debateTitle"],
        "candidates": [row["claim"] + ", because " + row["reason"]],
        "label": 1 if row_label == 1 else 0
    })


    claim_samples.append({
        "sentence": row["claim"] + ", so " + row["warrant0"],
        "topic": row["debateTitle"],
        "candidates": [row["reason"]],
        "label": 1 if row_label == 0 else 0
    })
    claim_samples.append({
        "sentence": row["claim"] + ", so " + row["warrant1"],
        "topic": row["debateTitle"],
        "candidates": [row["reason"]],
        "label": 1 if row_label == 1 else 0
    })

    reason_samples.append({
        "sentence": row["warrant0"] + ", so " + row["reason"],
        "topic": row["debateTitle"],
        "candidates": [row["claim"]],
        "label": 1 if row_label == 0 else 0
    })
    reason_samples.append({
        "sentence": row["warrant1"] + ", so " + row["reason"],
        "topic": row["debateTitle"],
        "candidates": [row["claim"]],
        "label": 1 if row_label == 1 else 0
    })

pandas.DataFrame(pure_samples).to_csv("../../data/arct_pure_with_candidates_poc.csv")
pandas.DataFrame(claim_samples).to_csv("../../data/arct_claim_with_candidates.csv")
pandas.DataFrame(reason_samples).to_csv("../../data/arct_reason_with_candidates.csv")
