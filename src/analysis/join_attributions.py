import glob

import pandas
from tqdm import tqdm

attribution_folders = glob.glob("./attributions-*/*/*/predictions/")

all_attributions = []

for attribution_folder in tqdm(attribution_folders):
    split_path = attribution_folder.split("/")
    task = split_path[1].replace("/", "attributions-")
    fold = int(split_path[2].replace("fold-", ""))
    seed = int(split_path[3].replace("seed-", ""))

    attribution_files = glob.glob(attribution_folder + "*")

    for attribution_file in attribution_files:
        attribution_set = "train" if "train" in attribution_file else "test"
        attributions = pandas.read_csv(attribution_file)

        for index, attribution_row in attributions.iterrows():
            label = attribution_row["label"]
            prediction = attribution_row["prediction"]
            raw_predictions = eval(attribution_row["raw_predictions"])
            raw_prediction = raw_predictions[prediction]
            raw_label = raw_predictions[label]
            prediction_is_correct = 1 if label == prediction else 0

            sep_counter = 0

            for token, attribution in eval(attribution_row["attribution"]):

                if token == "[SEP]":
                    sep_counter += 1

                if sep_counter == 1:
                    continue

                all_attributions.append({
                    "token": token,
                    "attribution": attribution,
                    "task": task,
                    "fold": fold,
                    "seed": seed,
                    "prediction_is_correct": prediction_is_correct,
                    "label": label,
                    "prediction": prediction,
                    "raw_prediction": raw_prediction,
                    "raw_label": raw_label,
                })

pandas.DataFrame(all_attributions).to_csv("joined_attributions.csv")

print()

