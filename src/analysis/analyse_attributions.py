import os

import pandas

def get_attribution_stats(frame):
    stats = [round(frame["attribution"].mean(), 3), round(frame["attribution"].std(), 3)]
    for fold in range(4):
        fold_frame = frame[frame["fold"] == fold]
        stats.append(round(fold_frame["attribution"].mean(), 3))
        stats.append(round(fold_frame["attribution"].std(), 3))

    return stats


attributions = pandas.read_csv("joined_attributions.csv")

attribution_columns = [
    "mean overall attr", "std overall attr",
    "mean fold-0 attr", "std fold-0 attr",
    "mean fold-1 attr", "std fold-1 attr",
    "mean fold-2 attr", "std fold-2 attr",
    "mean fold-3 attr", "std fold-3 attr",
]

for task, task_frame in attributions.groupby("task"):

    token_overall = [
        [token_frame.shape[0], token] + get_attribution_stats(token_frame)
        for token, token_frame in task_frame.groupby(["token"])
    ]

    token_seed = [
        [token_seed_frame.shape[0], token, seed] + get_attribution_stats(token_seed_frame)
        for (token, seed), token_seed_frame in task_frame.groupby(["token", "seed"])
    ]

    token_overall_predictions = [
        [token_pred_frame.shape[0], token, correct, label, pred] + get_attribution_stats(token_pred_frame)
        for (token, correct, label, pred), token_pred_frame in task_frame.groupby(["token", "prediction_is_correct", "label", "prediction"])
    ]

    token_seed_predictions = [
        [token_pred_seed_frame.shape[0], token, seed, correct, label, pred] + get_attribution_stats(token_pred_seed_frame)
        for (token, seed, correct, label, pred), token_pred_seed_frame in task_frame.groupby(["token", "seed", "prediction_is_correct", "label", "prediction"])
    ]

    pandas.DataFrame(token_overall, columns=["count", "token"] + attribution_columns).to_csv("./" + task + "/overall.csv", index=False)
    pandas.DataFrame(token_seed, columns=["count", "token", "seed"] + attribution_columns).to_csv("./" + task + "/seed.csv", index=False)
    pandas.DataFrame(token_overall_predictions, columns=["count", "token", "correct", "label", "pred"] + attribution_columns).to_csv("./" + task + "/overall_prediction.csv", index=False)
    pandas.DataFrame(token_seed_predictions, columns=["count", "token", "seed", "correct", "label", "pred"] + attribution_columns).to_csv("./" + task + "/seed_prediction.csv", index=False)


token_overall = [
    [token_frame.shape[0], token, task.replace("attributions-", "")]  + get_attribution_stats(token_frame)
    for (token, task), token_frame in attributions.groupby(["token", "task"])
]

token_seed = [
    [token_seed_frame.shape[0], token, task.replace("attributions-", ""),  seed] + get_attribution_stats(token_seed_frame)
    for (token, task, seed), token_seed_frame in attributions.groupby(["token", "task", "seed"])
]

token_overall_predictions = [
    [token_pred_frame.shape[0], token, task.replace("attributions-", ""), correct, label, pred]  + get_attribution_stats(token_pred_frame)
    for (token, task, correct, label, pred), token_pred_frame in attributions.groupby(["token", "task", "prediction_is_correct", "label", "prediction"])
]

token_seed_predictions = [
    [token_pred_seed_frame.shape[0], token, task.replace("attributions-", ""), seed, correct, label, pred]  + get_attribution_stats(token_pred_seed_frame)
    for (token, task, seed, correct, label, pred), token_pred_seed_frame in attributions.groupby(["token", "task", "seed", "prediction_is_correct", "label", "prediction"])
]

pandas.DataFrame(token_overall, columns=["count", "token", "task"] + attribution_columns).to_csv("./overall.csv", index=False)
pandas.DataFrame(token_seed, columns=["count", "token",  "task","seed"] + attribution_columns).to_csv("./seed.csv", index=False)
pandas.DataFrame(token_overall_predictions, columns=["count", "token", "task", "correct", "label", "pred"] + attribution_columns).to_csv("./overall_prediction.csv", index=False)
pandas.DataFrame(token_seed_predictions, columns=["count", "token",  "task","seed", "correct", "label", "pred"] + attribution_columns).to_csv("./seed_prediction.csv", index=False)


print()