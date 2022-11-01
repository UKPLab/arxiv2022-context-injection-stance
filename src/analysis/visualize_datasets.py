import os
import sys

import numpy
import numpy as np
import pandas
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE

from train.utils import topic_split_function

import plotly.express as px


datasets = ["ukp-sent", "ibm-evi", "ibm-arg-q-stance", "semevalt6", "perspectrum", "arc", "pstance", "iac"]
orig_path = '../../data/'

n_folds = 3

all_results = []

def read_frame(task):
    if os.path.exists(os.path.join(orig_path, task + ".csv")):
        frame = pandas.read_csv(os.path.join(orig_path, task + ".csv"))
    elif os.path.exists(os.path.join(orig_path, task + ".pickle")):
        frame = pandas.read_pickle(os.path.join(orig_path, task + ".pickle"))
    else:
        sys.exit("No datafile found for task {}".format(task))

    frame["task"] = task

    return frame



full_frame = pandas.concat([
    read_frame(dataset)
    for dataset in datasets
])

model = SentenceTransformer('all-mpnet-base-v2')

full_frame["sbert"] = list(model.encode(full_frame["sentence"].values, show_progress_bar=True, batch_size=200))
full_frame["sentence"] = full_frame["sentence"].apply(lambda sentence: " ".join(sentence.split(" ")[:10]))

tsnet_sbert = TSNE(2, random_state=0).fit_transform(list(full_frame["sbert"].values))

full_frame["tsne_x"] = tsnet_sbert.T[0]
full_frame["tsne_y"] = tsnet_sbert.T[1]

fig = px.scatter(full_frame, x="tsne_x", y="tsne_y", color="task", hover_data=["topic", "sentence"])

fig.write_html("overview.html")

plot_frame = []

for task, task_frame in full_frame.groupby("task"):

    topic_splits = topic_split_function[task](task_frame, n_folds=n_folds)


    for fold, (train_topics, dev_topics, test_topics) in enumerate(topic_splits):
        train_frame = task_frame[task_frame["topic"].isin(train_topics)]
        dev_frame = task_frame[task_frame["topic"].isin(dev_topics)]
        test_frame = task_frame[task_frame["topic"].isin(test_topics)]

        train_frame["split"] = "train"
        dev_frame["split"] = "dev"
        test_frame["split"] = "test"

        train_frame["fold"] = fold
        dev_frame["fold"] = fold
        test_frame["fold"] = fold

        plot_frame.append(train_frame)
        plot_frame.append(dev_frame)
        plot_frame.append(test_frame)

plot_frame = pandas.concat(plot_frame)

fig = px.scatter(
    plot_frame, x="tsne_x", y="tsne_y", color="split", height=2000,
    facet_col="fold", facet_row="task", hover_data=["topic", "sentence"]
)

fig.write_html("folds.html")

print()
