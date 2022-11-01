import os
import sys

import numpy
import numpy as np
import pandas
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

en_stop_words = set(stopwords.words('english'))

def jensen_shannon_divergence(p,q):
    return distance.jensenshannon(p, q) ** 2


def get_divergence(D_p, D_q, vectorizer):
    P = vectorizer.transform([D_p]).toarray().squeeze()
    P = P / np.sum(P)
    Q = vectorizer.transform([D_q]).toarray().squeeze()
    Q = Q / np.sum(Q)
    jsd = jensen_shannon_divergence(P,Q)

    return jsd

def jsd_combined_vocab(data, topics, key="topic"):
    n_topics = len(topics)
    jsd_matrix = np.zeros((n_topics, n_topics))
    for i in range(0, n_topics):
        jsd_matrix[i,i] = np.nan
        for j in range(i+1, n_topics):
            # get documents for topics
            topic_i = topics[i]
            topic_j = topics[j]

            D_p = " ".join(data[data[key] == topic_i]["sentence"].values)
            D_q = " ".join(data[data[key] == topic_j]["sentence"].values)

            vectorizer = CountVectorizer(stop_words=en_stop_words, max_features=20000)
            # create shared vocabulary of both topics
            vectorizer.fit([D_p, D_q])
            # count vectors, apply division by length of topic to get word distributions
            # print(vectorizer.transform([D_p]).shape)
            #print(sum(vectorizer.transform([D_p])))
            P = vectorizer.transform([D_p]).toarray().squeeze()
            P = P / np.sum(P)
            Q = vectorizer.transform([D_q]).toarray().squeeze()
            Q = Q / np.sum(Q)
            jsd = jensen_shannon_divergence(P,Q)
            jsd_matrix[i,j] = jsd
            jsd_matrix[j,i] = np.nan
            #print('topic 1:', topics[i], ', topics 2:', topics[j], ', JSD(t1,t2):', jsd)
            #paired_topics.append((topics[i], topics[j]))
    return jsd_matrix

def overall_vocab(data, topics, vectorizer, key="topic"):

    n_topics = len(topics)
    jsd_matrix = np.zeros((n_topics, n_topics))
    for i in range(0, n_topics):
        jsd_matrix[i,i] = np.nan
        for j in range(i+1, n_topics):
            # get documents for topics
            topic_i = topics[i]
            topic_j = topics[j]

            D_p = " ".join(data[data[key] == topic_i]["sentence"].values)
            D_q = " ".join(data[data[key] == topic_j]["sentence"].values)

            jsd_matrix[i,j] = get_divergence(D_p, D_q, vectorizer)
    jsd_matrix[jsd_matrix==0] = np.nan

    return jsd_matrix


def jsd_single_vocab(data, topics, key="topic"):
    n_topics = len(topics)
    jsd_matrix = np.zeros((n_topics, n_topics))
    for i in range(0, n_topics):
        jsd_matrix[i,i] = np.nan
        for j in range(i+1, n_topics):
            # get documents for topics
            topic_i = topics[i]
            topic_j = topics[j]

            D_p = " ".join(data[data[key] == topic_i]["sentence"].values)
            D_q = " ".join(data[data[key] == topic_j]["sentence"].values)

            vectorizer = CountVectorizer(stop_words=en_stop_words, max_features=20000)
            vectorizer.fit([D_p])
            # count vectors, apply division by length of topic to get word distributions
            P = vectorizer.transform([D_p]).toarray().squeeze()
            P = P / np.sum(P)
            Q = vectorizer.transform([D_q]).toarray().squeeze()
            Q = Q / np.sum(Q)
            jsd = jensen_shannon_divergence(P,Q)
            jsd_matrix[i,j] = jsd
            # create vocabulary of the other topic
            vectorizer = CountVectorizer(stop_words=en_stop_words, max_features=20000)
            vectorizer.fit([D_q])
            # count vectors, apply division by length of topic to get word distributions
            P = vectorizer.transform([D_p]).toarray().squeeze()
            P = P / np.sum(P)
            Q = vectorizer.transform([D_q]).toarray().squeeze()
            Q = Q / np.sum(Q)
            jsd = jensen_shannon_divergence(P,Q)
            jsd_matrix[j,i] = jsd
            #print('topic 1:', topics[i], ', topics 2:', topics[j], ', JSD(t1,t2):', jsd)
            #paired_topics.append((topics[i], topics[j]))
    return jsd_matrix
results = []

datasets = ["argmin", "arc", "wtwt", "perspectrum", "rumor", "mtsd"]
kbs = ["conceptnet", "relevant_final"]
orig_path = '../../data/'
n_folds = 3
all_results = []

def read_frame(task, kb):
    if os.path.exists(os.path.join(orig_path, task + "_with_candidates_" + kb + ".csv")):
        frame = pandas.read_csv(os.path.join(orig_path, task + "_with_candidates_" + kb + ".csv"))
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


full_vectorizer = CountVectorizer(stop_words=en_stop_words, max_features=20000)
full_vectorizer.fit([" ".join(full_frame["sentence"].values)])

tasks = list(full_frame["task"].unique())

overall_jsd = overall_vocab(full_frame, tasks , full_vectorizer, key="task")

pandas.DataFrame(overall_jsd, columns=tasks, index=tasks).to_csv("task_jsd.csv")
for dataset in datasets:
    for kb in kbs:

        frame = read_frame(dataset)

        full_data_set = " ".join(frame["sentence"].values)
        vectorizer = CountVectorizer(stop_words=en_stop_words, max_features=20000)
        vectorizer.fit([full_data_set])

        joined_input = " ".join(frame["sentence"].values)
        joined_topics = " ".join(frame["topic"].values)


        topics = list(set(frame['topic']))

        overall_jsd = overall_vocab(frame, topics, full_vectorizer)
        single_jsd = jsd_single_vocab(frame, topics)
        combined_jsd = jsd_combined_vocab(frame, topics)

        pandas.DataFrame(overall_jsd).to_csv("topic_analysis/" + dataset + "_overall.csv")
        pandas.DataFrame(single_jsd).to_csv("topic_analysis/" + dataset + "_single.csv")
        pandas.DataFrame(combined_jsd).to_csv("topic_analysis/" + dataset + "_combined.csv")

        results = {
            "dataset": dataset,
            "overall mean jsd": np.nanmean(overall_jsd),
            "overall std jsd": np.nanstd(overall_jsd),
            "single mean jsd": np.nanmean(single_jsd),
            "single std jsd": np.nanstd(single_jsd),
            "combined mean jsd": np.nanmean(combined_jsd),
            "combined std jsd": np.nanstd(combined_jsd),
            "input-topic jsd": get_divergence(joined_input, joined_topics, vectorizer),
            "average samples per topic": float(frame.groupby("topic").count()["sentence"].mean()),
            "std samples per topic": float(frame.groupby("topic").count()["sentence"].std()),
        }

        train_test_jsds = []

        train_val_frame = frame[frame["split"].isin(["train", "val", "dev"])].reset_index()
        train_test_frame = frame[frame["split"].isin(["train", "test"])].reset_index()
        val_test_frame = frame[frame["split"].isin(["test", "val", "dev"])].reset_index()

        train_dev_jsd = overall_vocab(train_val_frame, ["train", "val"], full_vectorizer, key="split")
        train_test_jsd = overall_vocab(train_test_frame, ["train", "test"], full_vectorizer, key="split")
        dev_test_jsd = overall_vocab(val_test_frame, ["val", "test"], full_vectorizer, key="split")

        results["train-test jsd"] = np.nanmean(train_test_jsd)
        train_test_jsds.append(np.nanmean(train_test_jsd))

        pandas.DataFrame(train_dev_jsd).to_csv("topic_analysis/" + dataset + "_train_val.csv")
        pandas.DataFrame(train_test_jsd).to_csv("topic_analysis/" + dataset + "_train_test.csv")
        pandas.DataFrame(dev_test_jsd).to_csv("topic_analysis/" + dataset + "_val_test.csv")

        results["train-test jsd mean"] = numpy.mean(train_test_jsds)
        results["train-test jsd std"] = numpy.std(train_test_jsds)

        all_results.append(results)

pandas.DataFrame(all_results).to_csv("topic_analysis/overall_topic_analysis.csv")
print()