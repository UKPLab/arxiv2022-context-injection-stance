# Copyright (c) Microsoft. All rights reserved.
# Modified Copyright by Ubiquitous Knowledge Processing (UKP) Lab, Technische Universität Darmstadt
import argparse
import os
import csv
import re
import ast
import json
import zipfile
import shutil
from collections import defaultdict
import pandas as pd
import numpy as np

argmin_label_dict = {
    "NoArgument": None,
    "Argument_for": 0,
    "Argument_against": 1,
}

scd_label_dict = {
    "for": 0,
    "against": 1
}

iac1_label_dict = {
    "pro": 0,
    "anti": 1,
    "other": 2
}

snopes_label_dict = {
    "agree": 0,
    "refute": 1
}

fnc1_label_dict = {
    "agree": 0,
    "disagree": 1,
    "unrelated": 2,
    "discuss": 3
}

semeval2016t6_label_dict = {
    "FAVOR": 0,
    "AGAINST": 1,
    "NONE": 2
}

semeval2019t7_label_dict = {
    "support": 0,
    "deny": 1,
    "query": 2,
    "comment": 3
}

ibmcs_label_dict = {
    "PRO": 0,
    "CON": 1
}

perspectrum_label_dict = {
    "SUPPORT": 0,
    "UNDERMINE": 1
}

arc_label_dict = {
    "agree": 0,
    "disagree": 1,
    "unrelated": 2,
    "discuss": 3
}

wtwt_label_dict = {
    "support": 0,
    "refute": 1,
    "unrelated": 2,
    "comment": 3
}

emergent_label_dict = {
    "for": 0,
    "against": 1,
    "observing": 2
}

# according to original github by Hallawy et al. : label, 0=con, 1=pro, 2=neutral
# we switch such that positive stance is always 0 across our benchmark
vast_label_dict = {
    0: 1,
    1: 0,
    2: 2
}

poldeb_label_dict = {
    "stance1": 0, # pro
    "stance2": 1 # con
}

rumor_label_dict = {
    '11': 0, # | **11** | the tweet endorses the rumor |
    '12': 1, # | **12** | the tweet denies the rumor |
    '14': 2, # | **14** | the tweet is neutral |
    '13': 3, # | **13** | the tweet questions the rumor |
    '0': 4, # | **0** | the tweet is not about the rumor |
    # | **2** | the annotator is undetermined |
}

mtsd_label_dict = {
    'FAVOR': 0,
    'AGAINST': 1,
    'NONE': 2
}


def preprocess_scd(file, output_folder, label_dict):
    # split is based on Schiller SD benchmark
    def read_and_tokenize_zip(file_name):
        X_train, X_dev, X_test = [], [], []
        y_train, y_dev, y_test = [], [], []
        topic_train, topic_dev, topic_test = [], [], []
        split_dict = {
            "abortion": "train",
            "gayRights": "train",
            "marijuana": "dev",
            "obama": "test"
        }
        label_converter = {"Stance=+1": "for", "Stance=-1": "against"}
        with zipfile.ZipFile(file+file_name, 'r') as z:
            z.extractall(file)

            for topic in split_dict.keys():
                topic_files = os.listdir(file+file_name[:-4]+"/"+topic+"/")

                for topic_file in topic_files:
                    if topic_file.endswith(".meta"):
                        continue

                    with open(file+file_name[:-4]+"/"+topic+"/"+topic_file, "r", encoding="ISO-8859-1") as data_f,\
                        open(file+file_name[:-4]+"/"+topic+"/"+topic_file[:-5]+".meta", "r") as meta_f:
                        text = data_f.readlines()
                        if topic_file == "G35.data" and topic == "gayRights":
                            text_label = "for"  # missing label: supports its parent, which is G34 and has stance +1
                        else:
                            text_label = label_converter[meta_f.readlines()[2].rstrip("\n")]

                    # the dataset contains duplicates, sometimes with contradicting labels
                    # -> we take only the first appearance of a sentence
                    if text[0] in X_train or text[0] in X_dev or text[0] in X_test:
                        continue

                    if split_dict[topic] == "train":
                        X_train.extend(text)
                        y_train.append(label_dict[text_label])
                        topic_train.append(topic)
                    elif split_dict[topic] == "dev":
                        X_dev.extend(text)
                        y_dev.append(label_dict[text_label])
                        topic_dev.append(topic)
                    else:
                        X_test.extend(text)
                        y_test.append(label_dict[text_label])
                        topic_test.append(topic)
        try:
            shutil.rmtree(file+file_name[:-4], ignore_errors=True)
        except Exception as e:
            print("Cannot delete folder at " + file+file_name[:-4])
        return X_train, X_dev, X_test, y_train, y_dev, y_test, topic_train, topic_dev, topic_test


    print("\n===================== Start preprocessing file: "+ file + " =====================")

    # read train/dev data
    X_train, X_dev, X_test, y_train, y_dev, y_test, target_train, target_dev, target_test = read_and_tokenize_zip("stance.zip")

    assert len(X_train) == 3251
    assert len(X_dev) == 624
    assert len(X_test) == 964

    data = []
    for x, y, target, split in [(X_train, y_train, target_train, 'train'), (X_dev, y_dev, target_dev, 'dev'), (X_test, y_test, target_test, 'test')]:
        for text, label, target in zip(x, y, target):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_argmin(file, output_folder, label_dict):
    # split is based on Schiller SD benchmark
    # load all into a topic-separated dict with train, dev, test lists
    print("\n===================== Start preprocessing file: "+ file + " =====================")
    gold_files = os.listdir(file)
    X_train, X_dev, X_test = [], [], []
    y_train, y_dev, y_test = [], [], []
    for data_file in gold_files:
        if data_file.endswith(".tsv"):
            topic = data_file.replace(".tsv", "")
            with open(file + data_file, 'r') as f_in:
                reader = csv.reader(f_in, delimiter="\t", quoting=3)
                next(reader, None)
                for row in reader:
                    if label_dict[row[5]] != None:
                        if topic == "death_penalty":
                            X_dev.append((row[0], row[4]))
                            y_dev.append(label_dict[row[5]])
                        elif topic == "school_uniforms" or topic == "gun_control":
                            X_test.append((row[0], row[4]))
                            y_test.append(label_dict[row[5]])
                        else:
                            X_train.append((row[0], row[4]))
                            y_train.append(label_dict[row[5]])

    assert len(X_train) == 6845
    assert len(X_dev) == 1568
    assert len(X_test) == 2726

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x,y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_iac1(file, output_folder, label_dict):
    # split is based on Schiller SD benchmark
    # load all into a topic-separated dict with train, dev, test lists
    def read_and_tokenize_csv(file, label_dict, is_test=False):

        def get_label(id):
            return {
                0: "pro",
                1: "anti",
                2: "other"
            }[id]

        # get sample rows with discussion ids and labels
        sample_dict = defaultdict(list)
        with open(file+"author_stance.csv", "r") as in_f:
            reader = csv.reader(in_f)
            next(reader)
            labels = [0, 0, 0]  # pro con other
            topics_count = []
            for row in reader:
                label_id = np.argmax([row[3], row[4], row[5]])
                labels[label_id] += 1

                topics_count.append(row[0])

                sample_dict[row[1]].append({
                    "author": row[2],
                    "topic": row[0],
                    "label": label_dict[get_label(label_id)]
                })

        # get dicussion_id -> author_sentences dict
        discussion_dict = {}
        for d_id in sample_dict.keys():
            with open(file+"discussions/"+str(d_id)+".json", "r") as in_f:
                discussion_data = json.load(in_f)
                user_posts = defaultdict(list)
                for post in discussion_data[0]:
                    user_posts[post[2]].append(post[3])
                discussion_dict[d_id] = user_posts

        # create X,y dample lists
        X = []
        y = []
        text_lens = []
        for d_id in sample_dict:
            for author_data in sample_dict[d_id]:
                text = " ".join([s for s in discussion_dict[d_id][author_data["author"]]])
                text_lens.append(len(text.split()))
                X.append((author_data["topic"], text))
                y.append(author_data["label"])

        return X, y

    print("\n===================== Start preprocessing file: "+ file + " =====================")

    X, y = read_and_tokenize_csv(file, label_dict, is_test=False)

    # split without topic leakage
    topic_set_dict = {
        "evolution": "train",
        "death penalty": "train",
        "gay marriage": "train",
        "climate change": "dev",
        "gun control": "train",
        "healthcare": "train",
        "abortion": "train",
        "existence of god": "test",
        "communism vs capitalism": "dev",
        "marijuana legalization": "test"
    }

    X_train, X_dev, X_test = [], [], []
    y_train, y_dev, y_test = [], [], []
    for i in range(len(X)):
        if topic_set_dict[X[i][0]] == "train":
            X_train.append(X[i])
            y_train.append(y[i])
        elif topic_set_dict[X[i][0]] == "dev":
            X_dev.append(X[i])
            y_dev.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])

    assert len(X_train) == 4227
    assert len(X_dev) == 454
    assert len(X_test) == 924

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x,y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_snopes(file, output_folder, label_dict):
    # split is based on Schiller SD benchmark
    def read(file, label_dict):
        data_dict = {}
        with open(os.path.join(file, "snopes_corpus_2.csv"), "r") as in_f, open(os.path.join(file, "snopes_train.csv"), "r") as train_split_out, \
                open(os.path.join(file, "snopes_dev.csv"), "r") as dev_split_out, open(os.path.join(file, "snopes_test.csv"), "r") as test_split_out:
            data = csv.reader(in_f, delimiter = ',', quotechar = '"')
            next(data)

            for row in data:
                if row[11] == "nostance" or "no_evidence" in row[12]:
                    continue

                evid_ids = row[12].split("_")[1:]
                fges = re.findall("[0-9]*_{(.+?)}", row[9])
                claim = row[6].lower()

                if len(evid_ids) == 0:
                    continue

                assert len(re.findall("[0-9]*_{(.+?)}", row[9])) == len(re.findall("[0-9]*_{.+?}", row[9])), "Error in regex"

                for evid_id in evid_ids:
                    split_fge = fges[int(evid_id)].lower()


                    if row[0] not in data_dict.keys():
                        data_dict[row[0]] = {
                            "claim": claim,
                            "label": label_dict[row[11]],
                            "evidence": {evid_id: split_fge}
                        }
                    else:
                        data_dict[row[0]]["evidence"][evid_id] = split_fge

            X_train, X_dev, X_test = [], [], []
            y_train, y_dev, y_test = [], [], []

            for line in train_split_out.readlines():
                claim_id, evid_id = line.rstrip().split(",")
                X_train.append((data_dict[claim_id]['claim'], data_dict[claim_id]['evidence'][evid_id]))
                y_train.append(data_dict[claim_id]["label"])

            for line in dev_split_out.readlines():
                claim_id, evid_id = line.rstrip().split(",")
                X_dev.append((data_dict[claim_id]['claim'], data_dict[claim_id]['evidence'][evid_id]))
                y_dev.append(data_dict[claim_id]["label"])

            for line in test_split_out.readlines():
                claim_id, evid_id = line.rstrip().split(",")
                X_test.append((data_dict[claim_id]['claim'], data_dict[claim_id]['evidence'][evid_id]))
                y_test.append(data_dict[claim_id]["label"])

        return X_train, y_train, X_dev, y_dev, X_test, y_test


    print("\n===================== Start preprocessing file: "+ file + " =====================")

    # read and tokenize data
    X_train, y_train, X_dev, y_dev, X_test, y_test = read(file, label_dict)

    assert len(X_train) == 14416
    assert len(X_dev) == 1868
    assert len(X_test) == 3154

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x,y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_fnc1_arc(file, output_folder, label_dict):
    # split is based on Schiller SD benchmark
    def create_data(split_path, data_sents, bodies_lookup, label_dict):
        X = []
        y = []
        with open(split_path, "r") as in_dev_split:
            ids = ast.literal_eval(in_dev_split.readline())
            for id in ids:
                X.append((data_sents[id][0], bodies_lookup[int(data_sents[id][1])]))
                y.append(label_dict[data_sents[id][2]])
        return X, y

    def read_csv_from_split(file, label_dict):
        # train and dev set are not fixed, hence, read from split_file

        if "FNC-1" in file:
            sents_path = os.path.join(file, "train_stances.csv")
            bodies_path = os.path.join(file, "train_bodies.csv")
            dev_split_path = os.path.join(file, "fnc1_dev.csv")
            train_split_path = os.path.join(file, "fnc1_train.csv")
        else: # if ARC corpus is preprocessed
            sents_path = os.path.join(file, "arc_stances_train.csv")
            bodies_path = os.path.join(file, "arc_bodies.csv")
            dev_split_path = os.path.join(file, "arc_dev.csv")
            train_split_path = os.path.join(file, "arc_train.csv")

        with open(sents_path, "r") as in_sents, open(bodies_path, "r") as in_bodies:

            data_sents = csv.reader(in_sents, delimiter = ',', quotechar = '"')
            data_bodies = csv.reader(in_bodies, delimiter = ',', quotechar = '"')
            next(data_bodies)
            next(data_sents)

            # generate lookup for bodies
            bodies_lookup = {int(tpl[0]):tpl[1] for tpl in data_bodies}
            data_sents = list(data_sents)

        X_train, y_train = create_data(train_split_path, data_sents, bodies_lookup, label_dict)
        X_dev, y_dev = create_data(dev_split_path, data_sents, bodies_lookup, label_dict)

        return X_train, y_train, X_dev, y_dev

    def read_csv(file, label_dict):
        # test set is fixed, hence, read from file

        X = []
        y = []

        if "FNC-1" in file:
            sents_path = file+"competition_test_stances.csv"
            bodies_path = file+"competition_test_bodies.csv"
        else: # if ARC corpus is preprocessed
            sents_path = file+"arc_stances_test.csv"
            bodies_path = file + "arc_bodies.csv"

        with open(sents_path, "r") as in_sents, open(bodies_path, "r") as in_bodies:

            data_sents = csv.reader(in_sents, delimiter = ',', quotechar = '"')
            data_bodies = csv.reader(in_bodies, delimiter = ',', quotechar = '"')
            next(data_bodies)
            next(data_sents)

            # generate lookup for bodies
            bodies_lookup = {int(tpl[0]):tpl[1] for tpl in data_bodies}

            # generate train instances
            for row in data_sents:
                if row[2] not in ["unrelated", "discuss", "disagree", "agree"]:
                    continue

                X.append((row[0], bodies_lookup[int(row[1])]))
                y.append(label_dict[row[2]])

            return X, y

    print("\n===================== Start preprocessing file: "+ file + " =====================")

    # read and tokenize data
    X_train, y_train, X_dev, y_dev = read_csv_from_split(file, label_dict)
    X_test, y_test = read_csv(file, label_dict)

    if "FNC-1" in file:
        assert len(X_train) == 42476
        assert len(X_dev) == 7496
        assert len(X_test) == 25413
    else: # if ARC corpus is preprocessed
        assert len(X_train) == 12382
        assert len(X_dev) == 1851
        assert len(X_test) == 3559
    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x,y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })

    write_data_files(data, output_folder)


def preprocess_semeval2016t6(file, output_folder, label_dict):
    # split is based on Schiller SD benchmark
    def read(train_file, trial_file, test_file, split_file, label_dict):
        X, y = [], []

        with open(file+train_file, "r", encoding="ISO-8859-1") as in_train_f,\
        open(file+trial_file, "r", encoding="ISO-8859-1") as in_trial_f,\
        open(file+test_file, "r", encoding="ISO-8859-1") as in_test_f,\
        open(file+split_file, "r") as in_split_file:

            data_train = list(csv.reader(in_train_f, delimiter='\t', quotechar = '"', ))
            data_trial = list(csv.reader(in_trial_f, delimiter='\t', quotechar = '"', ))
            data_test = list(csv.reader(in_test_f, delimiter='\t', quotechar = '"', ))

            for row in in_split_file.readlines():
                data_file, line_number = row.rstrip().split("_")
                line_number = int(line_number)

                if data_file == "train":
                    X.append((data_train[line_number][1], data_train[line_number][2]))
                    y.append(label_dict[data_train[line_number][3]])
                elif data_file == "trial":
                    X.append((data_trial[line_number][1], data_trial[line_number][2]))
                    y.append(label_dict[data_trial[line_number][3]])
                else:
                    X.append((data_test[line_number][1], data_test[line_number][2]))
                    y.append(label_dict[data_test[line_number][3]])

        return X, y

    print("\n===================== Start preprocessing file: "+ file + " =====================")

    # read and tokenize data
    X_train, y_train = read("trainingdata-all-annotations.txt",
                            "trialdata-all-annotations.txt",
                            "testdata-taskA-all-annotations.txt",
                            "semeval2016t6_train.csv", label_dict)
    X_dev, y_dev = read("trainingdata-all-annotations.txt",
                            "trialdata-all-annotations.txt",
                            "testdata-taskA-all-annotations.txt",
                            "semeval2016t6_dev.csv", label_dict)
    X_test, y_test = read("trainingdata-all-annotations.txt",
                            "trialdata-all-annotations.txt",
                            "testdata-taskA-all-annotations.txt",
                            "semeval2016t6_test.csv", label_dict)

    assert len(X_train) == 2497
    assert len(X_dev) == 417
    assert len(X_test) == 1249

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x,y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_semeval2019t7(file, output_folder, label_dict):
    # split is based on Schiller SD benchmark
    def parse_tweets(folder_path):
        # create a dict with key = reply_tweet_id and values = source_tweet_id, source_tweet_text, reply_tweet_text
        tweet_dict = {}
        with zipfile.ZipFile(file+folder_path, 'r') as z:
            for filename in z.namelist():
                if not filename.lower().endswith(".json") or filename.rsplit("/", 1)[1] in ['raw.json', 'structure.json', 'dev-key.json', 'train-key.json']:
                    continue
                with z.open(filename) as f:
                    data = f.read()
                    d = json.loads(data.decode("ISO-8859-1"))

                    if "data" in d.keys(): #reddit
                        if "body" in d['data'].keys(): # reply
                            tweet_dict[d['data']['id']] = d['data']['body']
                        elif "children" in d['data'].keys() and isinstance(d['data']['children'][0], dict):
                            tweet_dict[d['data']['children'][0]['data']['id']] = d['data']['children'][0]['data']['title']
                        else: # source
                            try:
                                tweet_dict[d['data']['children'][0]] = ""
                            except Exception as e:
                                print(e)

                    if "text" in d.keys(): # twitter
                        tweet_dict[str(d['id'])] = d['text']

        return tweet_dict

    def read_and_tokenize_json(file_name, label_dict, tweet_dict):
        X, y = [], []

        with open(file+file_name, "r") as in_f:
            split_dict = json.load(in_f)['subtaskaenglish']

            for tweet_id, label in split_dict.items():
                try:
                    X.append(tweet_dict[tweet_id])
                except:
                    continue
                y.append(label_dict[label])

        return X, y

    def read_and_tokenize_zip(folder_path, set_file, label_dict, tweet_dict):
        X, y = [], []

        with zipfile.ZipFile(file + folder_path, 'r') as z:
            with z.open("rumoureval-2019-training-data/"+set_file) as in_f:
                split_dict = json.load(in_f)['subtaskaenglish']

                for tweet_id, label in split_dict.items():
                    try:
                        X.append(tweet_dict[tweet_id])
                    except:
                        continue
                    y.append(label_dict[label])

            return X, y

    print("\n===================== Start preprocessing file: "+ file + " =====================")

    # read train/dev data
    tweet_dict = parse_tweets("rumoureval-2019-training-data.zip")
    X_train, y_train = read_and_tokenize_zip("rumoureval-2019-training-data.zip", "train-key.json", label_dict, tweet_dict)
    X_dev, y_dev = read_and_tokenize_zip("rumoureval-2019-training-data.zip", "dev-key.json", label_dict, tweet_dict)

    # read test data
    tweet_dict = parse_tweets("rumoureval-2019-test-data.zip")
    X_test, y_test = read_and_tokenize_json("final-eval-key.json", label_dict, tweet_dict)

    assert len(X_train) == 5217
    assert len(X_dev) == 1485
    assert len(X_test) == 1827

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for text, label in zip(x,y):
            data.append({
                'text': text.replace("\n", "").replace("\r", ""),
                'label': label,
                'target': "", # for this dataset there exists no target information to extract
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_ibmcs(filepath, output_folder, label_dict):
    # split is based on Schiller SD benchmark

    def read(file, split_file, label_dict):
        X, y = [], []

        with open(file, "r") as in_f, open(split_file, "r") as split_in:
            data = csv.reader(in_f, delimiter = ',', quotechar = '"')
            data = list(data)

            for split_id in split_in.readlines():
                id = int(split_id.rstrip())
                X.append((data[id][2], data[id][7]))
                y.append(label_dict[data[id][6]])

        return X, y

    print("\n===================== Start preprocessing file: "+ filepath + " =====================")

    # read and tokenize data
    file = os.path.join(filepath, "claim_stance_dataset_v1.csv")
    X_train, y_train = read(file, os.path.join(filepath, "ibmcs_train.csv"), label_dict)
    X_dev, y_dev = read(file, os.path.join(filepath, "ibmcs_dev.csv"), label_dict)
    X_test, y_test = read(file, os.path.join(filepath, "ibmcs_test.csv"), label_dict)

    assert len(X_train) == 935
    assert len(X_dev) == 104
    assert len(X_test) == 1355

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target,text), label in zip(x,y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_perspectrum(file, output_folder, label_dict):
    # split based on Schiller SD benchmark
    def read_and_tokenize_csv(file, label_dict):

        with open(file+"dataset_split_v1.0.json", "r") as split_in, \
                open(file+"perspectrum_with_answers_v1.0.json","r") as claims_in, \
                open(file+"perspective_pool_v1.0.json", "r") as perspectives_in:

            # load files
            data_split = json.load(split_in)
            claims = json.load(claims_in)
            perspectives = json.load(perspectives_in)

            # lookup for perspective ids
            perspectives_dict = {}
            for p in perspectives:
                perspectives_dict[p['pId']] = p['text']

            # init
            X_train, X_dev, X_test = [], [], []
            y_train, y_dev, y_test = [], [], []

            count_sup_cluster = 0
            count_und_cluster = 0

            # fill train/dev/test
            for claim in claims:
                cId = str(claim['cId'])
                for p_cluster in claim['perspectives']:
                    cluster_label = label_dict[p_cluster['stance_label_3']]
                    for pid in p_cluster['pids']:
                        if data_split[cId] == 'train':
                            X_train.append((claim['text'], perspectives_dict[pid]))
                            y_train.append(cluster_label)
                        elif data_split[cId] == 'dev':
                            X_dev.append((claim['text'], perspectives_dict[pid]))
                            y_dev.append(cluster_label)
                        elif data_split[cId] == 'test':
                            X_test.append((claim['text'], perspectives_dict[pid]))
                            y_test.append(cluster_label)
                        else:
                            print("Incorrect set type: "+data_split[claim['cId']])
                    if cluster_label == 1:
                        count_sup_cluster += 1
                    if cluster_label == 0:
                        count_und_cluster += 1

        return X_train, y_train, X_dev, y_dev, X_test, y_test

    print("\n===================== Start preprocessing file: "+ file + " =====================")

    # read and tokenize data
    X_train, y_train, X_dev, y_dev, X_test, y_test = read_and_tokenize_csv(file, label_dict)

    assert len(X_train) == 6978
    assert len(X_dev) == 2071
    assert len(X_test) == 2773

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x,y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_wtwt(file, output_folder, label_dict):
    # the split is based on the paper "Cross-Domain Label-Adaptive Stance Detection" by Hardalov et al. 2021
    topic_mapping = {
        "AET_HUM": "Aetna acquires Humana",
        "ANTM_CI": "Anthem acquires Cigna",
        "CVS_AET": "CVS Health acquires Aetna",
        "CI_ESRX": "Cigna acquires Express Scripts",
        "FOXA_DIS": "Disney acquires 21st Century Fox ",
    }
    print("\n===================== Start preprocessing file: "+ file + " =====================")
    X_train, y_train, X_dev, y_dev, X_test, y_test = [], [], [], [], [], []
    with open(os.path.join(file, "wtwt_with_text.json"), "r") as in_f:
        split_dict = json.load(in_f)

        for data_obj in split_dict:
            topic = data_obj['merger']
            if topic in ["CI_ESRX", "ANTM_CI", "CVS_AET"]:
                X_train.append((topic_mapping[topic], data_obj['text']))
                y_train.append(label_dict[data_obj['stance']])
            if topic in ["AET_HUM"]:
                X_dev.append((topic_mapping[topic], data_obj['text']))
                y_dev.append(label_dict[data_obj['stance']])
            if topic in ["FOXA_DIS"]:
                X_test.append((topic_mapping[topic], data_obj['text']))
                y_test.append(label_dict[data_obj['stance']])

    assert len(X_train) == 25193
    assert len(X_dev) == 7897
    assert len(X_test) == 18194

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x,y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_emergent(file, output_folder, label_dict):
    # splits are self-made in cross-target fashion
    # random split with train: 180, dev: 60, test: 60 (the train, test split were already predefined and
    # # we took 60 topics from train for dev with random selection)
    print("\n===================== Start preprocessing file: "+ file + " =====================")
    split_file = pd.read_csv(os.path.join(file, "emergent_splits.csv"))
    split_file.set_index("claimId", inplace=True)
    data_file = pd.read_csv(os.path.join(file, "url-versions-2015-06-14-clean.csv"))
    data = []
    for item in data_file.itertuples():
        data.append({
            'text': item.articleHeadline,
            'label': label_dict[item.articleHeadlineStance],
            'target': item.claimHeadline,
            'split': split_file.loc[item.claimId].split
        })

    assert len(data) == 2595
    write_data_files(data, output_folder)


def preprocess_rumor(file, output_folder, label_dict):
    # the test split is based on the paper "Cross-Domain Label-Adaptive Stance Detection" by Hardalov et al. 2021
    # the train and dev splits are self-made in cross-target fashion
    print("\n===================== Start preprocessing file: "+ file + " =====================")
    X_train, y_train, X_dev, y_dev, X_test, y_test = [], [], [], [], [], []
    with open(os.path.join(file, 'airfrance.txt'), 'r') as fp:
        for line in fp:
            items = line.strip().split('\t')
            if items[3] != 2 and items[3] != '2': # we ignore the 'annotator undetermined' label
                X_test.append(("Air France mid-air crash photos?", items[2]))
                y_test.append(label_dict[str(items[3])])
    with open(os.path.join(file, 'michelle.txt'), 'r', encoding='cp1252') as fp:
        for line in fp:
            items = line.strip().split('\t')
            if items[3] != 2 and items[3] != '2': # we ignore the 'annotator undetermined' label
                X_dev.append(("Michelle Obama hired too many staff?", items[2]))
                y_dev.append(label_dict[str(items[3])])
    with open(os.path.join(file, 'palin.txt'), 'r', encoding='cp1252') as fp:
        for line in fp:
            items = line.strip().split('\t')
            if items[3] != 2 and items[3] != '2': # we ignore the 'annotator undetermined' label
                X_train.append(("Sarah Palin getting divorced?", items[2]))
                y_train.append(label_dict[str(items[3])])
    cell_df = pd.read_json(os.path.join(file, 'cell.jsonl'), lines=True, orient='records')
    for item in cell_df.itertuples():
        if item.label != 2 and item.label != '2':
            X_train.append((item.target, item.text))
            y_train.append(label_dict[str(item.label)])
    obama_df = pd.read_json(os.path.join(file,'obama.jsonl'), lines=True, orient='records')
    for item in obama_df.itertuples():
        if item.label != 2 and item.label != '2':
            X_train.append((item.target, item.text))
            y_train.append(label_dict[str(item.label)])

    #assert len(X_train) == 3718
    #assert len(X_dev) == 520
    assert len(X_test) == 505
    # assert len(X_train) + len(X_dev) + len(X_test) == 7276
    # unfortunately, we were not able to recollect as many tweets as Hardalov et al. 2021

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x,y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_mtsd(file, output_folder, label_dict):
    # split is self-made in a cross-target fashion
    print("\n===================== Start preprocessing file: "+ file + " =====================")
    df = pd.read_csv(os.path.join(file, 'all_data_tweet_text.csv'), header=0, names=['text', 'target1', 'stance1',
                                                                                       'target2', 'stance2', 'split'])
    final_data = []
    for v in df.itertuples():
        final_data.append({'text': v.text, 'label': v.stance1, 'target': v.target1})
        final_data.append({'text': v.text, 'label': v.stance2, 'target': v.target2})
    df = pd.DataFrame(final_data)
    X_train, y_train, X_dev, y_dev, X_test, y_test = [], [], [], [], [], []
    # train: Donald Trump, Hillary Clinton
    # dev: Ted Cruz
    # test: Bernie Sanders
    for item in df.itertuples():
        if item.text == "" or not item.text:
            continue
        if item.target in ["Donald Trump", "Hilary Clinton"]:
            X_train.append((item.target, item.text))
            y_train.append(label_dict[item.label])
        if item.target in ["Ted Cruz"]:
            X_dev.append((item.target, item.text))
            y_dev.append(label_dict[item.label])
        if item.target in ["Bernie Sanders"]:
            X_test.append((item.target, item.text))
            y_test.append(label_dict[item.label])

    # assert len(X_train) == 3718
    # assert len(X_dev) == 520
    # assert len(X_test) == 1092
    assert len(X_train) + len(X_dev) + len(X_test) == 8910
    # unfortunately we were not able to reproduce the numbers by Hardalov et al. 2021; we are given the full dataset
    # by the original authors

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x,y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_poldeb(file, output_folder, label_dict):
    print("\n===================== Start preprocessing file: "+ file + " =====================")
    # the split is based on the paper "Cross-Domain Label-Adaptive Stance Detection" by Hardalov et al. 2021
    target_folder_mapping = {
        'abortion': 'abortion',
        'creation': 'creationism vs evolution in schools',
        'gayRights': 'adoption of children by same-sex couples',
        'god': 'is there a god',
        'guns': 'armed pilots in airplanes',
        'healthcare': 'public insurance option in US health care'
    }
    dir_listing = os.listdir(file)
    target_folders = [i for i in dir_listing if i in target_folder_mapping]

    def read_post(filepath):
        with open(filepath, 'r', errors='ignore') as fp:
            lines = fp.readlines()
            stance = lines[0].strip().split('=')[1]
            text = lines[3].strip()
            return stance, text

    X_train, y_train, X_dev, y_dev, X_test, y_test = [], [], [], [], [], []
    for folder in target_folders:
        for post in os.listdir(os.path.join(file,folder)):
            stance, text = read_post(os.path.join(file,folder,post))
            if folder in ['healthcare', 'guns', 'gayRights', 'god']:
                X_train.append((target_folder_mapping[folder], text))
                y_train.append(label_dict[stance])
            if folder in ['abortion']:
                X_dev.append((target_folder_mapping[folder], text))
                y_dev.append(label_dict[stance])
            if folder in ['creation']:
                X_test.append((target_folder_mapping[folder], text))
                y_test.append(label_dict[stance])

    assert len(X_train) == 4753
    assert len(X_dev) == 1151
    assert len(X_test) == 1230

    data = []
    for x, y, split in [(X_train, y_train, 'train'), (X_dev, y_dev, 'dev'), (X_test, y_test, 'test')]:
        for (target, text), label in zip(x, y):
            data.append({
                'text': text,
                'label': label,
                'target': target,
                'split': split
            })
    write_data_files(data, output_folder)


def preprocess_vast(file, output_folder, label_dict):
    #  we use the original split by the authors Halaway et al. which is the same as used by Hardalov et al. 2021
    print("\n===================== Start preprocessing file: "+ file + " =====================")
    data = []
    for fp, split in [('vast_train.csv', 'train'), ('vast_dev.csv', 'dev'), ('vast_test.csv', 'test')]:
        df = pd.read_csv(os.path.join(file, fp))
        for i in df.itertuples():
            data.append({
                'text': i.post,
                'label': label_dict[i.label],
                'target': i.new_topic,
                'split': split
            })

    assert len([i for i in data if i["split"] == 'train']) == 13477
    assert len([i for i in data if i["split"] == 'dev']) == 2062
    assert len([i for i in data if i["split"] == 'test']) == 3006

    write_data_files(data, output_folder)


def write_data_files(data, output_folder):
    df = pd.DataFrame(data)
    print('=== Writing to: ' + output_folder + ' ====')
    print('=== Before filtering empty texts: ' + str(len(df)))
    df = df[df["text"] != ""]
    print('=== After filtering empty texts: ' + str(len(df)))
    df.to_json(os.path.join(output_folder, 'data.jsonl'), lines=True, orient='records')


def main():
    parser = argparse.ArgumentParser(description='Preprocessing datasets.')
    parser.add_argument('--root_dir', type=str, default='benchmark_original')
    parser.add_argument('--output_dir', type=str, default='benchmark_processed')
    args = parser.parse_args()
    root = args.root_dir
    output_dir = args.output_dir

    assert os.path.exists(root)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ######################################
    # Stance Detection Tasks
    ######################################

    # snopes
    snopes_path = os.path.join(root, 'Snopes/')
    snopes_output_path = os.path.join(output_dir, 'snopes')
    if not os.path.exists(snopes_output_path):
        os.makedirs(snopes_output_path)
    preprocess_snopes(snopes_path, snopes_output_path, snopes_label_dict)

    # argmin 
    argmin_path = os.path.join(root, 'ArgMin/')
    argmin_output_path = os.path.join(output_dir, 'argmin')
    if not os.path.exists(argmin_output_path):
        os.makedirs(argmin_output_path)
    preprocess_argmin(argmin_path, argmin_output_path, argmin_label_dict)

    # fnc1
    fnc1_path = os.path.join(root, 'FNC-1/')
    fnc1_output_path = os.path.join(output_dir, 'fnc1')
    if not os.path.exists(fnc1_output_path):
        os.makedirs(fnc1_output_path)
    preprocess_fnc1_arc(fnc1_path, fnc1_output_path, fnc1_label_dict)

    # arc
    arc_path = os.path.join(root, 'ARC/')
    arc_output_path = os.path.join(output_dir, 'arc')
    if not os.path.exists(arc_output_path):
        os.makedirs(arc_output_path)
    preprocess_fnc1_arc(arc_path, arc_output_path, arc_label_dict)

    # ibmcs
    ibmcs_path = os.path.join(root, 'IBM_CLAIM_STANCE/')
    ibmcs_output_path = os.path.join(output_dir, 'ibmcs')
    if not os.path.exists(ibmcs_output_path):
        os.makedirs(ibmcs_output_path)
    preprocess_ibmcs(ibmcs_path, ibmcs_output_path, ibmcs_label_dict)

    # perspectrum
    perspectrum_path = os.path.join(root, 'PERSPECTRUM/')
    perspectrum_output_path = os.path.join(output_dir, 'perspectrum')
    if not os.path.exists(perspectrum_output_path):
        os.makedirs(perspectrum_output_path)
    preprocess_perspectrum(perspectrum_path, perspectrum_output_path, perspectrum_label_dict)

    # semeval2016t6
    semeval2016t6_path = os.path.join(root, 'SemEval2016Task6/')
    semeval2016t6_output_path = os.path.join(output_dir, 'semeval2016task6')
    if not os.path.exists(semeval2016t6_output_path):
        os.makedirs(semeval2016t6_output_path)
    preprocess_semeval2016t6(semeval2016t6_path, semeval2016t6_output_path, semeval2016t6_label_dict)

    # semeval2017t7
    semeval2019t7_path = os.path.join(root, 'SemEval2019Task7/')
    semeval2019t7_output_path = os.path.join(output_dir, 'semeval2019t7')
    if not os.path.exists(semeval2019t7_output_path):
        os.makedirs(semeval2019t7_output_path)
    preprocess_semeval2019t7(semeval2019t7_path, semeval2019t7_output_path, semeval2019t7_label_dict)

    # scd
    scd_path = os.path.join(root, 'SCD/')
    scd_output_path = os.path.join(output_dir, 'scd')
    if not os.path.exists(scd_output_path):
        os.makedirs(scd_output_path)
    preprocess_scd(scd_path, scd_output_path, scd_label_dict)

    # iac1 
    iac1_path = os.path.join(root, 'IAC/')
    iac1_output_path = os.path.join(output_dir, 'iac1')
    if not os.path.exists(iac1_output_path):
        os.makedirs(iac1_output_path)
    preprocess_iac1(iac1_path, iac1_output_path, iac1_label_dict)

    # wtwt
    wtwt_path = os.path.join(root, 'wtwt/')
    wtwt_output_path = os.path.join(output_dir, 'wtwt')
    if not os.path.exists(wtwt_output_path):
        os.makedirs(wtwt_output_path)
    preprocess_wtwt(wtwt_path, wtwt_output_path, wtwt_label_dict)

    # emergent
    emergent_path = os.path.join(root, 'emergent/')
    emergent_output_path = os.path.join(output_dir, 'emergent/')
    if not os.path.exists(emergent_output_path):
        os.makedirs(emergent_output_path)
    preprocess_emergent(emergent_path, emergent_output_path, emergent_label_dict)

    # rumor
    rumor_path = os.path.join(root, 'rumor/')
    rumor_output_path = os.path.join(output_dir, 'rumor/')
    if not os.path.exists(rumor_output_path):
        os.makedirs(rumor_output_path)
    preprocess_rumor(rumor_path, rumor_output_path, rumor_label_dict)

    # mtsd
    mtsd_path = os.path.join(root, 'mtsd/')
    mtsd_output_path = os.path.join(output_dir, 'mtsd/')
    if not os.path.exists(mtsd_output_path):
        os.makedirs(mtsd_output_path)
    preprocess_mtsd(mtsd_path, mtsd_output_path, mtsd_label_dict)

    # poldeb
    poldeb_path = os.path.join(root, 'poldeb/')
    poldeb_output_path = os.path.join(output_dir, 'poldeb')
    if not os.path.exists(poldeb_output_path):
        os.makedirs(poldeb_output_path)
    preprocess_poldeb(poldeb_path, poldeb_output_path, poldeb_label_dict)

    # vast
    vast_path = os.path.join(root, 'vast/')
    vast_output_path = os.path.join(output_dir, 'vast')
    if not os.path.exists(vast_output_path):
        os.makedirs(vast_output_path)
    preprocess_vast(vast_path, vast_output_path, vast_label_dict)

    # # pstance
    # pstance_path = os.path.join(root, 'pstance/') # split ? follow procedure by BeS MDL Benchmark

    # # # kialo 
    # # kialo_path = os.path.join(root, 'kialo/') # split ? follow procedure by BeS MDL Benchmark


if __name__ == '__main__':
    main()
