import itertools
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy
import pandas

from sentence_transformers import util
import torch
from tqdm import tqdm
import sys
modal_verbs = ["must", "shall", "will", "should", "would", "can", "could", "may", "might"]

def get_causal_concepts(sentence, causal_concepts):
    sentence = " " + sentence
    sentence = sentence.replace(".", " .").replace(",", " ,")\
        .replace("!", " !").replace("?", " ?").replace(":", " :").replace(";", " ;")

    return [
            causal_concept
            for causal_concept in causal_concepts
            if " " + causal_concept + " " in sentence
    ]


def get_causal_concepts_for_sentences(params):
    sentences, causal_concepts = params
    return [
        (sentence.lower(), get_causal_concepts(sentence.lower(), causal_concepts))
        for sentence, _ in tqdm(sentences)
    ]

def filter_candidates(initial_candidates, candidates):
    filtered_candidates = [candidate for candidate, score in candidates if candidate in initial_candidates]
    other_candidates = [candidate for candidate, score in candidates if candidate not in initial_candidates]
    return filtered_candidates + other_candidates


def get_candidates(sentence, causes):
    lower_sentence = " " + sentence.lower().replace(".", " .").replace(",", " ,")\
        .replace("!", " !").replace("?", " ?").replace('"', ' " ').replace("(", " ( ").replace(" ) ", " ) ")

    filtered_candidates = [
        candidates
        for ele, candidates in causes.items()
        if ele in lower_sentence
    ]

    return list(itertools.chain.from_iterable(filtered_candidates))

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,  default="scd")
    parser.add_argument("--data_path", type=str, default="../../data")
    args = parser.parse_args()

    task = args.task
    causenet_jsonl_path = os.path.join(args.data_path, "causenet-precision.jsonl")
    causenet_embeddings_path = os.path.join(args.data_path, "cause_sentences_embeddings.pickle") # cause_sentences_embeddings_all-mpnet-base-v2.pickle
    causes_frame = pandas.read_json(path_or_buf=causenet_jsonl_path, lines=True)

    causes = defaultdict(list)

    for index, row in causes_frame.iterrows():
        relation = row["causal_relation"]

        cause = relation["cause"]["concept"].replace("_", " ")
        effect = relation["effect"]["concept"].replace("_", " ")

        if "t.b ." in cause:
            cause = cause.replace("t.b .", "t.b.")

        if len(cause) < 2 or len(effect) < 2 or row["support"] == 0:
            continue

        if cause in modal_verbs or effect in modal_verbs:
            continue

        causes[cause] += [source["payload"]["sentence"] for source in row["sources"] if "sentence" in source["payload"]]
        causes[effect] += [source["payload"]["sentence"] for source in row["sources"] if "sentence" in source["payload"]]

    ## download https://www.dropbox.com/s/xehyq3vtp68ktty/cause_sentences_embeddings.pickle?dl=0
    casual_sentences = dict(pickle.load(open(causenet_embeddings_path, "rb")).items())

    if os.path.exists(os.path.join(args.data_path, task + ".csv")):
        samples = pandas.read_csv(os.path.join(args.data_path, task + ".csv"))
        samples["sentence_embeddings"] = samples["sentence_embeddings"].apply(eval)
    elif os.path.exists(os.path.join(args.data_path, task + ".pickle")):
        samples = pandas.read_pickle(os.path.join(args.data_path, task + ".pickle"))
    else:
        sys.exit("No datafile found for task {}".format(task))

    samples["candidates"] = samples["sentence"].apply(lambda sentence: get_candidates(sentence, causes))

    filtered_candidates = []

    n_splits = int(samples.shape[0] / 1000) + 1

    for split_frame in tqdm(numpy.array_split(samples, n_splits)):

        all_candidates = list(set(itertools.chain.from_iterable(split_frame["candidates"])))

        candidate_embeddings = [casual_sentences[candidate] for candidate in all_candidates]

        split = list(split_frame["sentence_embeddings"])

        corpus_elements = util.semantic_search(
            torch.Tensor(split), torch.Tensor(candidate_embeddings), top_k=10000
        )


        candidates = [
            [
                all_candidates[candidate["corpus_id"]]
                for candidate in corpus_element[:10]
            ]
            for corpus_element in corpus_elements
        ]

        filtered_candidates += candidates

    samples["candidates"] = filtered_candidates

    del samples["sentence_embeddings"]

    samples.to_csv(os.path.join(args.data_path, task + "_with_candidates_relevant_final.csv"))

if __name__ == '__main__':
    main()