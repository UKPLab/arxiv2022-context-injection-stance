import itertools
import pickle
from multiprocessing import Pool

import numpy
import pandas

from sentence_transformers import util
import torch
from tqdm import tqdm

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

def filter_candidates(sentence, sentence_concepts, candidates, candidates_concepts):
    filtered_candidates = pandas.DataFrame([
        (candidate, score, len(set(sentence_concepts).intersection(set(candidates_concepts[candidate.lower()]))))
        for candidate, score in candidates
    ], columns=["candidates", "score", "intersection"])

    filtered_candidates = filtered_candidates.sort_values(["intersection", "score"], ascending=False)

    return list(filtered_candidates["candidates"])

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, default="pstance")
    parser.add_argument("--data_path", required=True, type=str, default="../../data")
    args = parser.parse_args()
    causenet_jsonl_path = os.path.join(args.data_path, "causenet-precision.jsonl")
    causenet_embeddings_path = os.path.join(args.data_path, "cause_sentences_embeddings.pickle") # cause_sentences_embeddings_all-mpnet-base-v2.pickle
    task = args.task

    causes_frame = pandas.read_json(path_or_buf=causenet_jsonl_path, lines=True)

    causal_concepts = set()

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

        causal_concepts.update([cause, effect])

    ## download https://www.dropbox.com/s/xehyq3vtp68ktty/cause_sentences_embeddings.pickle?dl=0
    sentences = list(pickle.load(open(causenet_embeddings_path, "rb")).items())
    # sentences = list(pickle.load(open(, "rb")).items())

    sentence_chunks = numpy.array_split(sentences, 16)

    pool = Pool(8)

    found_causual_concepts = pool.map(
        get_causal_concepts_for_sentences,[
            [sentence_chunk, causal_concepts]
            for sentence_chunk in sentence_chunks
        ]
    )

    found_causual_concepts = dict(itertools.chain.from_iterable(found_causual_concepts))

    # samples = pandas.read_csv("../../data/" + task + "_all-mpnet-base-v2.csv")
    samples = pandas.read_pickle(os.path.join(args.data_path, task + ".pickle"))
    # samples["sentence_embeddings"] = samples["sentence_embeddings"].apply(eval)

    sentence_embeddings = [embeddings for _, embeddings in sentences]

    corpus_elements = util.semantic_search(
        torch.Tensor(list(samples["sentence_embeddings"])), torch.Tensor(sentence_embeddings), top_k=1000
    )
    cause_sentences = [sentence for sentence, _ in sentences]

    candidates = [
        [
            (cause_sentences[candidate["corpus_id"]], candidate["score"])
            for candidate in corpus_element
        ]
        for corpus_element in corpus_elements
    ]


    samples["causual_concepts"] = samples["sentence"].apply(lambda sentence: get_causal_concepts(sentence, causal_concepts))

    samples["candidates"] = [
        filter_candidates(
            sentence=row["sentence"], sentence_concepts=row["causual_concepts"],
            candidates=candidates[index], candidates_concepts=found_causual_concepts
        )[:10]
        for index, row in samples.iterrows()
    ]

    del samples["sentence_embeddings"]

    samples.to_csv(os.path.join(args.data_path, task + "_with_candidates_filtered.csv"))

if __name__ == '__main__':
    main()