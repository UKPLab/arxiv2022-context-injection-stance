import os
import sys
import itertools
import pickle
import re
import string
from collections import defaultdict

import conceptnet_lite
import networkx
import nltk
import pandas
from conceptnet_lite import Label
from nltk.corpus import stopwords
from pandarallel import pandarallel
from py2neo import Graph

from sentence_transformers import util
import torch
import spacy
from spacy.tokens import Doc
from tqdm import tqdm


stop_words = stopwords.words("english")

nlp = spacy.load("en_core_web_sm")
root_dir = "../../data"


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"cannot", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    if phrase[-1] == "." and phrase[-2] != " ":
        phrase = phrase[:-1] + " ."
    if phrase[-1] == "!" and phrase[-2] != " ":
        phrase = phrase[:-1] + " !"
    if phrase[-1] == "?" and phrase[-2] != " ":
        phrase = phrase[:-1] + " ?"

    return phrase

def custom_tokenizer(text):
    tokens = re.split(r"([!#$%&*+,./:;<=>?@^_`{|}~']+)?\s+", text)
    filtered_tokens = [token for token in tokens if token]
    return Doc(nlp.vocab, filtered_tokens)

nlp.tokenizer = custom_tokenizer

def get_conceptnet_candidate(graph, chunks):

    candidates = []

    for chunk in chunks:
        if len(chunk) < 2:
            continue

        sentence = decontracted(chunk)

        tokens = nlp(sentence)

        ngram_elements = [[
            (ele.text,)
            for ele in tokens
            if ele.pos_ not in ["VERB", "NUM", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "PART", "PRON", "PUNCT", "SCONJ", "SYM"]
        ]]

        for ngram_length in [2, 3, 4, 5, 6]:
            ngram_elements.append(list(nltk.ngrams([ele.text for ele in tokens], ngram_length)))

        concepts = set()

        for ngram_list in ngram_elements:
            for ngram in ngram_list:
                ngram_joined = ("_".join(ngram).lower())
                concept = "/c/en/" + ngram_joined

                if concept in graph.nodes:
                    concepts.add(concept)

        found_concept_relations = [
            (concept_i, concept_j, graph.edges[(concept_i, concept_j)])
            for concept_i, concept_j in itertools.permutations(concepts, r=2)
            if graph.has_edge(concept_i, concept_j)
        ]

        cleaned_relations = []

        for concept_i, concept_j, edge in found_concept_relations:
            start_concept = concept_i.replace("/c/en/", "").replace("_", " ")
            end_concept = concept_j.replace("/c/en/", "").replace("_", " ")

            cleaned_relations.append((
                start_concept, end_concept, edge["relation"], edge["text"], edge["weight"]
            ))

        cleaned_relations_frame = pandas.DataFrame(cleaned_relations, columns=["start", "end", "type", "text", "weight"])
        cleaned_relations_frame = cleaned_relations_frame.sort_values("weight", ascending=False)

        candidates.append(list(cleaned_relations_frame["text"]))

    return candidates


def main():
    import argparse
    import multiprocessing

    CHUNK_SIZE = 32

    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,  default="ibm-arg-q-stance")
    parser.add_argument("--data_path", type=str,  default="../../data")
    args = parser.parse_args()

    task = args.task

    # for task in ["semevalt6",  "perspectrum", "ukp-sent","ibm-evi", "ibm-arg-q-stance",  "arc", "vast"]:

    if os.path.exists(os.path.join(args.data_path, task + ".csv")):
        samples = pandas.read_csv(os.path.join(args.data_path, task + ".csv"))
    elif os.path.exists(os.path.join(args.data_path, task + ".pickle")):
        samples = pandas.read_pickle(os.path.join(args.data_path, task + ".pickle"))
    else:
        sys.exit("No datafile found for task {}".format(task))

    # for task in ["ukp-sent", "ibm-evi", "ibm-arg-q-stance", "semevalt6", "perspectrum", "arc"]:

    samples["sentence_splits"] = samples["sentence"].apply(lambda sentence: sentence.split(" "))

    samples["sentence_chunks"] = samples["sentence_splits"].apply(lambda sentence_split: [" ".join(sentence_split[i:i+CHUNK_SIZE]) for i in range(0, len(sentence_split), CHUNK_SIZE)])

    graph = networkx.read_graphml(os.path.join(root_dir, "conceptnet_graph.xml"))

    samples["candidates"] = samples["sentence_chunks"].parallel_apply(lambda sentence_chunk: get_conceptnet_candidate(graph, sentence_chunk))

    del samples["sentence_embeddings"]

    samples.to_csv(os.path.join(args.data_path, task + "_with_candidates_conceptnet_noun_chunk.csv"))

if __name__ == '__main__':
    main()