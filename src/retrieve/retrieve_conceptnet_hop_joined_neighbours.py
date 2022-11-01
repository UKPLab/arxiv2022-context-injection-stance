import os
import sys
import itertools
import multiprocessing
import re
from collections import defaultdict

import igraph
import more_itertools
import networkx
import nltk
import numpy
import pandas
import spacy
from pandarallel import pandarallel
from spacy.tokens import Doc
from tqdm import tqdm

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

    if len(phrase) == 0:
        return phrase

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


def find_paths(graph, edge_data_mapping, concepts):
    text_paths = []
    nodes = list(set([i for concept, i in concepts]))
    shortest_paths = graph.shortest_paths(nodes, nodes)
    shortest_paths = {
        nodes[i]:{
            nodes[j]:other_node
            for j, other_node in enumerate(other_nodes)
        }
        for i, other_nodes in enumerate(shortest_paths)
    }
    for (concept_i, i), (concept_j, j) in itertools.permutations(concepts, 2):
        if 0 < shortest_paths[i][j] < 100:
            paths = graph.get_shortest_paths(i, j, weights="weight")

            for path in paths:
                if len(path) == 1:
                    continue

                text_path = []
                summed_weight = 0

                for start_node, end_node in more_itertools.pairwise(path):
                    edge = edge_data_mapping[(start_node, end_node)]

                    text_path.append(edge["text"])
                    summed_weight += edge["weight"]

                average_weight = summed_weight / len(path)

                text_paths.append((text_path[0], text_path, average_weight))


    path_frame = pandas.DataFrame(text_paths, columns=["first_edge", "text", "weight"]).sort_values("weight",
                                                                                                    ascending=False)

    return path_frame


def get_concepts(text, concept_node_id):

    text = decontracted(text)

    tokens = nlp(text)

    ngram_elements = [[
        (ele.text,)
        for ele in tokens
        if ele.pos_ not in ["VERB", "NUM", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "PART", "PRON", "PUNCT",
                            "SCONJ", "SYM"]
    ]]

    for ngram_length in [2, 3, 4, 5, 6]:
        ngram_elements.append(nltk.ngrams([ele.text for ele in tokens], ngram_length))

    ngram_positions = defaultdict(lambda: 0)
    concepts = []

    for ngram_list in ngram_elements:
        for ngram in ngram_list:
            ngram_joined = "_".join(ngram).lower()
            concept = "/c/en/" + ngram_joined
            ngram_positions[ngram_joined] += 1
            if concept in concept_node_id:
                concepts.append((concept, concept_node_id[concept]))

    return concepts

def get_conceptnet_candidates(graph, concept_node_id, edge_data_mapping, sentence, topic):

    sentence_concepts = get_concepts(sentence, concept_node_id)
    topic_concepts = get_concepts(topic, concept_node_id)

    if len(sentence_concepts) == 0 or len(topic_concepts) == 0:
        joined_paths = pandas.DataFrame()
    else:
        joined_paths = pandas.concat([
            find_paths(graph, edge_data_mapping, joined_concepts)
            for joined_concepts in itertools.product(sentence_concepts, topic_concepts)
        ])

    sentence_paths = find_paths(graph, edge_data_mapping, sentence_concepts)

    path_frame = pandas.concat([joined_paths, sentence_paths])

    reduced_path_frame = path_frame.groupby("first_edge").first().sort_values("weight", ascending=False)
    selected_paths = list(
        reduced_path_frame.apply(lambda row: ". ".join(row["text"]).replace("..", "."), axis=1).values)


    return selected_paths

def process_split(split_frame):
    graph = networkx.read_graphml(os.path.join(root_dir, "conceptnet_graph.xml"))
    graph = igraph.Graph.from_networkx(graph)
    concept_node_id = { row["_nx_name"]: index for index, row in graph.get_vertex_dataframe().iterrows()}
    edge_data_mapping = {(source, target): {
            "text": edge_frame["text"].values[0],
            "weight": edge_frame["weight"].values[0],
            "relation": edge_frame["relation"].values[0],
        }
        for (source, target), edge_frame in graph.get_edge_dataframe().groupby(["source", "target"])}
    tqdm.pandas()
    return split_frame.apply(lambda row: get_conceptnet_candidates(graph, concept_node_id, edge_data_mapping, row["sentence"], row["topic"]), axis=1).values


def main():

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,  default="ibm-arg-q-stance")
    parser.add_argument("--data_path", type=str,  default="../../data")
    args = parser.parse_args()
    global root_dir
    root_dir = args.data_path

    task = args.task
    n_cpus = multiprocessing.cpu_count()

    # for task in ["ibm-evi", "ibm-arg-q-stance", "perspectrum", "arc", "vast"]:

    if os.path.exists(os.path.join(args.data_path, task + ".csv")):
        samples = pandas.read_csv(os.path.join(args.data_path, task + ".csv"))
    elif os.path.exists(os.path.join(args.data_path, task + ".pickle")):
        samples = pandas.read_pickle(os.path.join(args.data_path, task + ".pickle"))
    else:
        sys.exit("No datafile found for task {}".format(task))

    splits = numpy.array_split(samples, n_cpus)

    pool = multiprocessing.Pool()

    samples["candidates"] = list(itertools.chain.from_iterable(pool.map(process_split, splits)))

    del samples["sentence_embeddings"]

    samples.to_csv(os.path.join(args.data_path, task + "_with_candidates_conceptnet_hop_joined.csv"))


if __name__ == '__main__':
    main()
