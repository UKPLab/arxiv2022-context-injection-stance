import itertools
import multiprocessing
import os
import re
from collections import defaultdict

import igraph
import networkx
import nltk
import numpy
import pandas
import spacy
from nltk.corpus import stopwords
from spacy.tokens import Doc
from tqdm import tqdm
import sys

nlp = spacy.load("en_core_web_sm")
stop_words = stopwords.words("english")
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


def get_all_relationships(graph, concepts):
    concepts = ", ".join(['"/c/en/' + concept + '"' for concept in concepts])

    query = 'MATCH (c:CONCEPT) where c.concept in [' + concepts + '] with collect(id(c)) as node_ids CALL apoc.algo.cover(node_ids) YIELD rel return rel'
    results = graph.run(query)

    return list(results)


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


def get_conceptnet_candidate(concept_node_id_mapping, node_id_concept_mapping,  edge_data_mapping, sentence, topic):
    concepts = get_concepts(sentence, concept_node_id_mapping)

    found_concept_relations = [
        (node_id_i, node_id_j, edge_data_mapping[(node_id_i, node_id_j)])
        for (concept_i, node_id_i), (concept_j, node_id_j) in itertools.product(concepts, concepts)
        if (node_id_i, node_id_j) in edge_data_mapping and node_id_i != node_id_j
    ]

    cleaned_relations = []

    for source_node_id, target_node_id, relation in found_concept_relations:
        start_concept = node_id_concept_mapping[source_node_id].replace("/c/en/", "")
        end_concept = node_id_concept_mapping[target_node_id].replace("/c/en/", "")

        cleaned_relations.append((
            start_concept, end_concept, relation["text"], relation["weight"]
        ))

    cleaned_relations_frame = pandas.DataFrame(cleaned_relations, columns=["start", "end", "text", "weight"])
    cleaned_relations_frame = cleaned_relations_frame.sort_values("weight", ascending=False)

    return list(cleaned_relations_frame["text"])[:10]


def process_split(split_frame):
    graph = networkx.read_graphml(os.path.join(root_dir, "conceptnet_graph.xml"))
    graph = igraph.Graph.from_networkx(graph)
    concept_node_id_mapping = {row["_nx_name"]: index for index, row in graph.get_vertex_dataframe().iterrows()}
    node_id_concept_mapping = {index: row["_nx_name"] for index, row in graph.get_vertex_dataframe().iterrows()}
    edge_data_mapping = {(source, target): {
        "text": edge_frame["text"].values[0],
        "weight": edge_frame["weight"].values[0],
        "relation": edge_frame["relation"].values[0],
    }
        for (source, target), edge_frame in graph.get_edge_dataframe().groupby(["source", "target"])}
    tqdm.pandas()
    return split_frame.apply(
        lambda row: get_conceptnet_candidate(concept_node_id_mapping, node_id_concept_mapping, edge_data_mapping, row["sentence"], row["topic"]),
        axis=1).values


def main():
    import argparse
    import os
    tqdm.pandas()

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="argmin")
    parser.add_argument("--data_path", type=str, default="../../data")

    args = parser.parse_args()

    global root_dir
    root_dir = args.data_path

    task = args.task

    if os.path.exists(os.path.join(args.data_path, task + "_pure.csv")):
        samples = pandas.read_csv(os.path.join(args.data_path, task + "_pure.csv"))
    elif os.path.exists(os.path.join(args.data_path, task + ".pickle")):
        samples = pandas.read_pickle(os.path.join(args.data_path, task + ".pickle"))
    else:
        sys.exit("No datafile found for task {}".format(task))

    n_cpus = multiprocessing.cpu_count()

    splits = numpy.array_split(samples, n_cpus)

    pool = multiprocessing.Pool(1)

    samples["candidates"] = list(itertools.chain.from_iterable(pool.map(process_split, splits)))

    del samples["sentence_embeddings"]

    samples.to_csv(os.path.join(args.data_path, task + "_with_candidates_conceptnet_pairs.csv"))


if __name__ == '__main__':
    main()
