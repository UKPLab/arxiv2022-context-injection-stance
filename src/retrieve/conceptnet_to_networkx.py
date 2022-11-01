import networkx
import pandas
from neo4j import GraphDatabase
from nltk.corpus import stopwords
from py2neo import Graph

dump_path = "/Users/tresi/Downloads/conceptnet-assertions-5.7.0.csv"

dump_file = open(dump_path, 'r')

counter = 0

graph = networkx.DiGraph()

stop_words = stopwords.words("english")

while True:
    dump_line = dump_file.readline()

    if not dump_line:
        break

    dump_line_splitted = dump_line.split("\t")

    uri, relation_name, start_concept, end_concept, relation_data = dump_line_splitted

    relation_data = eval(relation_data)

    start_concept_text = start_concept.replace("/c/en/", "").replace("_", " ")
    end_concept_text = end_concept.replace("/c/en/", "").replace("_", " ")

    if "/en/" not in start_concept or "/en/" not in end_concept or "surfaceText" not in relation_data or start_concept_text in stop_words or end_concept_text in stop_words:
        continue
    relation_text = relation_data["surfaceText"].replace("[[", "").replace("]]", "").replace('"', "").replace("\\", "")
    weight = relation_data["weight"]
    relation_name = relation_name.replace("/r/", "")

    graph.add_weighted_edges_from([(start_concept, end_concept, weight)], text=relation_text, relation=relation_name)

    print("added", relation_name, start_concept, end_concept)

networkx.write_graphml(graph, "../../data/conceptnet_graph.xml")
