import pandas
from neo4j import GraphDatabase
from py2neo import Graph

dump_path = "/Users/tresi/Downloads/conceptnet-assertions-5.7.0.csv"

neo4j_server = "bolt://10.176.133.24:7687"
neo4j_user = "neo4j"
neo4j_pw = "admin"

graph = Graph(neo4j_server, auth=(neo4j_user, neo4j_pw))

dump_file = open(dump_path, 'r')

counter = 0

importing = False

while True:
    dump_line = dump_file.readline()

    if not dump_line:
        break

    dump_line_splitted = dump_line.split("\t")

    uri, relation_name, start_concept, end_concept, relation_data = dump_line_splitted

    relation_data = eval(relation_data)

    if "/en/" not in start_concept or "/en/" not in end_concept or "surfaceText" not in relation_data:
        continue

    start_text = relation_data["surfaceStart"].lower().replace('"', "").replace("\\", "")
    end_text = relation_data["surfaceEnd"].lower().replace('"', "").replace("\\", "")
    relation_text = relation_data["surfaceText"].replace("[[", "").replace("]]", "").replace('"', "").replace("\\", "")
    weight = relation_data["weight"]
    relation_name = relation_name.replace("/r/", "")

    cypher_query = 'MERGE (start:CONCEPT { concept: "' + start_concept + '"}) MERGE (end:CONCEPT { concept: "' + end_concept + '"}) MERGE (start)-[:' + relation_name + ' {text: "' + relation_text + '", weight: ' + str(weight) + '}]->(end)'

    if relation_name + " " + start_text + " " + end_text == "HasFirstSubevent write a book brainstorm":
        importing = True
    if importing:
        graph.run(cypher_query)
        counter += 1
        print(relation_name, relation_name, start_text, end_text)
    else:
        print("ignoring", relation_name, start_text, end_text)

print("done")