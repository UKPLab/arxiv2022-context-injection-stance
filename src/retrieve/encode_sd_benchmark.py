import glob

import pandas as pandas
from sentence_transformers import SentenceTransformer


for task_path in glob.glob("../../data/processed_data/*"):
    task = task_path.split("/")[-1]
    samples = pandas.read_json(task_path + "/data.jsonl", lines=True, orient='records')
    samples = samples.rename(columns={"text": "sentence", "target": "topic"})
    samples["sentence"] = samples["sentence"].apply(lambda sentence: sentence.encode('utf-8', errors='replace').decode('utf-8'))
    samples = samples[samples["sentence"].notna()]
    sentences = list(samples["sentence"])
    model = SentenceTransformer("bert-base-uncased")
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=200).tolist()
    sentence_embeddings = {sentence:embedding for sentence, embedding in zip(sentences, embeddings)}

    samples.to_csv("../../data/" + task + "_pure.csv")
    samples["sentence_embeddings"] = samples["sentence"].apply(lambda sentence: sentence_embeddings[sentence])
    samples.to_csv("../../data/" + task + ".csv")
