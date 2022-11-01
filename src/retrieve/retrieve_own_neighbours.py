import pickle
import pandas

from sentence_transformers import util
import torch

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, default="ibm-arg-q-stance")
    parser.add_argument("--data_path", type=str, required=True, default="../../data")
    args = parser.parse_args()
    task = args.task

    samples = pandas.read_pickle(os.path.join(args.data_path, task + ".pickle"))

    # top_k = 11 because the first result will always be the identical (sim=1) to the search query
    candidates = util.semantic_search(torch.Tensor(list(samples["sentence_embeddings"])), torch.Tensor(list(samples["sentence_embeddings"])), top_k=11)

    self_sentences = list(samples["sentence"].values)

    samples["candidates"] = [
        [
            self_sentences[sample_candidate["corpus_id"]]
            for sample_candidate in sample_candidates[1:] # ignore the first candidate because it is identical to the initial search query
        ]
        for sample_candidates in candidates
    ]

    del samples["sentence_embeddings"]

    samples.to_csv(os.path.join(args.data_path, task + "_with_candidates_self.csv"))

if __name__ == '__main__':
    main()