import pickle
import pandas

from sentence_transformers import util
import torch

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,  default="ibm-arg-q-stance")
    parser.add_argument("--data_path", type=str,  default="../../data")
    args = parser.parse_args()
    task = args.task

    causenet_embeddings_path = os.path.join(args.data_path, "cause_sentences_embeddings.pickle")

    ## download https://www.dropbox.com/s/xehyq3vtp68ktty/cause_sentences_embeddings.pickle?dl=0
    sentences = list(pickle.load(open(causenet_embeddings_path, "rb")).items())

    samples = pandas.read_pickle(os.path.join(args.data_path, task + ".pickle"))

    sentence_embeddings = [embeddings for _, embeddings in sentences]

    candidates = util.semantic_search(torch.Tensor(list(samples["sentence_embeddings"])), torch.Tensor(sentence_embeddings), top_k=10)

    cause_sentences = [sentence for sentence, _ in sentences]

    samples["candidates"] = [
        [
            cause_sentences[sample_candidate["corpus_id"]]
            for sample_candidate in sample_candidates
        ]
        for sample_candidates in candidates
    ]

    del samples["sentence_embeddings"]

    samples.to_csv(os.path.join(args.data_path, task + "_with_candidates.csv"))

if __name__ == '__main__':
    main()