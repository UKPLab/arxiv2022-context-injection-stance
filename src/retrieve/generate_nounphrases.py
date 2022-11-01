import logging
import os
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import requests
from tqdm import tqdm
import argparse
en_stop_words = stopwords.words('english')

def chunker(seq, size):
    # https://stackoverflow.com/questions/434287/how-to-iterate-over-a-list-in-chunks
    res = []
    for el in seq:
        res.append(el)
        if len(res) == size:
            yield res
            res = []
    if res:
        yield res


# 1st strategy:
# 1. identify noun-phrases in the text
# 2. prompt T0++ for prompts"


def extract_phrase(tree_str, label):
    phrases = []
    trees = Tree.fromstring(tree_str)
    for tree in trees:
        for subtree in tree.subtrees():
            if subtree.label() == label:
                t = subtree
                t = ' '.join(t.leaves())
                phrases.append(t)

    return phrases


def filter_nps(nps, target, nr_words=3):
    # filter stop words
    filtered = []
    for i in nps:
        words = i.split(" ")
        words = [j for j in words if j.lower() not in en_stop_words]
        if len(words) > 0 and words != ['']:
            filtered.append(" ".join(words))
    # filter terms equal to the target
    filtered = [i for i in filtered if i.lower() != target]
    # only up to 3 words
    nps = [i for i in filtered if len(i.split(' ')) <= nr_words]
    return nps


def get_single_context_prompts(text):
    return ['define ' + text,
            'describe ' + text,
            'what is the definition of ' + text,
            'explain ' + text
            ]

def get_relation_context_prompt(text_a, text_b):
    # text-target prompts (using noun-phrase aka np)
    return ['relation between ' + text_a + ' and  ' + text_b, # 0
            'how is ' + text_a + ' related to ' + text_b, # 1
            'explain ' + text_a + ' in terms of ' + text_b # 2
            ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='data')
    parser.add_argument("--task", type=str, default='argmin')
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--stanfordcorenlp", type=str, default='stanford-corenlp-4.4.0')
    args = parser.parse_args()

    # create outputdir
    output_dir = os.path.join(args.output_dir, args.task)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Stanford Core NLP
    #stanford_core_nlp = StanfordCoreNLP(args.stanfordcorenlp, timeout=50000, logging_level=logging.DEBUG, quiet=False)
    stanford_core_nlp = StanfordCoreNLP("http://localhost", port=9000)

    # read data
    df = pd.read_json(os.path.join(args.root, args.task, 'data.jsonl'), lines=True, orient='records')
    #df = df.sample(50).reset_index(drop=True)

    # extract noun-phrases
    print('===== extract nounphrases =====')
    noun_phrases = {}
    for i in tqdm(df.itertuples(), total=len(df)):
        if i.text and i.text != "":
            sentence = i.text[:1000] # introduce cutoff due to long processing times
            tree_str = stanford_core_nlp.parse(sentence)
            nps = extract_phrase(tree_str, 'NP')
        else:
            nps = []
        noun_phrases[i.Index] = nps

    # # merge data back again
    print('===== merge data =====')
    final_data = []
    for i in df.itertuples():
        #data_idx, target_nounphrase_context_idx, target_nounphrase_context in tqdm(zip(data_idxs, context_idxs, target_nounphrase_contexts)):
        final_data.append({
            'text': i.text,
            'target': i.target,
            'label': i.label,
            'split': i.split,
            'noun_phrases': noun_phrases[i.Index],
        })

    assert len(final_data) == len(df)

    pd.DataFrame(final_data).to_json(os.path.join(output_dir, 'data_nounphrases.jsonl'),
                                            lines=True, orient='records')
    # df.to_json(os.path.join(output_dir, 'data_t0pp_context.jsonl'), lines=True, orient='records')

if __name__ == '__main__':
    main()