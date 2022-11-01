import argparse
import glob
import os.path

import pandas

from keybert import KeyBERT

keybert_model = KeyBERT()

def check_candidate(candidate):
    tokens = candidate.split(" ")

    if len(tokens) - len(set(tokens)) > int(len(tokens) / 2):
        return False

    return True

def get_best_candidate(sentence, candidates):
    filtered_candidates = [candidate for candidate in candidates if check_candidate(candidate)]
    cleaned_candidates = [candidate.replace("</s>", "").replace("<pad>", "").strip() for candidate in filtered_candidates]
    sorted_candidates = sorted(cleaned_candidates, key=lambda ele: len(ele), reverse=True)

    if len(sorted_candidates) == 0:
        return ""
    else:
        return sorted_candidates[0]

def get_best_prompt(sentence, prompts):
    candidates = [prompt["context"] for prompt in prompts]
    return get_best_candidate(sentence, candidates)

def score_noun_phrases(sentence, noun_phrases):
    if len(noun_phrases) == 0:
        return noun_phrases
    else:
        max_noun_phrase_length = max([len(noun_phrase.split(" ")) for noun_phrase in noun_phrases])
        scored_noun_phrases = keybert_model.extract_keywords(sentence, candidates=noun_phrases, top_n=len(noun_phrases)*10, keyphrase_ngram_range=(1, max_noun_phrase_length))

        return scored_noun_phrases

def get_noun_phrase_prompts(sentence, prompts):
    candidates_frame = pandas.DataFrame([[prompt["np"], prompt["context"]] for prompt in prompts if prompt["prompt_type"] == "np_prompt"], columns=["noun_phrase", "context"])

    noun_phrases = list(candidates_frame["noun_phrase"].unique())

    scored_noun_phrases = score_noun_phrases(sentence, noun_phrases)

    noun_phrase_candidates = {}

    for noun_phrase, frame in candidates_frame.groupby("noun_phrase"):
        noun_phrase_candidates[noun_phrase] = get_best_candidate(sentence, list(frame["context"]))

    sorted_noun_phrase_candidates = [noun_phrase_candidates[noun_phrase] for noun_phrase, _ in scored_noun_phrases if noun_phrase in noun_phrase_candidates]

    return sorted_noun_phrase_candidates


def get_noun_phrase_target_prompts(sentence, prompts):
    candidates_frame = pandas.DataFrame([[prompt["np"], prompt["context"]] for prompt in prompts if prompt["prompt_type"] == "target_np_prompt"], columns=["noun_phrase", "context"])

    noun_phrases = list(candidates_frame["noun_phrase"].unique())

    scored_noun_phrases = score_noun_phrases(sentence, noun_phrases)
    noun_phrase_candidates = {}

    for noun_phrase, frame in candidates_frame.groupby("noun_phrase"):
        noun_phrase_candidates[noun_phrase] = get_best_candidate(sentence, list(frame["context"]))

    sorted_noun_phrase_candidates = [noun_phrase_candidates[noun_phrase] for noun_phrase, _ in scored_noun_phrases if noun_phrase in noun_phrase_candidates]

    return sorted_noun_phrase_candidates



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='argmin')
    parser.add_argument("--variation", type=str, default='pp')
    args = parser.parse_args()

    task = args.task
    variation = args.variation

    task_folder = "../../data/t0-context/" + task

    task = task_folder.split("/")[-1]
    if not os.path.exists(task_folder + "/data_t0" + variation + "_context.jsonl"):
        print(task, "not found")
        return

    retrieved_context = pandas.read_json(task_folder + "/data_t0pp_context.jsonl", lines=True, orient='records')
    retrieved_context = retrieved_context.rename(columns={"text":"sentence", "target": "topic"})

    noun_phrase_context = retrieved_context.apply(lambda row: get_noun_phrase_prompts(row["sentence"], row["target_nounphrase_context"]), axis=1)
    noun_phrase_target_context = retrieved_context.apply(lambda row: get_noun_phrase_target_prompts(row["sentence"], row["target_nounphrase_context"]), axis=1)

    noun_phrase_samples = retrieved_context[["sentence", "topic", "label", "split"]]
    noun_phrase_samples["candidates"] = noun_phrase_context
    noun_phrase_samples.to_csv("../../data/" + task + "_with_candidates_t0" + variation + "_key_np.csv")

    noun_phrase_target_samples = retrieved_context[["sentence", "topic", "label", "split"]]
    noun_phrase_target_samples["candidates"] = noun_phrase_target_context
    noun_phrase_target_samples.to_csv("../../data/" + task + "_with_candidates_t0" + variation + "_key_np_target.csv")


if __name__ == '__main__':
    main()