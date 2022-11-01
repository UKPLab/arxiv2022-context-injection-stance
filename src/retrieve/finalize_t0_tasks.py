import glob

import pandas

def check_candidate(candidate):
    tokens = candidate.split(" ")

    if "</s>" not in candidate:
        return False
    if len(tokens) - len(set(tokens)) > int(len(tokens) / 2):
        return False

    return True

def get_best_candidate(candidates):
    filtered_candidates = [candidate for candidate in candidates if check_candidate(candidate)]
    cleaned_candidates = [candidate.replace("</s>", "").replace("<pad>", "").strip() for candidate in filtered_candidates]
    sorted_candidates = sorted(cleaned_candidates, key=lambda ele: len(ele), reverse=True)

    if len(sorted_candidates) == 0:
        return ""
    else:
        return sorted_candidates[0]

def get_best_prompt(prompts):
    candidates = [prompt["context"] for prompt in prompts]
    candidates = [candidate[0] for candidate in candidates if len(candidate) > 0]
    return get_best_candidate(candidates)

def get_nounphrase_prompts(prompts):
    candidates_frame = pandas.DataFrame([[prompt["np"], prompt["context"]] for prompt in prompts if prompt["prompt_type"] == "np_prompt"], columns=["nounphrase", "context"])

    nounphrase_candidates = []

    for nounphrase, frame in candidates_frame.groupby("nounphrase"):
        nounphrase_candidates.append(get_best_candidate(list(frame["context"])))

    return nounphrase_candidates


def get_nounphrase_target_prompts(prompts):
    candidates_frame = pandas.DataFrame([[prompt["np"], prompt["context"]] for prompt in prompts if prompt["prompt_type"] == "target_np_prompt"], columns=["nounphrase", "context"])

    nounphrase_candidates = []

    for nounphrase, frame in candidates_frame.groupby("nounphrase"):
        nounphrase_candidates.append(get_best_candidate(list(frame["context"])))

    return nounphrase_candidates

for task_folder in glob.glob("../../data/t0-context/*"):
    task = task_folder.split("/")[-1]
    retrieved_context = pandas.read_json(task_folder + "/data_t0pp_context.jsonl", lines=True, orient='records')
    retrieved_context = retrieved_context.rename(columns={"text":"sentence", "target": "topic"})
    target_context = retrieved_context["target_context"].apply(get_best_prompt)
    nounphrase_context = retrieved_context["target_nounphrase_context"].apply(get_nounphrase_prompts)
    nounphrase_target_context = retrieved_context["target_nounphrase_context"].apply(get_nounphrase_target_prompts)

    nounphrase_samples = retrieved_context[["sentence", "topic", "label", "split"]]
    nounphrase_samples["candidates"] = nounphrase_context
    nounphrase_samples.to_csv("../../data/" + task + "_with_candidates_t0_np.csv")

    nounphrase_target_samples = retrieved_context[["sentence", "topic", "label", "split"]]
    nounphrase_target_samples["candidates"] = nounphrase_target_context
    nounphrase_target_samples.to_csv("../../data/" + task + "_with_candidates_t0_np_target.csv")

    target_samples = retrieved_context[["sentence", "topic", "label", "split"]]
    target_samples["candidates"] = target_context
    target_samples.to_csv("../../data/" + task + "_with_candidates_t0_target.csv")
