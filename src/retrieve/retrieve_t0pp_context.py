import os
from nltk.tree import Tree
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import requests
from tqdm import tqdm
import argparse
import torch
en_stop_words = stopwords.words('english')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


# # using the HF Inference API
# API_TOKEN = "api_token"
# API_URL = "https://api-inference.huggingface.co/models/bigscience/T0pp"
# headers = {"Authorization": f"Bearer {API_TOKEN}"}
# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()

def query(prompts, model, tokenizer):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=40
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_tokens

def fake_query(prompts, model, tokenizer):
    if type(prompts) == str:
        return 'PROCESSED: ' + prompts
    else:
        return ['PROCESSED: ' + i for i in prompts]

# 1st strategy:
# 1. identify noun-phrases in the text
# 2. prompt T0++ for prompts"


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
    parser.add_argument("--root", type=str, default='../../data/benchmark/nounphrases')
    parser.add_argument("--task", type=str, default='argmin')
    parser.add_argument("--batchsize", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default='../../data/benchmark/t0_output')
    args = parser.parse_args()

    # create outputdir
    output_dir = os.path.join(args.output_dir, args.task)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create model
    tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
    model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp").to(device)
    # tokenizer = None # AutoTokenizer.from_pretrained("bigscience/T0pp")
    # model = None # AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")

    # read data
    df = pd.read_json(os.path.join(args.root, args.task, 'data_nounphrases.jsonl'), lines=True, orient='records')

    # create contexts based on target
    print('===== generate target prompts =====')
    target_contexts = {}
    targets = list(df['target'].unique())
    for target in tqdm(targets):
        target_contexts[target] = []
        target_context_prompts = get_single_context_prompts(target.lower())
        
        for idx, target_context_prompt in enumerate(target_context_prompts):
            generated_context = query([target_context_prompt], model=model, tokenizer=tokenizer)[0]
            # generated_context = fake_query(target_context_prompt, model=model, tokenizer=tokenizer)
            target_contexts[target].append({'prompt_type': 'target_prompt', 'prompt': target_context_prompt,
                                            'idx': idx, 'context': generated_context})

    # create contexts based on target and noun-phrases
    print('===== generate (target-)nounphrase prompts =====')
    nounphrase_prompts = []
    for i in tqdm(df.itertuples(), total=len(df)):
        nps = i.noun_phrases
        nps = filter_nps(nps, i.target)
        for np in nps:
            np_context_prompts = get_single_context_prompts(np.lower())
            for idx, np_context_prompt in enumerate(np_context_prompts):
                nounphrase_prompts.append(('np_prompt', i.Index, np, idx, np_context_prompt))
            relation_context_prompts = get_relation_context_prompt(np.lower(), i.target.lower())
            for idx, relation_context_prompt in enumerate(relation_context_prompts):
                nounphrase_prompts.append(('target_np_prompt', i.Index, np, idx, relation_context_prompt))

    # batch processing of text generation
    print('===== generate contexts using t0pp =====')
    target_nounphrase_contexts = []
    prompt_types, data_idxs, noun_phrases, context_idxs, prompts = list(zip(*nounphrase_prompts)) # unmerge to process all prompts in batches
    for chunk in tqdm(chunker(prompts, args.batchsize), total=int(args.batchsize/len(prompts))):
        generated_contexts = query(chunk, tokenizer=tokenizer, model=model)
        # generated_contexts = fake_query(chunk, tokenizer=tokenizer, model=model)
        target_nounphrase_contexts.extend(generated_contexts)
    assert len(target_nounphrase_contexts) == len(prompts) == len(data_idxs) \
           == len(context_idxs) == len(noun_phrases) == len(prompt_types)
    data_idx_to_target_nounphrase_context = {}
    for prompt_type, data_idx, np, context_idx, prompt, target_nounphrase_context in zip(prompt_types, data_idxs,
                                                                                         noun_phrases, context_idxs,
                                                                                         prompts,
                                                                                         target_nounphrase_contexts):
        tmp = {'prompt_type': prompt_type, 'np': np, 'idx': context_idx,
               'prompt': prompt, 'context': target_nounphrase_context}
        if data_idx in data_idx_to_target_nounphrase_context:
            data_idx_to_target_nounphrase_context[data_idx].append(tmp)
        else:
            data_idx_to_target_nounphrase_context[data_idx] = [tmp]

    # merge data back again
    print('===== merge data =====')
    final_data = []
    for i in tqdm(df.itertuples(), total=len(df)):
        dp = {
            'text': i.text,
            'target': i.target,
            'label': i.label,
            'split': i.split,
            'target_context': [],
            'target_nounphrase_context': []
        }
        for tpl in target_contexts[i.target]:
            dp['target_context'].append(tpl)
        if i.Index in data_idx_to_target_nounphrase_context:
            for tpl in data_idx_to_target_nounphrase_context[i.Index]:
                dp['target_nounphrase_context'].append(tpl)
        final_data.append(dp)

    assert len(final_data) == len(df)

    pd.DataFrame(final_data).to_json(os.path.join(output_dir, 'data_t0pp_context.jsonl'),
                                            lines=True, orient='records')
    # df.to_json(os.path.join(output_dir, 'data_t0pp_context.jsonl'), lines=True, orient='records')

if __name__ == '__main__':
    main()