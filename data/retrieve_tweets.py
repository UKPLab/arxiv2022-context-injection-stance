import tweepy
import pandas as pd
import os
import time

def chunker(seq, size):
    res = []
    for el in seq:
        res.append(el)
        if len(res) == size:
            yield res
            res = []
    if res:
        yield res

# Twitter Developer keys here
# It is CENSORED
consumer_key = 'consumer_key'
consumer_key_secret = 'consumer_key_secret'
access_token = 'access_token'
access_token_secret = 'access_token_secret'

def retrieve_rumor_data(path, api):
    print('Fetching RUMOR data')
    obama_data = {}
    with open(os.path.join(path,'obama-ids-annotations.txt'), 'r') as fp:
        for line in fp:
            items = line.strip().split('\t')
            obama_data[items[0]] = {'tweet_id': items[0], 'label': items[1], 'target': 'Is Barack Obama muslim?',
                                    'text': ''}

    for chunk in chunker(list(obama_data.keys()), 100):
        results = [i._json for i in api.lookup_statuses(chunk, include_entities=False, trim_user=True)]
        for res in results:
            obama_data[res['id_str']]['text'] = res['text']

    df = pd.DataFrame(list(obama_data.values()))
    df = df[df['text'] != '']
    df.to_json(os.path.join(path,'obama.jsonl'), lines=True, orient='records')

    cell_data = {}
    with open(os.path.join(path,'cell-ids-annotations.txt'), 'r') as fp:
        for line in fp:
            items = line.strip().split('\t')
            cell_data[items[0]] = {'tweet_id': items[0], 'label': items[1], 'target': 'Cell phone numbers going public?',
                                    'text': ''}

    for chunk in chunker(list(cell_data.keys()), 100):
        results = [i._json for i in api.lookup_statuses(chunk, include_entities=False, trim_user=True)]
        for res in results:
            cell_data[res['id_str']]['text'] = res['text']

    df = pd.DataFrame(list(cell_data.values()))
    df = df[df['text'] != '']
    df.to_json(os.path.join(path,'cell.jsonl'), lines=True, orient='records')


def retrieve_mtsd_data(path, api):
    print('fetching MTSD data')
    df = pd.read_csv(os.path.join(path, 'all_data_tweet_id.txt'), dtype=str, header=0,
                     names=['tweet_id', 'target1', 'stance1', 'target2', 'stance2', 'split'])
    data = {}
    for item in df.itertuples():
        data[item.tweet_id] = {'tweet_id': item.tweet_id, 'target1': item.target1, 'stance1': item.stance1,
                               'target2': item.target2, 'stance2': item.stance2, 'text': ''}

    for chunk in chunker(list(data.keys()), 100):
        results = [i._json for i in api.lookup_statuses(chunk, include_entities=False, trim_user=True)]
        for res in results:
            data[res['id_str']]['text'] = res['text']
    final_data = []
    for k,v in data.items():
        final_data.append({'text': v['text'], 'label': v['stance1'], 'target': v['target1']})
        final_data.append({'text': v['text'], 'label': v['stance2'], 'target': v['target2']})
    df = pd.DataFrame(final_data)
    df = df[df['text'] != '']
    df.to_json(os.path.join(path, 'mtsd.jsonl'), lines=True, orient='records')


auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rumor_path", default='./rumor')
parser.add_argument("--mtsd_path", default='./mtsd')
args = parser.parse_args()


retrieve_rumor_data(args.rumor_path, api)
time.sleep(900) # Twitter API rate limit
retrieve_mtsd_data(args.mtsd_path, api)

