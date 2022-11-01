import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def get_indomain_splits(samples, seed=0, stratify=True, dev_test_size=.15):
    # prepare in-domain splits in a stratified fashion
    indices = list(samples.index)
    labels = list(samples['label'])
    # generate test set
    train_idx, test_idx, train_labels, test_labels = train_test_split(indices, labels, test_size=dev_test_size,
                                                                              shuffle=stratify, random_state=seed,
                                                                              stratify=labels)
    # generate dev set
    train_idx, val_idx, _, _ = train_test_split(train_idx, train_labels, test_size=dev_test_size,
                                                                              shuffle=stratify, random_state=seed,
                                                                              stratify=train_labels)
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(samples)
    return samples.iloc[train_idx], samples.iloc[val_idx], samples.iloc[test_idx]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", required=True, type=str)
parser.add_argument("--cross_target", action="store_true")
args = parser.parse_args()

assert os.path.exists(args.root_path)
if args.cross_target:
    print('Cross-Target')
else:
    print('In-Target')

for task in ['emergent', 'argmin', 'vast', 'poldeb','mtsd', 'rumor', 'wtwt', 'iac1', 'scd', 'semeval2019t7',
             'semeval2016task6', 'perspectrum', 'ibmcs', 'arc', 'fnc1', 'snopes']:
    path = args.root_path + "/" + task + "_with_candidates_conceptnet.csv"
    assert os.path.exists(path)
    samples = pd.read_csv(path).fillna("")
    print('##### Processing: ', task)
    if args.cross_target:
        for split in sorted(samples['split'].unique()):
            print(split, ': ', len(samples[samples['split'] == split]))
    else:
        train_samples, val_samples, test_samples = get_indomain_splits(samples, 0)
        for split,data in [('train', train_samples), ('dev', val_samples), ('test', test_samples)]:
            print(split, ': ', len(data))