import itertools

import numpy
import pandas

datasets = ["emergent", "argmin", "vast", "poldeb", "mtsd", "rumor", "wtwt", "iac1", "scd", "semeval2019t7", "semeval2016task6", "perspectrum", "ibmcs", "arc", "fnc1", "snopes"]
kbs = ["conceptnet", "relevant_final", "t0pp_key_np", "t0pp_key_np_target" ]

context_lengths = []

for kb in kbs:
    context_length_entry = {
        "kb": kb
    }

    for dataset in datasets:

        frame = pandas.read_csv("../../data/" + dataset + "_with_candidates_" + kb + ".csv")

        context_length = list(itertools.chain.from_iterable(frame["candidates"].apply(lambda context: [len(sentence.split()) for sentence in eval(context)[:2]]).values))
        context_length_entry[dataset] = round(numpy.mean(context_length), 1)
    context_lengths.append(context_length_entry)

pandas.DataFrame(context_lengths).to_csv("kb_lengths.csv")