import argparse
import glob

import pandas as pandas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default='../../data/benchmark_processed/')
    parser.add_argument("--context_dir", type=str, default='../../data/benchmark_context/')
    parser.add_argument("--output_dir", type=str, default='../../data/benchmark_final/')
    args = parser.parse_args()

    root_dir = args.root_dir
    context_dir = args.context_dir
    output_dir = args.output_dir

    for task_path in glob.glob(root_dir + "/*"):
        task = task_path.split("/")[-1]
        samples = pandas.read_json(task_path + "/data.jsonl", lines=True, orient='records')
        samples = samples.rename(columns={"text": "sentence", "target": "topic"})
        samples["sentence"] = samples["sentence"].apply(lambda sentence: sentence.encode('utf-8', errors='replace').decode('utf-8'))
        samples = samples[samples["sentence"].notna()]
        samples.to_csv(output_dir + task + "_pure.csv")

        for source in [
            "all", "random", "causenet", "conceptnet", "t0pp-np", "t0pp-np-targ",
        ]:
            context_frame = pandas.read_csv(context_dir + "/" + task + "_with_candidates_" + source + ".csv")

            samples["candidates"] = context_frame["candidates"]
            samples.to_csv(output_dir + task  + "_with_candidates_" + source + ".csv")

if __name__ == '__main__':
    main()