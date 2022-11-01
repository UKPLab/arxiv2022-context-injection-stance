import json
import os
import pathlib
import uuid

import click
import numpy
import pandas
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding

from model import Inject
from train.encoding import encoding_functions, select_context_elements, load_tokenized_dataset
from train.utils import InjectDataCollatorWithPadding, topic_split_function, compute_metrics, seed_all, check_for_run, \
    truncate_sentence, get_indomain_splits


def load_samples(task, variation, setting, k, nfolds, fold, is_cross_topic, data_folder):


    if variation == "":
        samples = pandas.read_csv(data_folder + "/" + task + "_with_candidates.csv").fillna("")
    elif variation == "pure":
        samples = pandas.read_csv(data_folder + "/" + task + "_pure.csv").fillna("")
        samples["candidates"] = ""
    else:
        samples = pandas.read_csv(data_folder + "/" + task + "_with_candidates_" + variation + ".csv").fillna("")

    samples["context"] = samples["candidates"].apply(lambda candidates: select_context_elements(candidates, setting=setting, k=k))
    if "topic" not in samples.columns:
        samples["topic"] = ""
    if "split" in samples.columns:
        # in-target setting
        return samples[["topic", "sentence", "label", "split", "context", "candidates"]]

    train_topics, val_topics, test_topics = topic_split_function[task](samples, n_folds=nfolds)[fold]

    if is_cross_topic:
        samples = samples[samples["topic"].isin(train_topics + val_topics + test_topics)]
        samples["split"] = samples["topic"].apply(
            lambda topic: "train" if topic in train_topics else "val" if topic in val_topics else "test")
    else:
        all_splits = topic_split_function[task](samples, n_folds=nfolds)

        n_train_samples = [samples[samples["topic"].isin(train_topics)].shape[0] for train_topics, val_topics, test_topics in all_splits]
        n_val_samples = [samples[samples["topic"].isin(val_topics)].shape[0] for train_topics, val_topics, test_topics in all_splits]
        n_test_samples = [samples[samples["topic"].isin(test_topics)].shape[0] for train_topics, val_topics, test_topics in all_splits]

        test_fold_splits = []

        samples = samples.sample(frac=1, random_state=0)

        start_index = 0

        for fold_length in n_test_samples:
            test_fold_splits.append(samples.iloc[start_index:start_index + fold_length])
            start_index = start_index + fold_length

        test_samples = test_fold_splits[fold]

        train_size = n_train_samples[fold]
        val_size = n_val_samples[fold]

        other_samples = samples[~samples.index.isin(test_samples.index)]
        train_samples = other_samples.iloc[:train_size]
        val_samples = other_samples.iloc[train_size: train_size + val_size]

        train_samples["split"] = "train"
        val_samples["split"] = "val"
        test_samples["split"] = "test"

        samples = pandas.concat([train_samples, val_samples, test_samples])

    samples = samples[["topic", "sentence", "label", "split", "context", "candidates"]]

    return samples


@click.command()
@click.option('--setting', type=str, default="INJECT_JOINED_TOPIC")
@click.option('--k', type=int, default=2)
@click.option('--fold', type=int, default=0)
@click.option('--nfolds', type=int, default=1)
@click.option('--is_cross_topic', type=bool, default=False)
@click.option('--input_encoder_layers', type=str, default="11")
@click.option('--context_encoder_layers', type=str, default="11")
@click.option('--task', type=str, default="semeval2019t7")
@click.option('--variation', type=str, default="conceptnet")
@click.option('--model_name', type=str, default="bert-base-uncased")
@click.option('--retrieval_model_name', type=str)
@click.option('--output_dir', type=str, default="./")
@click.option('--data_folder', type=str, default="../data/")
@click.option('--random_seed', type=int, default=1)
@click.option('--batch_size', type=int, default=16)
@click.option('--gradient_accumulation_steps', type=int, default=2)
@click.option('--dump_model', type=bool, default=False)
@click.option('--shared_model', type=bool, default=False)
@click.option('--frozen_model', type=bool, default=False)
@click.option('--synchronised_model', type=bool, default=True)
@click.option('--eval_other_k', type=bool, default=False)
@click.option('--eval_other_variations', type=bool, default=False)
@click.option('--truncation_length', type=int, default=100)
@click.option('--experiment_prefix', type=str, default="bench_100")
def main(setting, k, fold, nfolds, is_cross_topic, input_encoder_layers, context_encoder_layers, task, variation,
         model_name, retrieval_model_name, output_dir, data_folder, random_seed, batch_size,
         gradient_accumulation_steps, dump_model, shared_model, frozen_model, synchronised_model,
         eval_other_k, eval_other_variations, truncation_length, experiment_prefix):

    print('================================')
    print('start: task={}, setting={}, k={}, variation={}, '
          'model_name={}, seed={}, truncation={}'.format(task, setting, k, variation, model_name,
                                                         random_seed, truncation_length))

    os.environ["WANDB_DISABLED"] = "true"

    assert fold < nfolds

    input_encoder_cross_attn_layers = [int(ele) for ele in input_encoder_layers.split(",")]
    context_encoder_cross_attn_layers = [int(ele) for ele in context_encoder_layers.split(",")]

    run_parameters = {
        "k": k if setting not in ["BASELINE", "BASELINE_TOPIC", "OTHER_TOPIC", "RANDOM_INPUT"] else None,
        "training": "CROSS-TOPIC" if is_cross_topic else "IN-TOPIC",
        "fold": fold,
        "setting": setting,
        "variation": variation,
        "model_name_str": model_name,
        "seed": random_seed,
        "shared_model": shared_model,
        "frozen_model": frozen_model,
        "synchronised_model": synchronised_model,
        "retrieval_model_name": retrieval_model_name
    }

    if "RETRO" in setting:
        run_parameters["input_encoder_cross_attn_layers"] = input_encoder_cross_attn_layers
        run_parameters["context_encoder_cross_attn_layers"] = context_encoder_cross_attn_layers

    # run_exist = check_for_run(
    #     mlflow_url,
    #     task,
    #     run_parameters,
    #     experiment_prefix
    # )
    #
    # if run_exist:
    #     print("Run already done")
    #     return

    local_run_id = str(uuid.uuid1())
    run_dir = output_dir + "/output-" + local_run_id
    run_parameters["dump_path"] = local_run_id

    seed_all(random_seed)

    encoding_function = encoding_functions[setting]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    samples = load_samples(task, variation, setting, k, nfolds, fold, is_cross_topic, data_folder)
    samples["sentence"] = samples["sentence"].apply(lambda sentence: truncate_sentence(sentence, truncation_length, tokenizer))
    samples["context"] = samples["context"].apply(lambda context: [truncate_sentence(ele, 500, tokenizer) for ele in context])

    num_labels = len(set(samples["label"]))

    if num_labels < 10:
        samples["label"] = samples["label"].apply(int)


    if "RETRO" in setting:
        if setting in ["RETRO_TOPIC", "RETRO_RANDOM_INPUT", "RETRO_CONCAT", "RETRO_CONCAT_JOINED_TOPIC"]:
            k = 1

        model = Inject(
            model_name,
            retrieval_model_name = retrieval_model_name,
            input_encoder_cross_attn_layers=input_encoder_cross_attn_layers,
            context_encoder_cross_attn_layers=context_encoder_cross_attn_layers,
            k=k,
            setting = setting,
            num_labels = num_labels,
            shared_model = shared_model,
            frozen_model = frozen_model,
            synchronised_model = synchronised_model
        )

    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


    for key, value in run_parameters.items():
        setattr(model.config, key, value)

    samples = samples[samples["split"].isin(["train", "val", "dev", "test"])]

    if is_cross_topic:
        train_samples = samples[samples["split"] == "train"]
        val_samples = samples[(samples["split"] == "val") | (samples["split"] == "dev") ]
        test_samples = samples[samples["split"] == "test"]
    else:
        train_samples, val_samples, test_samples = get_indomain_splits(samples, random_seed)

    train_dataset = load_tokenized_dataset(train_samples, setting, tokenizer, encoding_function)
    val_dataset = load_tokenized_dataset(val_samples, setting, tokenizer, encoding_function)
    test_dataset = load_tokenized_dataset(test_samples, setting, tokenizer, encoding_function)

    if "RETRO" in setting:
        data_collator = InjectDataCollatorWithPadding(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=512, padding="max_length")


    # experiment_name = experiment_prefix + "-" + task + "-retro"

    gradient_accumulation_steps = gradient_accumulation_steps
    per_device_train_batch_size = int(batch_size / gradient_accumulation_steps)

    training_args = TrainingArguments(
        output_dir=run_dir, evaluation_strategy="epoch", save_strategy="epoch", warmup_ratio=0.2, learning_rate=0.00002,
        metric_for_best_model="f1-macro", save_total_limit=1, load_best_model_at_end=True,gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=5, per_device_train_batch_size=per_device_train_batch_size, per_device_eval_batch_size=per_device_train_batch_size, seed=random_seed, remove_unused_columns="RETRO" not in setting
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics, data_collator=data_collator,
    )

    trainer.train()
    #trainer.evaluate(test_dataset, metric_key_prefix="test")

    predictions = trainer.predict(test_dataset, metric_key_prefix="test")

    prediction_frame = test_samples[["topic", "sentence", "label"]]
    prediction_frame["preds_raw"] = list(predictions.predictions)
    prediction_frame["preds"] = predictions.predictions.argmax(axis=1)

    if eval_other_variations:
        for other_variation in ["conceptnet_hop", "conceptnet", "conceptnet_hop_joined", "wikipedia", "common", "relevant_final", ""]:
            if other_variation != variation:

                if not os.path.exists(data_folder + "/" + task + "_with_candidates_" + other_variation + ".csv"):
                    continue

                other_samples = load_samples(task, other_variation, setting, k, nfolds, fold, is_cross_topic, data_folder)
                other_test_samples = other_samples[other_samples["split"] == "test"]

                test_dataset = load_tokenized_dataset(other_test_samples, setting, tokenizer, encoding_function)

                predictions = trainer.predict(test_dataset, metric_key_prefix="test")
                prediction_frame["preds_" + other_variation] = predictions.predictions.argmax(axis=1)

    if eval_other_k:
        for other_k in range(1,11):
            if other_k == k:
                continue

            if "RETRO" in setting:
                model.update_k(other_k)

            test_samples["neighbours"] = test_samples["candidates"].apply(lambda candidates: select_context_elements(candidates, setting=setting, k=other_k))

            test_dataset = load_tokenized_dataset(test_samples, setting, tokenizer, encoding_function)

            predictions = trainer.predict(test_dataset, metric_key_prefix="test")

            prediction_frame["preds_" + str(other_k)] = predictions.predictions.argmax(axis=1)

    prediction_file = run_dir + "/predictions.csv"
    prediction_frame.to_csv(prediction_file, index=False)


if __name__ == '__main__':
    main()
