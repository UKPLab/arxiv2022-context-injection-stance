import collections
import itertools
import random
from dataclasses import dataclass
from typing import List, Any, Dict, Union, Optional, Sized, Iterator
import numpy
import pandas
import torch
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizer
from transformers.file_utils import PaddingStrategy
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers import is_datasets_available
from transformers import Trainer
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler
import datasets


def truncate_sentence(sentence, truncation_length, tokenizer):
    tokens = tokenizer.encode(text=sentence, max_length=truncation_length, truncation=True, add_special_tokens=False)
    return tokenizer.decode(tokens)


def get_indomain_splits(samples, seed, stratify=True, dev_test_size=.15):
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


class RandomDatasetSampler(torch.utils.data.Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            yield from torch.randperm(n, generator=generator).tolist()

    def __len__(self) -> int:
        return self.num_samples


class RandomTrainingDatasetTrainer(Trainer):
    #
    # def get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    #     if not isinstance(self.train_dataset, collections.abc.Sized):
    #         return None
    #
    #     generator = None
    #     if self.args.world_size <= 1 and _is_torch_generator_available:
    #         generator = torch.Generator()
    #         generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    #
    #     # Build the sampler.
    #     if self.args.group_by_length:
    #         if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
    #             lengths = (
    #                 self.train_dataset[self.args.length_column_name]
    #                 if self.args.length_column_name in self.train_dataset.column_names
    #                 else None
    #             )
    #         else:
    #             lengths = None
    #         model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
    #         if self.args.world_size <= 1:
    #             return LengthGroupedSampler(
    #                 self.args.train_batch_size * self.args.gradient_accumulation_steps,
    #                 dataset=self.train_dataset,
    #                 lengths=lengths,
    #                 model_input_name=model_input_name,
    #                 generator=generator,
    #             )
    #         else:
    #             return DistributedLengthGroupedSampler(
    #                 self.args.train_batch_size * self.args.gradient_accumulation_steps,
    #                 dataset=self.train_dataset,
    #                 num_replicas=self.args.world_size,
    #                 rank=self.args.process_index,
    #                 lengths=lengths,
    #                 model_input_name=model_input_name,
    #                 seed=self.args.seed,
    #             )
    #     else:
    #         return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = RandomDatasetSampler() # self.get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


@dataclass
class InjectDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    truncation: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    add_task_name: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [{
            "input_ids": feature["input_ids"],
            "label": feature["label"],
        } for feature in features]

        max_input_length = max([len(feature["input_ids"]) for feature in features])
        max_context_length = max(itertools.chain.from_iterable([
            [len(neighbour) for neighbour in feature["context_ids"]]
            for feature in features
        ]))

        self.max_length = min(512, max(max_input_length, max_context_length))

        batch = self.tokenizer.pad(
            input_ids,
            padding=self.padding,
            max_length=max_input_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # torch.Size([2, 32, 2, 128])
        if self.add_task_name:
            batch["task_name"] = features[0]['task_name']

        # batch_size, chunks, neighbours, seq
        context = [
            self.tokenizer.pad(
                [{"input_ids": ele} for ele in feature["context_ids"]],
                padding=self.padding,
                max_length=min(max_context_length, 512),
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            for feature in features
        ]
        batch["context_input_ids"] = torch.stack([ele["input_ids"] for ele in context])
        batch["context_attention_mask"] = torch.stack([ele["attention_mask"] for ele in context])

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch


@dataclass
class MultiCLSDataCollator:
    tokenizer: PreTrainedTokenizer
    add_topic: bool = True
    add_task_name: bool = False
    truncation_length: int = 100
    truncation_strategy: str = 'only_first'
    return_tensors: str = 'pt'

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        if self.add_topic:
            input = [(feature['sentence'], feature['topic']) for feature in features]
        else:
            input = [feature['sentence'] for feature in features]
        batch = self.tokenizer(input, truncation=self.truncation_strategy,
                                   max_length=self.truncation_length, padding='max_length',
                                   return_tensors=self.return_tensors)
        batch['labels'] = torch.tensor([feature['labels'] for feature in features])
        context_input_ids = []
        context_attention_mask = []
        context_token_type_ids = []
        for feature in features:
            for context in feature['context']: # list of contexts because 1 <= k < len(contexts)
                res = self.tokenizer(context, truncation=self.truncation_strategy,
                                   max_length=self.truncation_length, padding='max_length',
                                   return_tensors=self.return_tensors)
                context_input_ids.append(res['input_ids'])
                context_attention_mask.append(res['attention_mask'])
                context_token_type_ids.append(res['token_type_ids'])

        batch['context_input_ids'] = torch.stack(context_input_ids)
        batch['context_attention_mask'] = torch.stack(context_attention_mask)
        batch['context_token_type_ids'] = torch.stack(context_token_type_ids)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        if self.add_task_name:
            batch["task_name"] = features[0]['task_name']

        return batch


@dataclass
class MultiCLSCrossTargetDataCollator:
    tokenizer: PreTrainedTokenizer
    add_topic: bool = True
    add_task_name: bool = False
    truncation_length: int = 100
    return_tensors: str = 'pt'

def get_ukp_sent_topic_splits(samples: pandas.DataFrame, n_folds: int = 4):
    return [[["gun control", "marijuana legalization", "minimum wage", "nuclear energy"], ["cloning", "death penalty"],
             ["abortion", "school uniforms"]],
            [["school uniforms", "gun control", "minimum wage", "death penalty"], ["abortion", "cloning"],
             ["marijuana legalization", "nuclear energy"]],
            [["school uniforms", "marijuana legalization", "death penalty", "nuclear energy"],
             ["abortion", "minimum wage"], ["cloning", "gun control"]],
            [["abortion", "cloning", "marijuana legalization", "school uniforms"], ["gun control", "nuclear energy"],
             ["minimum wage", "death penalty"]]]



def get_topic_splits(samples: pandas.DataFrame, n_folds: int = 4):
    fold_topics = []

    topics = list(sorted(samples["topic"].unique()))

    test_topic_splits = numpy.array_split(range(len(topics)), n_folds)

    target_length = len(test_topic_splits[0])

    for fold, test_topic_ids in enumerate(test_topic_splits):

        if len(test_topic_ids) != target_length:

            diff = target_length - len(test_topic_ids)

            topic_ids_to_exclude = test_topic_splits[fold - 1][-diff:]
            topics_to_exclude = [topics[test_topic_id] for test_topic_id in topic_ids_to_exclude]
        else:
            topics_to_exclude = []

        test_topics = [topics[test_topic_id] for test_topic_id in test_topic_ids]

        train_topics = sorted([topic for topic in topics if topic not in test_topics and topic not in topics_to_exclude])

        train_topics, dev_topics = train_test_split(train_topics, train_size=0.8, random_state=0)

        fold_topics.append([train_topics, dev_topics, list(test_topics)])

    return fold_topics


topic_split_function = {
    "ukp-sent": get_topic_splits,
    "iac": get_topic_splits,
    "ibm-evi": get_topic_splits,
    "pstance": get_topic_splits,
    "ibm-arg-q-stance": get_topic_splits,
    "semevalt6": get_topic_splits,
    "perspectrum": get_topic_splits,
    "vast": get_topic_splits,
    "arc": get_topic_splits,
    "arct_pure": get_topic_splits,
    "arct_claim": get_topic_splits,
    "arct_reason": get_topic_splits,
}



def compute_metrics(eval_preds):
    logits, labels = eval_preds

    predictions = numpy.argmax(logits, axis=-1)

    metrics = {
        "accuracy": accuracy_score(y_true=labels, y_pred=predictions),
        "f1-macro": f1_score(y_true=labels, y_pred=predictions, average="macro"),
    }

    _, _, class_based_f1,_ = precision_recall_fscore_support(y_true=labels, y_pred=predictions)

    for label, f1 in enumerate(class_based_f1):
        metrics["label-" + str(label) + "-f1"] = f1

    return metrics


def seed_all(seed = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_for_run(mlflow_url, task, params, experiment_prefix):
    client = MlflowClient(mlflow_url)

    experiment = client.get_experiment_by_name(experiment_prefix + "-" + task + "-retro")

    if experiment:

        filter_string = " and ".join(["params." + key + "='" + str(value) + "'" for key, value in params.items() if key != "k"])

        if params["setting"] not in ["BASELINE", "BASELINE_TOPIC", "OTHER_TOPIC", "RANDOM_INPUT"]:
            filter_string = filter_string + " and params.k='" + str(params["k"]) + "'"

        #if params["synchronised_model"]:
        #    filter_string = filter_string + " and params.synchronised_model='" + str(params["synchronised_model"]) + "'"

        print("check for run with", filter_string)

        runs = client.search_runs(
            experiment_ids=experiment.experiment_id,
            filter_string=filter_string
        )

        relevant_runs = []

        for run in runs:
            final_run_metrics = [metric for metric, score in run.data.metrics.items()]

            if len(final_run_metrics) == 0:
                client.delete_run(run_id=run.info.run_id)
            else:
                relevant_runs.append(run)

        return len(relevant_runs) > 0
    else:
        return False
