# Contextual information integration for stance detection via cross-attention

This repository includes the code for integrating contextual information for supervised text 
classification tasks using a dual-encoder approach and information exchange via cross-attention.


Further details can be found in our publication [Contextual information integration for stance 
detection via cross-attention](https://arxiv.org/abs/2211.01874).


> **Abstract:** Stance detection deals with the identification of an author's stance towards a target and is applied on various text domains like social media and news.
In many cases, inferring the stance is challenging due to insufficient access to contextual information.
Complementary context can be found in knowledge bases but integrating the context into pretrained language models is non-trivial due to their graph structure.
In contrast, we explore an approach to integrate contextual information as text which aligns better with transformer architectures.
Specifically, we train a model consisting of dual encoders which exchange information via cross-attention.
This architecture allows for integrating contextual information from heterogeneous sources.
We evaluate context extracted from structured knowledge sources and from prompting large language models.
Our approach is able to outperform competitive baselines (1.9pp on average) on a large and diverse stance detection benchmark, both (1) in-domain, i.e. for seen targets, and (2) out-of-domain, i.e. for targets unseen during training.
Our analysis shows that it is able to regularize for spurious label correlations with target-specific cue words.

## Information

Contact person: Tilman Beck, <lastname>@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to e-mail us or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Project structure
**(change this as needed!)**

* `data/` -- container for the data which includes the scripts for processing the benchmark datasets and retrieving tweets for Twitter datasets
* `src/analysis` -- Python scripts to analyze data, attention attribution, compute significance and some visualization utils
* `src/model` -- contains the model files
* `src/retrieve` -- the code for retrieving contextual information from external knowledge sources (e.g. ConceptNet, CauseNet, T0pp)
* `src/train` -- utility files for training 

## Requirements

* Python3.6 or higher
* PyTorch 1.10.2 or higher

## Data

We make use of the benchmark datasets provided by [Schiller et al. 2021](https://doi.org/10.1007/s13218-021-00714-w) 
and [Hardalov et al. 2021](https://aclanthology.org/2021.emnlp-main.710.pdf). The datasets are linked in the respective 
repositories [here](https://github.com/UKPLab/mdl-stance-robustness#preprocessing) 
and [here](https://github.com/checkstep/mole-stance)

Once you have obtained all datasets, put them in a folder (e.g. `benchmark_original`) and run

```
$ preprocess_benchmark.py --root_dir /path/to/benchmarḱ_original --output_dir /path/to/benchmark_processed
```

Now all datasets should be available in the same JSON format, that is

`{"text":"This is sample text", "label":1, "target":"example target", "split":"train"}`

Our dataset numbers differ slightly from the ones reported by Hardalov et al. (2021) due to the following reasons:

* rumor: not all tweets could be downloaded
* mtsd: we were provided the full dataset by the original authors

To produce the final files for running the benchmark experiments you can either use the pre-computed context or compute it by your self. 
Note you can also add other tasks. Therefore, you need to create a new folder within the processed benchmark folder (e.g. `benchmark_processed`) for each task and compute the context on your own. There you need to put the samples as `.jsonl` in the same format as mentioned above.

### Use Pre-Computed Context

The pre-computed context is available to download (TBD) and is the same as we used in our paper. It includes the following sources:
* `CauseNet` [Heindorf et al. 2020](https://papers.dice-research.org/2020/CIKM-20/heindorf_2020a_public.pdf)
* `ConceptNet` [here](https://conceptnet.io/)
* `T0pp-NP` using [Sanh et al. 2021](https://arxiv.org/abs/2110.08207)
* `T0pp-NP-Targ` using [Sanh et al. 2021](https://arxiv.org/abs/2110.08207)
* `All` combination of the four sources above
* `Random*` context sampled from the Gutenberg Project [here](https://www.nltk.org/book/ch02.html)

After downloading the pre-computed context, you need to extract into a folder (e.g. `benchmark_context`). The following will then compose the final files to run the experiments.
```
$ finalize_sd_benchmark.py --root_dir /path/to/benchmark_processed --context_dir /path/to/benchmark_context  --output_dir /path/to/benchmark_final
```

### Do-it-yourself
TODO
* download conceptnet, extract to graphfile, find relevant nodes
* download causenet, encode causenet and task inputs, find the most closest
* find NPs, describe them using T0pp

## Setup

* Clone the repository
```
$ git clone https://github.com/UKPLab/arxiv2022-context-injection-stance
$ cd arxiv2022-context-injection-stance
```
* Create the environment and install dependencies

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Running the experiments

To run the `INJECT` model with `k=2` context candidates retrieved from `conceptnet` for dataset `argmin`:

```
$ python run.py --task argmin --variation conceptnet --setting INJECT_JOINED_TOPIC --k 2 --output_dir /path/to/results/
```

### Parameter description
**(change this as needed!)**

* `--setting`
  * The experiment setting which defines which context injection to use from `[BASELINE, BASELINE_TOPIC, BASELINE_JOINED_TOPIC, INJECT_TOPIC, INJECT_JOINED_TOPIC]`
* `--k`
  * The number of context candidates to use (default 2)
* `--is_cross_topic`
  * Declare if the evaluation setting should be cross-target (True) or in-target (False) 
* `--input_encoder_layers`
  * The layer of the input encoder at which cross-attention should be applied to (default 11)  
* `--context_encoder_layers`
  * The layer of the context encoder at which cross-attention should be applied to (default 11)
* `--task`
  * The task which is used (choose from `[emergent,argmin,vast,poldeb,mtsd,rumor,wtwt,iac1,scd,semeval2019t7,semeval2016task6,perspectrum,ibmcs,arc,fnc1,snopes]`) 
* `--variation`
  * The context source to use (choose from `[conceptnet, causenet, t0pp_key_np, t0pp_key_np_target]`) 
* `--model_name`
  * The backbone transformer model which is used for the encoders (e.g. `bert-base-uncased`) 
* `--output_dir`
  * The directory to write the experiment outputs 
* `--data_folder`
  * The directory where the data files are stored 
* `--shared_model`
  * Both input encoder and context encoder use shared weights 
* `--frozen_model`
  * Only train classification head, freeze basemodel parameters of both encoders 
* `--truncation_length`
  * Cut input at `truncation_length` 
* `--random_seed` 
* `--batch_size` 


## Citing

Please use the following citation:

```
@article{beck-waldis-gurevych:2022,
  title = {Contextual information integration for stance detection via cross-attention},
  author = {Beck, Tilman and Waldis, Andreas and Gurevych, Iryna},
  journal = {arXiv},
  year = {2022},
  url = "https://arxiv.org/abs/2211.01874"
}
```
