# Contextual information integration for stance detection via cross-attention

This repository includes the code for integrating contextual information for supervised text 
classification tasks using a dual-encoder approach and information exchange via cross-attention.


Further details can be found in our publication [Contextual information integration for stance 
detection via cross-attention](https://arxiv.org/abs/TODO).


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

Contact person: Tilman Beck, beck@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

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
**(change this as needed!)**

```
$cd bla/bla/bla
$some_cool_commands_here
```

### Expected results
**(change this as needed!)**

After running the experiments, you should expect the following results:

(Feel free to describe your expected results here...)

### Parameter description
**(change this as needed!)**

* `x, --xxxx`
  * This parameter does something nice
...
* `z, --zzzz`
  * This parameter does something even nicer


## Citing

Please use the following citation:

```
@article{beck-waldis-gurevych:2022,
  title = {Contextual information integration for stance detection via cross-attention},
  author = {Beck, Tilman and Waldis, Andreas and Gurevych, Iryna},
  journal = {arXiv},
  year = {2022},
  url = "https://arxiv.org/abs/TODO"
}
```