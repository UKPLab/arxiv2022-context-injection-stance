import glob
import itertools
import json
from collections import defaultdict, Counter, OrderedDict
from multiprocessing import Pool

import plotly.express as px

import numpy
import pandas
from tqdm import tqdm

def parse_correlation_file(file_path):
    correlation_frame = pandas.read_csv(file_path)
    return correlation_frame

correlations = pandas.concat([
    parse_correlation_file(file_path)
    for file_path in glob.glob("correlations/*.csv")
])

correlations.to_csv("token_activation_correlations.csv")
