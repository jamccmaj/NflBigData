#! /usr/bin/env python3

import os
import sys
import itertools

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn

play = int(sys.argv[1])

# adapt for windows
try:
    homedir = os.environ['HOME']
except KeyError:
    homedir = os.path.join(
        os.environ['HOMEDRIVE'], os.environ['HOMEPATH']
    )

datapath = os.path.join(
    homedir, 'dev/NflBigData/data/kaggle/train.csv'
)

data = pd.read_csv(datapath, low_memory=False)

for b, e in zip(range(0, len(data), 22), range(22, len(data) + 22, 22)):
    play_data = data.iloc[b:e]