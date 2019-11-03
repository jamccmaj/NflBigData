#! /usr/bin/env python3

import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

homedir = os.environ['HOME']
datapath = os.path.join(
    homedir, 'dev/NflBigData/data/kaggle/train.csv'
)

data = pd.read_csv(datapath, low_memory=False)
print(data.head())

# get id of rusher on the first play of the first game
rusher = np.argmax(
    np.array(
        data.head(n=22)['NflId'] == data.head(n=22)['NflIdRusher']
    )
)

color_ids = ['red'] * 11 + ['blue'] * 11
color_ids[rusher] = 'green'

# show player position in first play in scatter plot with colors
plt.scatter(data['X'][0:22], data['Y'][0:22], color=color_ids)
plt.show()

# TODO (1) Plot vectors with orientation and direction of movement
#      (2) Scale direction of movement vectors by speed and accel
#      (3) Calculate rusher space between all defenders
