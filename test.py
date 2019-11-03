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

# need to convert to radians
degs = data['Dir']
rads = degs * (np.pi/180)

# getting x, y components on the unit circle
# will scale by speed at some point
x_dirs = np.array(np.cos(rads)).reshape(-1, 1)
y_dirs = np.array(np.sin(rads)).reshape(-1, 1)
dir_vecs = np.concatenate([x_dirs, y_dirs], axis=1)

# getting the origin of the vectors to plot, i.e. player positions
x_pos = np.array(data['X']).reshape(-1,1)
y_pos = np.array(data['Y']).reshape(-1,1)
pos = np.concatenate([x_pos, y_pos], axis=1)

# plot the vectors ...
# not quite correct, need to take into account direction of play
plt.quiver(
    pos[0:22, 0], pos[0:22, 1],
    dir_vecs[0:22, 0], dir_vecs[0:22, 1],
    color=['r']*11 + ['b']*11, scale=21
)
