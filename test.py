#! /usr/bin/env python3

import os
import sys

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

play = int(sys.argv[1])

homedir = os.environ['HOME']
datapath = os.path.join(
    homedir, 'dev/NflBigData/data/kaggle/train.csv'
)

data = pd.read_csv(datapath, low_memory=False)
print(data.head())

play_data = data[play*22: (play+1)*22]

rusher = np.argmax(
    np.array(play_data['NflId'] == play_data['NflIdRusher'])
)

color_ids = ['red'] * 11 + ['blue'] * 11
color_ids[rusher] = 'green'

# show player position in first play in scatter plot with colors
plt.scatter(play_data['X'], play_data['Y'], color=color_ids)
# plt.show()

# TODO (1) Plot vectors with orientation and direction of movement
#      (2) Scale direction of movement vectors by speed and accel
#      (3) Calculate rusher space between all defenders
#      (4) Calculate player influence using basic bivariate normal

# need to convert to radians
degs_dir = play_data['Dir']
rads_dir = degs_dir * (np.pi/180)

# getting x, y components on the unit circle
# will scale by speed at some point
x_dirs = np.array(np.cos(rads_dir)).reshape(-1, 1)
y_dirs = np.array(np.sin(rads_dir)).reshape(-1, 1)
dir_vecs = np.concatenate([x_dirs, y_dirs], axis=1)

# getting the origin of the vectors to plot, i.e. player positions
x_pos = np.array(play_data['X']).reshape(-1,1)
y_pos = np.array(play_data['Y']).reshape(-1,1)
pos = np.concatenate([x_pos, y_pos], axis=1)

# plot the vectors ...
# not quite correct, need to take into account direction of play
plt.quiver(
    pos[:, 0], pos[:, 1],
    dir_vecs[:, 0], dir_vecs[:, 1],
    color=color_ids, scale=50
)
plt.show()

# TODO -- Project: Calculating player influence
# (1) Generate mesh grid, which will be image pixels
# (2) Calculate influence for each player and point on the image
# Note: should be 11 images for defense, 10 for offense, still deciding for ball carrier
# (3) Add images together for defense and offense separately
# (4) Take difference between summed images for defense and offense, sigmoid it for a single image
