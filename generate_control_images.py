#! /usr/bin/env python3

import os
import sys
import itertools

import numpy as np
import pandas as pd

from utils.math import sigmoid
from utils.images import compute_image_for_team, compute_ctrl_prob

datadir = 'dev/NflBigData/data'

# adapt for windows
try:
    homedir = os.environ['HOME']

except KeyError:
    homedir = os.path.join(
        os.environ['HOMEDRIVE'], os.environ['HOMEPATH']
    )

datapath = os.path.join(
    homedir, f'{datadir}/kaggle/train.csv'
)

outdir = os.path.join(homedir, datadir)

data = pd.read_csv(datapath, low_memory=False)
num_images = data.shape[0] // 22

im_wd, im_ht = 120, 57
image_data = np.zeros((num_images, im_ht, im_wd))

zipper = zip(range(0, len(data), 22), range(22, len(data) + 22, 22))
for i, (b, e) in enumerate(zipper):
    play_data = data.iloc[b:e]
    play_images = compute_image_for_team(play_data, im_wd, im_ht)
    image_data[i] = compute_ctrl_prob(play_images)

    if i % 100 == 0:
        print(i)

np.save(f"{outdir}/influence_images.npy", image_data)
