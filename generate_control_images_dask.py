#! /usr/bin/env python3

import os
import sys
from functools import partial

import pandas as pd
import numpy as np

import dask
import dask.dataframe as dd

from utils.images import compute_control_images

try:
    nworkers = int(sys.argv[1])
except IndexError:
    print("Specify number of cpus to use for computations")
    sys.exit(1)

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
df = dd.from_pandas(data)
num_images = data.shape[0] // 22

im_wd, im_ht = 120, 57
part_cci = partial(compute_control_images, image_wd=im_wd, image_ht=im_ht)
computation_graph = df.map_partitions(
    lambda df: df.groupby(['PlayId']).apply(part_cci)
)

dask.config.set(scheduler='processes')
image_data = computation_graph.compute(num_workers=nworkers)
image_data = np.concatenate(
    [image_data.iloc[i].reshape(
        1, *image_data.iloc[i].shape
    ) for i in range(len(image_data))], axis=0
)

np.save(f"{outdir}/influence_images_dask.npy", image_data)
