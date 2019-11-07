#! /usr/bin/env python3

import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))