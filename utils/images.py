
import sys
import itertools
import numpy as np

from scipy.stats import multivariate_normal as mvn

from utils.math import deg_to_rad, sigmoid

def compute_covariance(player_data):
    dir_rad = deg_to_rad(player_data.Dir)
    speed = player_data.S
    accel = player_data.A

    R = np.array(
        [
            [np.cos(dir_rad), -np.sin(dir_rad)],
            [np.sin(dir_rad), np.cos(dir_rad)]
        ]
    )

    Sx = speed * accel * np.cos(dir_rad)
    Sy = speed * accel * np.sin(dir_rad)
    S = np.array([[Sx, 0.0], [0.0, Sy]])

    return R @ S @ S @ np.linalg.pinv(R)


def compute_image(player_data, image_wd, image_ht):
    image = np.zeros((image_ht, image_wd))
    sigma = compute_covariance(player_data)

    if np.linalg.cond(sigma) < (1 / sys.float_info.epsilon):

        mvn_player = mvn(
            mean=[player_data.X, player_data.Y], cov=sigma,
        )

        a, b = zip(*list(
            itertools.product(range(image_ht), range(image_wd))
        ))

        image[a, b] = mvn_player.pdf(list(zip(b, a)))

    return image


def compute_image_for_team(play_data, image_wd, image_ht):
    images = np.zeros((len(play_data), image_ht, image_wd))

    for x in range(len(play_data)):
        images[x] = compute_image(
            play_data.iloc[x], image_wd, image_ht
        )

    return images


def compute_ctrl_prob(play_images):
    return sigmoid(
        play_images[:11].sum(axis=0) - play_images[11:].sum(axis=0)
    )