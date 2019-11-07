
import itertools
import numpy as np

from scipy.stats import multivariate_normal as mvn

from utils.math import deg_to_rad

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

    return R @ S @ S @ np.linalg.inv(R)

def compute_image(player_data, image_wd, image_ht):
    image = np.zeros((image_ht, image_wd))
    sigma = compute_covariance(player_data)
    mvn_player = mvn(
        mean=[player_data.X, player_data.Y], cov=sigma
    )

    for _, (a, b) in enumerate(
        itertools.product(range(image_ht), range(image_wd))
    ):
        a = image_ht - a - 1
        image[a, b] = mvn_player.pdf([b, a])

    return image
