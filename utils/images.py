
import numpy as np

from utils.math import deg_to_rad

def compute_covariance(player_data):
    dir_rad = deg_to_rad(data.Dir)
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
    S = np.array([[Sx, 0.0], [Sy, 0.0]])

    return R @ S @ S @ np.linalg.inv(R)

