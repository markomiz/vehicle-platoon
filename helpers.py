import numpy as np

def mod2pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))