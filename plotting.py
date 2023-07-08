
import numpy as np
import matplotlib.pyplot as plt
from vehicle import *
from vehicle_estimator import *
from controller import *
from vehicle_system import *
from helpers import *
from copy import deepcopy
from world import *

OFFSET = 50
# Helper functions for color calculations
def hue_to_rgb(p, q, t):
    if t < 0:
        t += 1
    if t > 1:
        t -= 1
    if t < 1/6:
        return p + (q - p) * 6 * t
    if t < 1/2:
        return q
    if t < 2/3:
        return p + (q - p) * (2/3 - t) * 6
    return p
def hsl_to_rgb(h, s, l):
    # Convert HSL values to the range [0, 1]
    h /= 255.0
    s /= 255.0
    l /= 255.0

    # Check if saturation is 0 (achromatic)
    if s == 0:
        r = g = b = l
    else:
        # Calculate intermediate values for conversion
        if l < 0.5:
            q = l * (1 + s)
        else:
            q = l + s - l * s
        p = 2 * l - q

        # Calculate RGB components
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

    # Convert RGB values to the range [0, 255]
    r = round(r * 255)
    g = round(g * 255)
    b = round(b * 255)

    # Return the RGB values as a string compatible with matplotlib
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_poses(poses, desired_poses=[]):
    hue_increase = 255.0 /poses.shape[1] * 2
    hue = 0
    # Plot the errors
    plt.figure()
    plt.plot(poses[:, 0, 0], poses[:,0,1], label='Lead Vehicle')
    for i in range(1,poses.shape[1]):
        plt.plot(poses[:, i, 0], poses[:,i,1] - 5 * i, label='Follower ' + str(i), color=hsl_to_rgb(hue, 255,120))
        plt.plot(desired_poses[:, i, 0], desired_poses[:,i,1] - 5* i, label='Desired ' + str(i), linestyle="dotted", color=hsl_to_rgb(hue, 170,100))
        hue += hue_increase
        hue = hue % 255


    plt.legend()
    plt.grid(True)
    plt.show()

def plot_errors(errors):
    for i in range(1,errors.shape[1]):
        plt.plot(range(errors.shape[0]), errors[:,i] , label='Error ' + str(i))

    plt.legend()
    plt.grid(True)
    plt.show()