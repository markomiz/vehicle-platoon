
import numpy as np
import matplotlib.pyplot as plt
from vehicle import *
from vehicle_estimator import *
from controller import *
from vehicle_system import *
from helpers import *
from copy import deepcopy
from world import *

OFFSET = 5

def plot_poses(poses, desired_poses=[]):
    # Plot the errors
    plt.figure()
    plt.plot(poses[:, 0, 0], poses[:,0,1], label='Lead Vehicle')
    for i in range(1,poses.shape[1]):
        plt.plot(poses[:, i, 0], poses[:,i,1] - 5 * i, label='Follower ' + str(i))
        plt.plot(desired_poses[:, i, 0], desired_poses[:,i,1] - 5* i, label='Desired ' + str(i), linestyle="dotted")


    plt.legend()
    plt.grid(True)
    plt.show()

def plot_errors(errors):
    for i in range(1,errors.shape[1]):
        plt.plot(range(errors.shape[0]), errors[:,i] , label='Error ' + str(i))

    plt.legend()
    plt.grid(True)
    plt.show()