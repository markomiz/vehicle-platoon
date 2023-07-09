
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from vehicle import *
from vehicle_estimator import *
from controller import *
from vehicle_system import *
from helpers import *
from copy import deepcopy
from world import *

OFFSET = 10

def plot_poses(poses, desired_poses=[]):
    CMAP = cm.get_cmap("Wistia")

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(poses[:, 0, 0], poses[:,0,1], label='Lead Vehicle')
    N = poses.shape[1]
    for i in range(1,N):
        ax.plot(poses[:, i, 0], poses[:,i,1] - 5 * i, label='Follower ' + str(i), color=CMAP(float(i)/N))
        ax.plot(desired_poses[:, i, 0], desired_poses[:,i,1] - 5* i, label='Desired ' + str(i), linestyle="dotted", color=CMAP(float(i)/N))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    plt.show()

def plot_desired_errors(errors):
    CMAP = cm.get_cmap("Wistia")
    N = errors.shape[1]
    for i in range(1,N):
        plt.plot(range(errors.shape[0]), errors[:,i] , label='Follower ' + str(i), color=CMAP(float(i)/N))

    plt.legend()
    plt.grid(True)
    plt.show()

def plot_estimate_errors(pos_estimate_errors, estimate_type):
    
    #text portion
    fig, ax = plt.subplots()
    ind_array = np.arange(0, pos_estimate_errors.shape[0], 1)
    x, y = np.meshgrid(ind_array, ind_array)
    ax.matshow(pos_estimate_errors, cmap="Wistia")
    plt.xlabel("Estimate Error of Other Vehicle")
    plt.ylabel("Vehicle Position in Platoon")
    ax.set_title("Mean Estimation Error of " + estimate_type )
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = pos_estimate_errors[y_val, x_val]
        c = round(c, 3)
        ax.text(x_val, y_val, c, va='center', ha='center')

 
    plt.show()