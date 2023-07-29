
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

def plot_poses(poses, desired_poses, v1, v2, lead):
    CMAP = cm.get_cmap("tab10")

    fig = plt.figure()
    ax = plt.subplot(111)

    if lead:
        ax.plot(poses[:, 0, 0], poses[:,0,1], label='Lead Vehicle Position', color=CMAP(1.0))
        ax.plot(desired_poses[:, 0, 0], desired_poses[:,0,1] , label='Lead Desired Position', linestyle="dotted",color=CMAP(1.0) )
    N = poses.shape[1]
    for i in range(lead,N):
        ax.plot(poses[:, i, 0], poses[:,i,1] - 5 * i, label=v1 + " " + str(i), color=CMAP(float(i)/N))
        ax.plot(desired_poses[:, i, 0], desired_poses[:,i,1] - 5* i, label=v2 + " " + str(i), linestyle="dotted", color=CMAP(float(i)/N))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_title("Postions of "+v1+ " vs. " +v2 + "\n(vehicle positions offset vertically by 5x their place in platoon for clarity)") 
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    ax.set_xlabel("(m)")
    plt.show()

def plot_single(single, name):
    CMAP = cm.get_cmap("tab10")

    fig = plt.figure()
    ax = plt.subplot(111)
    N = len(single[0])
    for i in range(0,N):
        ax.plot(range(len(single)), single[:,i] , label=name + str(i), color=CMAP(float(i)/N))
        
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    plt.show()

def plot_desired_errors(errors, ylabel):
    CMAP = cm.get_cmap("cividis")
    N = errors.shape[1]
    for i in range(0,N):
        plt.plot(range(errors.shape[0]), errors[:,i] , label='Follower ' + str(i), color=CMAP(float(i)/N))
    
    plt.xlabel('Time Step')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_estimate_errors(pos_estimate_errors, estimate_type):
    
    #text portion
    fig, ax = plt.subplots()
    ind_array = np.arange(0, pos_estimate_errors.shape[0], 1)
    x, y = np.meshgrid(ind_array, ind_array)
    ax.matshow(pos_estimate_errors, cmap="Wistia")
    plt.xlabel("Mean Estimate Error of Other Vehicle (m)")
    plt.ylabel("Vehicle Position in Platoon")
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = pos_estimate_errors[y_val, x_val]
        c = round(c, 3)
        ax.text(x_val, y_val, c, va='center', ha='center')

 
    plt.show()


def plot_comparison_table():

    labels = ["Reactive, \n No Communication", "Reactive,\n Communicative",  "5 Step\n Horizon", "9 Step\n Horizon"]
    means = np.array([179.17895725,89.02834428 ,70.55984074,47.66678051]) / 200
    plt.bar(labels, means, color=['red','blue','green','green'])
    plt.ylabel("Mean Follow Error (m)")
    plt.title("Comparison of Follow Errors for Different Methods")
    plt.show()


if __name__ == "__main__":
    plot_comparison_table()