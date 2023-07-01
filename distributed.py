
import numpy as np
import matplotlib.pyplot as plt
from vehicle import *
from vehicle_estimator import *
from controller import *
from vehicle_system import *
from helpers import *
from copy import deepcopy
from world import *


def distributed_control(num_cars=20, num_convergence_steps=3):

    world = World(num_cars)

    # Parameters
    num_steps = 50
    timestep = 0.1
    steering_angle = np.radians(2)  # Gentle steering angle
    acceleration = 0.0  # Gentle acceleration

    position_history = np.zeros((num_steps, num_cars, 2))

    com_graph = np.zeros((num_cars, num_cars))
    for i in range(num_cars):
        for j in range(num_cars):
            if abs(i-j) == 1:
                com_graph[i, j] = 1

    print(com_graph)


    # Run the estimation test loop
    for step in range(num_steps):

        for i, vs in enumerate(world.all_vehicle_systems):
            
            # Measure GNSS position
            vs.simulate_GNSS( )

            # Measure speed
            vs.simulate_speedometer( )

            # Measure angle
            vs.simulate_compass( )

            # Lidar ahead and behind
            if i - 1 >= 0:
                vsyst_in_front = world.all_vehicle_systems[i-1]
                vs.simulate_lidar( vsyst_in_front)
            
            if i + 1 < len(world.all_vehicle_systems):
                vsyst_behind = world.all_vehicle_systems[i+1]
                vs.simulate_lidar( vsyst_behind)

        # first each predicts where the others will be
        # then they each create a control without knowledge of neighbour controls
        # then they repeatedly recalculate controls based on where they predict they will be based on these controls
            if vs.id > 0:
                # calculate control to follow car ahead
                vs.compute_predictive_control(   timestep)
                position_history[step, i+1, :] = vs.controller.desired_state[:2]
            else:
                vs.set_next_ctrl(steering_angle, acceleration)
                vs.emit_control_message()
                

            position_history[step, i, :] = vs.vehicle.get_state()[:2]

        for v in world.all_vehicle_systems: #
            v.update(timestep)

    # Plot the errors
    plt.figure()

    # plt.plot(reals[:,0], reals[:,1],label='real')
    # plt.plot(ests[:,0], ests[:,1],label='ests')
    plt.plot(position_history[:, 0, 0], position_history[:,0,1], label='lead')
    # plt.plot(position_history[:, 1, 0], position_history[:,1,1], label='Follower')
    plt.plot(position_history[:, 2, 0], position_history[:,2,1], label='Desired')
    # plt.axis('square')
    plt.legend()
    plt.grid(True)
    plt.show()

distributed_control()