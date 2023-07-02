
import numpy as np
import matplotlib.pyplot as plt
from vehicle import *
from vehicle_estimator import *
from controller import *
from vehicle_system import *
from helpers import *
from copy import deepcopy
from world import *


def simple_follow(num_cars=10):

    world = World(num_cars)

    # Parameters
    num_steps = 50
    timestep = 0.1
    steering_angle = np.radians(1)  # Gentle steering angle
    acceleration = 0.0  # Gentle acceleration

    position_history = np.zeros((num_steps, num_cars, 2))
    estimate_history = np.zeros((num_steps, num_cars, 2))


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

            if vs.id > 0:
                # calculate control to follow car ahead
                vs.compute_follow_control( vs.estimates[i-1].state  ,  timestep)

                estimate_history[step, :, :] = [e.state[:2] for e in vs.estimates.values()]
                
            else:
                vs.set_next_ctrl(steering_angle, acceleration)
                

            position_history[step, i, :] = vs.vehicle.get_state()[:2]

        for v in world.all_vehicle_systems: #
            v.update(timestep)

    # Plot the errors
    plt.figure()
    plt.plot(position_history[:, 0, 0], position_history[:,0,1], label='lead')
    for i in range(1,num_cars):
        plt.plot(position_history[:, i, 0], position_history[:,i,1], label='Follower ' + str(i))
    plt.plot(position_history[:, 2, 0], position_history[:,2,1], label='Desired')
    # plt.axis('square')
    plt.legend()
    plt.grid(True)
    plt.show()

simple_follow()