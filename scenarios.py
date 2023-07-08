
import numpy as np
import matplotlib.pyplot as plt
from vehicle import *
from vehicle_estimator import *
from controller import *
from vehicle_system import *
from helpers import *
from copy import deepcopy
from world import *
from plotting import *


def run_scenario(scenario="simple", num_steps=100,  num_cars=2, world=None, plot=True, num_convergence_steps=1, v2x=True):

    if world is None:
        world = World(num_cars)

    # Parameters
    timestep = 0.1
    steering_angles = np.sin(np.linspace(0, 10*np.pi, num_steps)) /16.0 # Gentle steering angle
    accelerations = np.sin(np.linspace(0, 4*np.pi, num_steps)) / 5.0  # Gentle acceleration

    position_history = np.zeros((num_steps, num_cars, 2))
    desired_history = np.zeros((num_steps, num_cars, 2))
    pos_error_history = np.zeros((num_steps, num_cars))

    # com_graph = np.zeros((num_cars, num_cars))
    # for i in range(num_cars):
    #     for j in range(num_cars):
    #         if abs(i-j) == 1:
    #             com_graph[i, j] = 1

    # print(com_graph)


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
                vs.simulate_lidar( vsyst_behind )

            # first each predicts where the others will be
            # then they each create a control without knowledge of neighbour controls
            # then they repeatedly recalculate controls based on where they predict they will be based on these controls
            if vs.id > 0:
                # calculate control to follow car ahead
                if scenario == "predictive":
                    vs.compute_predictive_control( timestep )
                else:
                    vs.compute_follow_control( timestep )
                desired_history[step, i, :] = vs.controller.desired_state[:2]
            else:
                vs.set_next_ctrl(steering_angles[step], accelerations[step])
            if v2x:
                vs.emit_control_message()
                vs.emit_estimate_message()
                

            position_history[step, i, :] = vs.vehicle.get_state()[:2]
            pos_error_history[step, i] = np.linalg.norm(position_history[step, i, :] - desired_history[step, i, :])

        if scenario == "averaged":
            for _ in range(num_convergence_steps):
                for v in world.all_vehicle_systems:
                    if vs.id > 0:
                        # calculate control to follow car ahead
                        vs.compute_averaged_control( timestep )

                for v in world.all_vehicle_systems:
                    v.emit_control_message()

        for v in world.all_vehicle_systems: #
            v.update(timestep)
    if plot:
        plot_poses(position_history, desired_history)
        plot_errors(pos_error_history)
    
    return np.sum(pos_error_history)

if __name__ == '__main__':
    # run_scenario(num_cars=10)
    run_scenario(scenario="predictive", num_cars=10)