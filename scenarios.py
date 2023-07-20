
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


def run_scenario(scenario="simple", num_steps=100,  num_cars=2, world=None, plot=True, mpc_horizon=1, v2v=True, network_loss=0.0):

    if world is None:
        world = World(num_cars, mpc_horizon=mpc_horizon, network_loss=network_loss )

    # Parameters
    timestep = 0.1
    road_points_x = np.linspace(0,1000, 10000)
    road_points_y = np.sin(road_points_x / 10) * 2 
    road_points = np.stack((road_points_x,road_points_y))
    steering_angles = np.sin(np.linspace(0, 10*np.pi, num_steps + mpc_horizon)) / 10.0 # Gentle steering angle
    accelerations = np.sin(np.linspace(0, 4*np.pi, num_steps + mpc_horizon)) / 2.0  # Gentle acceleration

    position_history = np.zeros((num_steps, num_cars, 2))
    desired_history = np.zeros((num_steps, num_cars, 2))
    follow_dist_error_history = np.zeros((num_steps, num_cars -1))
    estimate_errors = np.zeros((num_steps, num_cars, num_cars, 3))

    # com_graph = np.zeros((num_cars, num_cars))
    # for i in range(num_cars):
    #     for j in range(num_cars):
    #         if abs(i-j) == 1:
    #             com_graph[i, j] = 1

    # print(com_graph)


    # Run the estimation test loop
    for step in range(num_steps):

        for i, vs in enumerate(world.all_vehicle_systems):
            position_history[step, i, :] = vs.vehicle.get_state()[:2]
            if i > 0:
                follow_dist = np.linalg.norm(position_history[step, i, :] - position_history[step, i-1, :])
                follow_dist_error = abs(follow_dist - vs.controller.desired_follow_time * vs.vehicle.state[3])
                follow_dist_error_history[step, i - 1] = follow_dist_error
            
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
                vs.set_next_ctrl(steering_angles[step:step+mpc_horizon], accelerations[step:step+mpc_horizon])
            if v2v:
                vs.emit_control_message()
                vs.emit_estimate_message()
                
        estimate_errors[step] = world.get_estimate_errors()
        
        for v in world.all_vehicle_systems: #
            v.update(timestep)
    if plot:
        plot_poses(position_history, desired_history)
        plot_desired_errors(follow_dist_error_history)

        average_errors = np.mean(estimate_errors, axis=0)
        pos_errors = average_errors[:,:,0]
        ang_errors = average_errors[:,:,1] * 180 / np.pi
        vel_errors = average_errors[:,:,2]
        plot_estimate_errors(pos_errors, "Position (m)")
        # plot_estimate_errors(ang_errors, "Angle (degrees)")
        # plot_estimate_errors(vel_errors, "Speed (m/s)")
    
    return np.sum(follow_dist_error_history)

if __name__ == '__main__':
    # run_scenario(num_cars=10)
    run_scenario(scenario="predictive", num_cars=10, mpc_horizon=10)