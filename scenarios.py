
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
    # Parameters
    timestep = 0.1
    road_points_x = np.linspace(-1000,1000, 20000)
    # road_points_y = np.piecewise(road_points_x, [road_points_x < 0, road_points_x >= 0], [0.0,0.0])
    road_points_y = np.piecewise(road_points_x, [road_points_x < 0, road_points_x >= 0], [0.0, lambda road_points_x: np.cos(road_points_x/ 15) * 2 - 2 ])
    road_points = np.stack((road_points_x,road_points_y), axis=1)
    road_speeds = np.ones(num_steps + 10) * 10

    if world is None:
        world = World(num_cars, mpc_horizon=mpc_horizon, network_loss=network_loss, road_points=road_points )

    position_history = np.zeros((num_steps, num_cars, 2))
    desired_history = np.zeros((num_steps, num_cars, 2))
    follow_dist_error_history = np.zeros((num_steps, num_cars -1))
    estimate_errors = np.zeros((num_steps, num_cars, num_cars, 3))
    # Run the estimation test loop
    for step in range(num_steps):

        for i, vs in enumerate(world.all_vehicle_systems):
            position_history[step, i, :] = vs.vehicle.get_state()[:2] * 1.0
            if i > 0 and step >0:
                follow_dist = np.linalg.norm(position_history[step, i, :] - desired_history[step -1, i, :])
                follow_dist_error = follow_dist #  - vs.controller.desired_follow_time * vs.vehicle.state[3]
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

            if vs.id > 0:
                # calculate control to follow car ahead
                if scenario == "predictive":
                    vs.compute_predictive_control( timestep )
                else:
                    vs.compute_follow_control( timestep )
                 
            else:
                vs.compute_follow_road_control( road_speeds[step:step + mpc_horizon ], timestep )

            desired_history[step, i, :] = vs.controller.desired_state[:2] * 1.0


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
    run_scenario(scenario="predictive", v2v=True, num_cars=10, mpc_horizon=5, network_loss=0.0, num_steps=200)