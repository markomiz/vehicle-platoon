
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
from scipy import signal
import cProfile


def create_test_road(num_steps, speeds="constant"):
    road_points_x = np.linspace(-100,200, 30000)
    # road_points_y = np.piecewise(road_points_x, [road_points_x < 0, road_points_x >= 0], [0.0,0.0])
    road_points_y = np.piecewise(road_points_x, [road_points_x < 0, road_points_x >= 0], [0.0, lambda road_points_x: np.cos(road_points_x/ 15) * 4 - 4 ])
    road_points = np.stack((road_points_x,road_points_y), axis=1)
    c = np.linspace(0,num_steps + 100, num_steps+100)
    if speeds == "constant": # cruising
        road_speeds = np.piecewise(c,[c < num_steps / 4, c > num_steps / 4], [10.0,10.0])
    elif speeds=="step": # increase in speed limit
        road_speeds = np.piecewise(c,[c < num_steps / 4, c > num_steps / 4], [10.0,15.0])
    elif speeds =="stop": # emergency stop
        road_speeds = np.piecewise(c,[ c < num_steps / 2, c > num_steps / 2], [10.0,0.0]) 

    return road_points, road_speeds

def run_scenario(scenario="simple", num_steps=100, road_speeds="constant", num_cars=2, world=None, plot=True, mpc_horizon=1, v2v=True, network_loss=0.0):
    # Parameters
    dt = 0.1

    road_points, road_speeds = create_test_road(num_steps, speeds=road_speeds)

    if world is None:
        world = World(num_cars, mpc_horizon=mpc_horizon, network_loss=network_loss, road_points=road_points )

    state_history = np.zeros((num_steps, num_cars, 4))
    acc_history = np.zeros((num_steps, num_cars))
    desired_history = np.zeros((num_steps, num_cars, 4))
    dists_from_desired = np.zeros((num_steps -1, num_cars -1))
    estimates_of_ahead = np.zeros((num_steps, num_cars -1, 4))
    estimate_errors = np.zeros((num_steps, num_cars, num_cars, 3))
    vehicle_gap = np.zeros((num_steps -1, num_cars -1))
    gap_minus_desired_gap_sq = np.zeros((num_steps -1, num_cars -1))
    # Run the estimation test loop
    for step in range(num_steps):

        # SIMULATE MEASUREMENTS
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
                vs.simulate_range( vsyst_in_front)
            if i + 1 < len(world.all_vehicle_systems):
                vsyst_behind = world.all_vehicle_systems[i+1]
                vs.simulate_range( vsyst_behind )

            
        # SIMULATE INFO EXCHANGE
            if v2v:
                vs.emit_estimate_message()
        
        before_share_error = world.get_estimate_errors()
        if v2v:
            for vs in world.all_vehicle_systems:
                vs.process_estimate_buffer()

        estimate_errors[step] = world.get_estimate_errors()

        estimate_improvement = before_share_error - estimate_errors[step]
        # print(np.sum(estimate_improvement))
        

        # CALCULATE CONTROLS
        for i,vs in enumerate(world.all_vehicle_systems):
            if vs.id > 0:
                estimates_of_ahead[step, i-1] = vs.estimates[i - 1].state
                # calculate control to follow car ahead
                if scenario == "predictive":
                    vs.compute_predictive_optimal_control( dt )
                else:
                    vs.compute_follow_control( dt )
            else:
                vs.compute_follow_road_control( road_speeds[step:step + mpc_horizon ], dt )
            
            if v2v and scenario == "predictive":
                vs.emit_control_message()

            if len(vs.next_accs[i]) == 1:
                acc_history[step, i] = vs.next_accs[i]
            else:
                acc_history[step, i] = vs.next_accs[i][0]
            desired_history[step, i, :] = vs.controller.desired_state * 1.0
            
            
        # UPDATE 
        for i,vs in enumerate(world.all_vehicle_systems): #
            vs.update(dt)
            
            state_history[step, i, :] = vs.vehicle.get_state() 
            if i > 0 and step >0:
                desired_gap = vs.vehicle.get_state()[3] * vs.controller.desired_follow_time + vs.controller.safety_dist
                dist_from_desired = np.linalg.norm(state_history[step, i, :2] - desired_history[step  , i, :2])
                gap = np.linalg.norm(state_history[step, i, :2] - state_history[step, i - 1, :2])
                if state_history[step, i, 0] > state_history[step, i - 1, 0]: gap = -gap
                dists_from_desired[step -1, i - 1] = dist_from_desired
                vehicle_gap[step -1, i-1] = gap
                gap_minus_desired_gap_sq[step -1, i-1] = (gap - desired_gap) **2
                # estimates_of_ahead[step, i-1, :] = vs.estimates[i - 1].state[:2]
                
                
    if plot:
        # plot_poses(state_history, desired_history, "Real Position",  "Desired Position", True)
        # plot_poses( estimates_of_ahead, desired_history[:, 1:, :2],"Estimated", "Desired",  False)
        # plot_poses(estimates_of_ahead, state_history[:,:-1, :2], "Estimated", "Real",  False)

        # plot_single(estimates_of_ahead[:,:,3], "Speed Estimate")
        # plot_single(state_history[:,:,3], "Speed History")
        # plot_single(acc_history, "Accellerations")
        # plot_desired_errors(dists_from_desired, "Dist From Desired")
        # plot_desired_errors(vehicle_gap, "Gap to Vehicle Ahead")
        # plot_desired_errors(np.sqrt(gap_minus_desired_gap_sq), "Follow Distance Error")

        average_errors = np.mean(estimate_errors, axis=0)
        pos_errors = average_errors[:,:,0]
        ang_errors = average_errors[:,:,1] * 180 / np.pi
        vel_errors = average_errors[:,:,2]
        plot_estimate_errors(pos_errors, "Position (m)")
        # plot_estimate_errors(ang_errors, "Angle (degrees)")
        # # plot_estimate_errors(vel_errors, "Speed (m/s)")
        # print("TOTAL POS EST ERROR: ", np.sum( np.linalg.norm(state_history[:,1:, :2] - estimates_of_ahead[:,:,:2], axis=1)  ))
        # print("TOTAL ANGLE EST ERROR: ", np.sum( np.linalg.norm(state_history[:,1:, 2] - estimates_of_ahead[:,:,2], axis=1)  ))
        # print("TOTAL SPEED EST ERROR: ", np.sum( np.linalg.norm(state_history[:,1:, 3] - estimates_of_ahead[:,:,3], axis=1)  ))
        print("dists from desired: " , np.sum(dists_from_desired))
    return np.sum(gap_minus_desired_gap_sq)


def compare_methods(num_runs, mpcs, speeds, num_steps=200):
    num_controls = 2 + len(mpcs)
    ## create data store
    data = np.zeros((num_controls, num_runs))

    ## fill data

    for run in range(num_runs):
        print(run)
        data[0,run] = run_scenario(scenario="simple", v2v=False, num_cars=10, num_steps=num_steps, plot=False, road_speeds=speeds)
        data[1,run] = run_scenario(scenario="simple", v2v=True, num_cars=10, num_steps=num_steps, plot=False, road_speeds=speeds)
        for i, mpc in enumerate(mpcs):
            # run_scenario(scenario="simple", v2v=True, num_cars=10, num_steps=num_steps)
            data[2+i, run] = run_scenario(scenario="predictive", v2v=True, num_cars=10, mpc_horizon=mpc, network_loss=0.00, num_steps=num_steps, plot=False, road_speeds=speeds)

    ## produce table with mean stdev for each control
    means = np.mean(data, axis=1)

    stds = np.std(data, axis=1)

    print("full table: ", data)
    print("means: ", means)
    print("stds: ", stds)

        

if __name__ == '__main__':
    
    score = run_scenario(scenario="simple", v2v=False, num_cars=5, num_steps=200, road_speeds="constant")
    print("reactive: ", score)
    score = run_scenario(scenario="simple", v2v=True, num_cars=5, num_steps=200, road_speeds="constant")
    print("reactive, communicative: ",score)
    score = run_scenario(scenario="predictive", mpc_horizon=9, v2v=True, num_cars=5, num_steps=200, road_speeds="constant")
    print("predictive: ", score)
    # cProfile.run("score = run_scenario(scenario='predictive', v2v=True, num_cars=5, mpc_horizon=5, network_loss=0.00, num_steps=20, road_speeds='stop', plot=False)")

    # compare_methods(5, [1,5,9], "constant")