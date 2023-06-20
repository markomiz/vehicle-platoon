
# create lead car (no estimates)

# create follow car with estimator and controller

# update lead car with arbitrary control inputs

# estimator does predictions/measurements for itself and the car it is folowing


import numpy as np
import matplotlib.pyplot as plt
from vehicle import *
from vehicle_estimator import *
from controller import *
from vehicle_system import *
from helpers import *
from copy import deepcopy



def simple_follow(num_cars=2):

    all_vehicle_systems = []
    all_vehicle_estimates = []
    for i in range(num_cars):
        # Create a Vehicle
        initial_state = np.array([-i * 0.5 * 10 , 0, 1.0, 10. ])  # [x, y, theta, speed]
        vehicle = Vehicle(initial_state)
        estimator = VehicleEstimator(deepcopy(initial_state), np.diag([1.,1.,0.1,1]))
        system = VehicleSystem(vehicle, estimator, Controller(), deepcopy(all_vehicle_estimates))
        all_vehicle_systems.append(system)
        all_vehicle_estimates.append(estimator)

    # Parameters
    num_steps = 50
    timestep = 0.1
    steering_angle = np.radians(0)  # Gentle steering angle
    acceleration = 0.0  # Gentle acceleration

    gnss_uncertainty = 1.0  # Reasonable GNSS measurement uncertainty (in meters)
    speedometer_uncertainty = 0.5  # Reasonable speedometer measurement uncertainty (in m/s)
    compass_uncertainty = 0.01 # Reasonable compass measurement uncertainty
    lidar_uncertainty = 0.1 # Reasonable lidar measurement uncertainty


    position_history = np.zeros((num_steps, num_cars + 1, 2))

    # Run the estimation test loop
    for step in range(num_steps):

        for i, vs in enumerate(all_vehicle_systems):
            
            # Measure GNSS position
            measured_position = vs.vehicle.get_state()[:2] + np.random.randn(2) * gnss_uncertainty
            vs.estimator.gnss_measurement(measured_position, gnss_uncertainty)

            # Measure speed
            measured_speed = vs.vehicle.get_state()[3] + np.random.randn() * speedometer_uncertainty
            vs.estimator.speedometer_measurement(measured_speed, speedometer_uncertainty)

            # Measure angle
            measured_heading = vs.vehicle.get_state()[2] + np.random.randn() * compass_uncertainty
            vs.estimator.compass_measurement(mod2pi(measured_heading), speedometer_uncertainty)

            # Lidar ahead
            if len(vs.other_estimates) > 0 : ## if not lead car
                vsyst_in_front = all_vehicle_systems[i-1]
                other_v_pos = vsyst_in_front.vehicle.get_state()[:2]
                real_diff = other_v_pos - vs.vehicle.get_state()[:2]
                measured_lidar = real_diff + np.random.randn() * lidar_uncertainty
                lidar_covariance = np.diag([lidar_uncertainty ** 2, lidar_uncertainty ** 2])
                vs.estimator.lidar_measurement(measured_lidar, other_v_pos, lidar_covariance, vsyst_in_front.estimator.covariance[:2,:2])

                # calculate control to follow car ahead
                vs.compute_follow_control( all_vehicle_systems[i-1].vehicle.get_state(), timestep)

                position_history[step, i+1, :] = vs.controller.desired_state[:2]
            else:
                vs.set_next_ctrl(steering_angle, acceleration)

            position_history[step, i, :] = vs.vehicle.get_state()[:2]

        for v in all_vehicle_systems: #
            v.update(timestep)

    # Plot the errors
    plt.figure()

    # plt.plot(reals[:,0], reals[:,1],label='real')
    # plt.plot(ests[:,0], ests[:,1],label='ests')
    # plt.plot(position_history[:, 0, 0], position_history[:,0,1], label='lead')
    plt.plot(position_history[:, 1, 0], position_history[:,1,1], label='Follower')
    plt.plot(position_history[:, 2, 0], position_history[:,2,1], label='Desired')
    plt.axis('square')
    plt.legend()
    plt.grid(True)
    plt.show()




simple_follow()