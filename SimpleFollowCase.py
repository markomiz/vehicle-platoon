
import numpy as np
import matplotlib.pyplot as plt
from vehicle import *
from vehicle_estimator import *
from controller import *
from vehicle_system import *
from helpers import *
from copy import deepcopy

class world:
    def __init__(self, num_cars=2):
        self.all_vehicle_systems = []
        self.all_vehicle_estimates = []
        for i in range(num_cars):
            # Create a Vehicle
            initial_state = np.array([-i * 0.5 * 10 , 0, 0.0, 10. ])  # [x, y, theta, speed]
            vehicle = Vehicle(initial_state)
            estimator = VehicleEstimator(deepcopy(initial_state), np.diag([1.,1.,0.1,1]))
            system = VehicleSystem(vehicle, estimator, Controller(), deepcopy(self.all_vehicle_estimates), self, i)
            self.all_vehicle_systems.append(system)
            self.all_vehicle_estimates.append(estimator)


    def transmit_control_message(self, id, steer, accel):
        for vs in self.all_vehicle_systems:
            vs.recieve_control_message(id, steer, accel)



def simple_follow(num_cars=2):

    all_vehicle_systems = []
    all_vehicle_estimates = []
    for i in range(num_cars):
        # Create a Vehicle
        initial_state = np.array([-i * 0.5 * 10 , 0, 0.0, 10. ])  # [x, y, theta, speed]
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

    position_history = np.zeros((num_steps, num_cars + 1, 2))

    # Run the estimation test loop
    for step in range(num_steps):

        for i, vs in enumerate(all_vehicle_systems):
            
            # Measure GNSS position
            vs.simulate_GNSS( )

            # Measure speed
            vs.simulate_speedometer( )

            # Measure angle
            vs.simulate_compass( )

            # Lidar ahead
            if len(vs.other_estimates) > 0 : ## if not lead car
                vsyst_in_front = all_vehicle_systems[i-1]
                vs.simulate_lidar( vsyst_in_front)

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