import numpy as np
import matplotlib.pyplot as plt
from vehicle import *
from vehicle_estimator import *
from controller import *
from vehicle_system import *
from helpers import *
from copy import deepcopy

# TODO 
""" 
- give each car an estimate for the others
- maybe instead of init with all cars, add a add_car method to add an estimator
- make estimators a key-estimator pair so they can be looked up by id
- s
"""

class World:
    def __init__(self, num_cars=2):
        self.all_vehicle_systems = []
        self.all_vehicle_estimates = dict.fromkeys(range(num_cars))

        for i in range(num_cars):
            # Create a Vehicle
            initial_state = np.array([-i * 0.5 * 10 , 0, 0.0, 10. ])  # [x, y, theta, speed]
            vehicle = Vehicle(initial_state)
            estimator = VehicleEstimator(deepcopy(initial_state), np.diag([1.,1.,0.1,1]))
            system = VehicleSystem(vehicle, Controller(), self, i)
            self.all_vehicle_systems.append(system)
            self.all_vehicle_estimates[i] = estimator
            system.next_steers = np.zeros(num_cars)
            system.next_accs = np.zeros(num_cars)
        
        for car in self.all_vehicle_systems:
            car.estimates = deepcopy(self.all_vehicle_estimates)

    def transmit_control_message(self, id, steer, accel):
        for vs in self.all_vehicle_systems:
            vs.recieve_control_message(id, steer, accel)

    def transmit_estimate(self, id, state, cov, from_id):
        for vs in self.all_vehicle_systems:
            vs.receive_estimate_message(id, state, cov, from_id)

    def set_controller_gains(self, params):
        for vs in self.all_vehicle_systems:
            vs.controller.kp = params[0]  # Proportional gain
            vs.controller.ki = params[1]  # Integral gain
            vs.controller.kd = params[2]  # Derivative gain
            vs.controller.k_path = params[3] # Path gain 
            vs.controller.k_heading = params[4] # heading gain

    def get_estimate_errors(self):
        # euclidean, angle, vel errors
        errors = np.zeros((len(self.all_vehicle_systems), len(self.all_vehicle_systems), 3))
        real_states = [v.vehicle.state for v in self.all_vehicle_systems]
        real_states = np.array(real_states)
        for i,v in enumerate(self.all_vehicle_systems):
            ests = [est.state for est in v.estimates.values()]
            ests = np.array(ests)
            diff = np.abs(real_states - ests)
            errors[i] = [[np.linalg.norm(d[:2]), d[2], d[3]] for d in diff]

        return errors 
