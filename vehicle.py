import math
import random
import numpy as np
from helpers import *

class Vehicle:
    def __init__(self, state, wheelbase=2.5):
        self.state = np.array(state)  # [x, y, theta, speed]
        self.wheelbase = wheelbase

    def noisy_single_track_update(self, steering_angle, acceleration, timestep):
        # Add noise to the control inputs
        noisy_steering_angle = self.add_noise(steering_angle, 0.01)
        noisy_acceleration = self.add_noise(acceleration, 0.05)

        # Update the vehicle state
        delta_theta = (self.state[3] * math.tan(noisy_steering_angle)) / self.wheelbase * timestep
        rotation_matrix = np.array([[math.cos(self.state[2]), -math.sin(self.state[2])],
                                    [math.sin(self.state[2]), math.cos(self.state[2])]], dtype='float64')
        translation = np.array([self.state[3] * math.cos(self.state[2]) * timestep,
                               self.state[3] * math.sin(self.state[2]) * timestep], dtype='float64')

        self.state[:2] += np.matmul(rotation_matrix, translation)
        self.state[2] += delta_theta
        self.state[2] = mod2pi(self.state[2])
        self.state[3] += noisy_acceleration * timestep

    def add_noise(self, value, stddev):
        noise = random.gauss(0, stddev)
        return value + noise

    def get_state(self):
        return self.state