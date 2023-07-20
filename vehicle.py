
import random
import numpy as np
from helpers import *

class Vehicle:
    def __init__(self, state, wheelbase=2.5):
        self.state = np.array(state, dtype='float64')  # [x, y, theta, speed]
        self.wheelbase = wheelbase

    def noisy_single_track_update(self, steering_angle, acceleration, dt):
        # Add noise to the control inputs
        noisy_steering_angle =  self.add_noise(steering_angle, 0.01)
        noisy_acceleration = self.add_noise(acceleration, 0.05)

        self.state[2] += (self.state[3] * np.tan(noisy_steering_angle)) / self.wheelbase * dt
        self.state[2] = mod2pi(self.state[2])
        self.state[3] += noisy_acceleration * dt

        self.state[0] += self.state[3] * dt * np.cos(self.state[2])
        self.state[1] += self.state[3] * dt * np.sin(self.state[2])

    def add_noise(self, value, stddev):
        noise = random.gauss(0, stddev)
        return value + noise

    def get_state(self):
        return self.state