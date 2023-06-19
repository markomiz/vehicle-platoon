import numpy as np
from helpers import * 
from copy import deepcopy

class Controller:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, k_path=-0.1, k_heading=0.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.k_path = k_path # Path gain 
        self.k_heading = k_heading # heading gain
        self.long_error_integral = 0  # Integral of the error
        self.previous_long_error = 0  # Previous error
        self.desired_follow_time = 0.5
        self.desired_state = np.zeros(4)

    def compute_control(self, following_state, current_state, dt):
        # Calculate the error between desired and current state
        desired_state = deepcopy(following_state)
        follow_distance = following_state[3] * self.desired_follow_time

        angle = mod2pi(following_state[2] + np.pi)

        desired_state[0] += np.cos(angle) * follow_distance
        desired_state[1] += np.sin(angle) * follow_distance

        heading_error = desired_state[2] - current_state[2] # 

        diff = desired_state[:2] - current_state[:2]
        dist = np.linalg.norm(diff)

        diff_angle = desired_state[2] - np.arctan2(diff[1], diff[0])


        val = (float(desired_state[1] - current_state[1]) * (following_state[0] - desired_state[0])) - \
           (float(desired_state[0] - current_state[0]) * (following_state[1] - desired_state[1]))
        sign = 0
        if val > 0: sign = -1
        elif val < 0: sign = 1

        path_error = dist * np.sin(diff_angle)
        long_error = dist * np.cos(diff_angle)

        print("path error ", path_error)

        # Proportional term
        proportional_term = self.kp * long_error

        # Integral term
        self.long_error_integral += long_error * dt
        integral_term = self.ki * self.long_error_integral

        # Derivative term
        derivative_term = self.kd * (long_error - self.previous_long_error) / dt
        self.previous_long_error = long_error

        accelleration = proportional_term + integral_term + derivative_term

        steering_angle = heading_error * self.k_heading + path_error * self.k_path

        self.desired_state = desired_state

        return accelleration, steering_angle