import numpy as np
from helpers import * 
from copy import deepcopy

class Controller:
    def __init__(self, kp=2.0, ki=0.1, kd=0.2, k_path=.5, k_heading=1.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.k_path = k_path # Path gain 
        self.k_heading = k_heading # heading gain
        self.long_error_integral = 0  # Integral of the error
        self.desired_follow_time = 0.5
        self.desired_state = np.zeros(4)
        self.max_steer = np.pi/5
        self.max_acc = 5.0

    def clip_control(self, acc, steer):
        steer = np.clip(steer, - self.max_steer, self.max_steer)
        acc = np.clip(acc, -2 * self.max_acc, self.max_acc)
        return acc, steer


    def compute_follow_control(self, following_state, current_state, dt):
        # Calculate the error between desired and current state
        self.desired_state = deepcopy(following_state)
        follow_distance = following_state[3] * self.desired_follow_time
        angle = mod2pi(following_state[2] + np.pi)
        self.desired_state[0] += np.cos(angle) * follow_distance
        self.desired_state[1] += np.sin(angle) * follow_distance

        heading_error = self.desired_state[2] - current_state[2] # 

        diff = self.desired_state[:2] - current_state[:2]
        dist = np.linalg.norm(diff)
        
        diff_angle = mod2pi( np.arctan2(diff[1], diff[0]) )

        path_error = - dist * np.sin(diff_angle - angle) 
        long_error = - dist * np.cos(diff_angle - angle)

        # Proportional term
        proportional_term = self.kp * long_error

        # Integral term
        self.long_error_integral += long_error * dt
        integral_term = self.ki * self.long_error_integral

        # Derivative term
        derivative_term = self.kd * (self.desired_state[3] - current_state[3]) / dt

        accelleration = proportional_term + integral_term + derivative_term

        steering_angle = heading_error * self.k_heading + path_error * self.k_path

        return self.clip_control(accelleration, steering_angle)