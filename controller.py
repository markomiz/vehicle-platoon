import numpy as np
from helpers import * 
from copy import deepcopy
from scipy.optimize import minimize

class Controller:
    def __init__(self, kp=2.0, ki=0.1, kd=0.2, k_path=2.0, k_heading=0.8):
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

        self.prev_follows = []
        self.buffer_size = 10

        self.dist_weight = 1.0
        self.angle_weight = 1.0
        self.vel_weight = 0.0

    def add_follow_state(self, state):
        # print(self.prev_follows)
        self.prev_follows.insert(0,state)
        if len(self.prev_follows) > self.buffer_size:
            self.prev_follows.pop()
        if len(self.prev_follows) < 2:
            # give some invented linear history so we can always interpolate
            angle = mod2pi(state[2] + np.pi)
            dist = 10000.0
            behind = state * 1.0
            behind[0] += dist * np.cos(angle)
            behind[1] += dist * np.sin(angle)
            self.prev_follows.append(behind)

    def get_state_from_buffer(self, s):
        total = 0.0
        index = 0
        # go as far back into the history as needed and interpolate
        while True:
            previous = self.prev_follows[index]
            current = self.prev_follows[index + 1]
            index += 1
            diff = current - previous
            diff[2] = mod2pi(diff[2])
            dist = np.linalg.norm(diff[:2])

            remaining = s - total
            if dist < remaining:
                total += dist
            else:
                interpolated = previous + (remaining / dist) * diff
                return interpolated


    def clip_control(self, acc, steer):
        steer = np.clip(steer, - self.max_steer, self.max_steer)
        acc = np.clip(acc, -2 * self.max_acc, self.max_acc)
        return acc, steer

    def calculate_desired_state(self, state_to_follow ):
        
        follow_distance = state_to_follow[3] * self.desired_follow_time
        desired_state = self.get_state_from_buffer(follow_distance)

        return desired_state


    def compute_follow_control(self, state_to_follow, current_state, dt):
        self.add_follow_state(state_to_follow)

        # Calculate the error between desired and current state
        self.desired_state = self.calculate_desired_state(state_to_follow)
        heading_error = self.desired_state[2] - current_state[2] 
        angle = state_to_follow[2]

        diff = self.desired_state[:2] - current_state[:2]
        diff_angle = np.arctan2(diff[1], diff[0])
        dist = np.linalg.norm(diff)

        path_error = dist * np.sin(diff_angle - angle) 
        long_error = dist * np.cos(diff_angle - angle)

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

    def compute_predictive_control(self, other_state, current_state, dt):
        self.add_follow_state(other_state)
        # calculate where I want to be in dt
        desired_state = self.calculate_desired_state(other_state)
        self.desired_state = desired_state

        res = minimize(self.to_minimise, [0.0,0.0], (desired_state, current_state, dt), method='BFGS')
        steer = res.x[0]
        acc = res.x[1]

        return self.clip_control(acc, steer)
    
    def to_minimise(self, x, desired, current_state, dt ):
        new_state = self.simple_single_track(current_state, x, dt)
        diff = new_state - desired
        dist_error = (diff[0] **2 + diff[1] **2) * self.dist_weight
        angle_error = diff[2] * self.angle_weight
        vel_error = diff[3] * self.vel_weight
        total_cost = dist_error + angle_error + vel_error
        return total_cost

    def simple_single_track(self, state,  controls, dt):
        wheelbase = 2.5
        new_state = state * 1.0
        new_state[2] += (new_state[3] * np.tan(controls[0])) / wheelbase * dt
        new_state[2] = mod2pi(new_state[2])
        new_state[3] += controls[1] * dt
        new_state[0] += new_state[3] * dt * np.cos(new_state[2])
        new_state[1] += new_state[3] * dt * np.sin(new_state[2])
        return new_state