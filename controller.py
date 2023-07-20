import numpy as np
from helpers import * 
from copy import deepcopy
from scipy.optimize import minimize
from itertools import chain
from collections import deque

class Controller:
    def __init__(self, kp=5.35, ki=5.23, kd=0.84, k_path=3.7, k_heading=2.5):
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

        self.buffer_size = 500
        self.prev_follows = deque(maxlen=self.buffer_size)

        self.dist_weight = 1.0
        self.angle_weight = 0.6
        self.vel_weight = 0.6
        self.discount_factor = 0.7

        

    def add_follow_state(self, state):
        # print(self.prev_follows)
        self.prev_follows.appendleft(state)

        if len(self.prev_follows) < 2:
            # give some invented linear history so we can always interpolate
            angle = mod2pi(state[2] + np.pi)
            for i in range(self.buffer_size):
                dist = 10.0 * i
                behind = state * 1.0
                behind[0] += dist * np.cos(angle)
                behind[1] += dist * np.sin(angle)
                self.prev_follows.append(behind)
    
    def replace_follow_states(self, states):
        self.add_follow_state(states[0]) # to shift the deque along
        for i, state in enumerate(states):
            self.prev_follows[i] = state



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
    
    def calculate_desired_states(self, states_to_follow ):
        states_to_follow = np.array(states_to_follow)
        follow_distances = states_to_follow[:,3] * self.desired_follow_time
        desired_states = [self.get_state_from_buffer(d) for d in follow_distances]
        return desired_states

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

    def compute_predictive_control(self, other_states, current_state, last_control, dt):
        self.add_follow_state(other_states[0])
        # calculate where I want to be in dt
        desired_states = self.calculate_desired_states(other_states)
        self.desired_state = desired_states[0]
        horizon = len(other_states)
        initial_guess = np.array([last_control for i in range(horizon)]) 
        initial_guess = initial_guess.flatten()
        acc_bounds = [(-self.max_acc,self.max_acc) for i in range(horizon) ]
        steer_bounds = [(-self.max_steer,self.max_steer) for i in range(horizon) ]
        bounds = list(chain.from_iterable(zip(acc_bounds,steer_bounds)))
        res = minimize(self.to_minimise, initial_guess, (desired_states, current_state, dt), method="Powell", tol=0.01, bounds=bounds)
        res = np.array(res.x)
        res = res.reshape((horizon,2))
        acc = res[:,0]
        steer = res[:,1]

        return acc, steer
    
    def to_minimise(self, all_x, all_desired, start_state, dt ):
        current_state = start_state
        # x = self.clip_control(x[0], x[1])
        total_error = 0.0
        current_discount = 1.0
        all_x = np.array(all_x)
        n_ctrl = int(len(all_x) / 2)
        all_x = all_x.reshape((n_ctrl, 2))
        for i, x in enumerate(all_x):
            new_state = self.simple_single_track(current_state, x, dt)
            desired = all_desired[i]
            diff = desired - new_state
            dist_error = (diff[0] **2 + diff[1] **2) * self.dist_weight
            angle_error = diff[2] **2 * self.angle_weight
            vel_error = diff[3] ** 2 * self.vel_weight
            total_cost = dist_error + angle_error + vel_error
            total_error += total_cost * current_discount
            current_discount *= self.discount_factor
        return total_error

    def simple_single_track(self, state,  controls, dt):
        wheelbase = 2.5
        new_state = state * 1.0
        new_state[2] += (new_state[3] * np.tan(controls[1])) / wheelbase * dt
        new_state[2] = mod2pi(new_state[2])
        new_state[3] += controls[0] * dt
        new_state[0] += new_state[3] * dt * np.cos(new_state[2])
        new_state[1] += new_state[3] * dt * np.sin(new_state[2])
        return new_state
