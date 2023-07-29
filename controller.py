import numpy as np
from helpers import * 
from copy import deepcopy
from scipy.optimize import minimize
from itertools import chain
from collections import deque

class Controller:
    def __init__(self, kp=7.0, ki=.4, kd=1.2, k_path=2.0, k_heading=4.5):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.k_path = k_path # Path gain 
        self.k_heading = k_heading # heading gain
        self.long_error_integral = 0  # Integral of the error
        self.desired_follow_time = 0.5
        self.safety_dist = 0.5
        self.desired_state = np.zeros(4)
        self.max_steer = np.pi/5
        self.max_acc = 5.0

        self.buffer_size = 500
        self.prev_follows = deque(maxlen=self.buffer_size)

        self.dist_weight = 1.0
        self.angle_weight = 0.5
        self.vel_weight = 0.5
        self.acc_weight = 0.2
        self.steer_weight = 0.1
        self.discount_factor = 0.8

        self.anchor_to_road_point = False # rather than following the car in front, keep a distance but stay on the road
        
        self.road_points = []

    def get_closest_road_pos(self, pos):
        dists = np.linalg.norm(self.road_points - pos, axis=1)
        mindex = np.argmin(dists)
        closest_point = self.road_points[mindex] * 1.0
        return closest_point


    def clip_control(self, acc, steer):
        
        steer = np.clip(steer, - self.max_steer, self.max_steer)
        acc = np.clip(acc, -2 * self.max_acc, self.max_acc)
        
        return acc, steer

    def calculate_desired_state(self, state_to_follow , current_state):

        desired_state = state_to_follow *1.0
        # get point on road ahead
        dist = self.desired_follow_time * current_state[3] + self.safety_dist # needed for when car is not moving
        diff = state_to_follow - current_state
        angle = np.arctan2(diff[1], diff[0])
        desired_state[0] -= dist * np.cos(angle)
        desired_state[1] -= dist * np.sin(angle)
        if self.anchor_to_road_point:
            dists = np.linalg.norm(self.road_points - desired_state[:2], axis=1)
            mindex = np.argmin(dists)
            closest_point = self.road_points[mindex]
            road_point_diff = self.road_points[mindex] - self.road_points[mindex +1]
            desired_state[2] = np.arctan2(-road_point_diff[1], -road_point_diff[0]) 
            desired_state[:2] = closest_point
        return desired_state
    
    def calculate_desired_states(self, states_to_follow, current_state ):
        states_to_follow = np.array(states_to_follow, dtype='float64')
        follow_distances = states_to_follow[:,3] * self.desired_follow_time + self.safety_dist # needed for when car is not moving
        desired_states = states_to_follow * 1.0
        prev_state = current_state * 1.0
        for i, dist in enumerate(follow_distances):
            diff = desired_states[i] - prev_state
            angle = np.arctan2(diff[1], diff[0])
            desired_states[i,0] -= dist * np.cos(angle)
            desired_states[i,1] -= dist * np.sin(angle)
            prev_state = desired_states[i] * 1.0
            if self.anchor_to_road_point:
                dists = np.linalg.norm(self.road_points - desired_states[i,:2], axis=1)
                mindex = np.argmin(dists)
                closest_point = self.road_points[mindex]
                road_point_diff = self.road_points[mindex] - self.road_points[mindex +1]
                desired_states[i,2] = np.arctan2(-road_point_diff[1], -road_point_diff[0]) 
                desired_states[i,:2] = closest_point

        return desired_states

    def compute_follow_road_control(self, desired_speeds, current_state, dt):
        desired_states = np.zeros((len(desired_speeds), 4))
        next_ctrls = np.zeros((len(desired_speeds), 2))
        follow_distances = desired_speeds * (dt ) 
        last_state = current_state * 1.0
        next_state = last_state * 1.0
        for i, dist in enumerate(follow_distances):
            # get point on road ahead
            next_state[0] += dist * np.cos(last_state[2])
            next_state[1] += dist * np.sin(last_state[2])
            dists = np.linalg.norm(self.road_points - last_state[:2], axis=1)
            mindex = np.argmin(dists)
            closest_point = self.road_points[mindex]
            road_point_diff = self.road_points[mindex] - self.road_points[mindex +1]
            next_state[2] = np.arctan2(-road_point_diff[1], -road_point_diff[0]) 
            next_state[3] = desired_speeds[i]
            next_state[:2] = closest_point[:]
            desired_states[i] = next_state * 1.0
            next_ctrls[i] = self.reactive_control(last_state, next_state, dt, False)
            last_state = next_state * 1.0
        self.desired_state = desired_states[0] * 1.0
        # sinple control
        return next_ctrls[:,0], next_ctrls[:,1]

    def compute_follow_control(self, state_to_follow, current_state, dt):
        # Calculate the error between desired and current state
        self.desired_state = self.calculate_desired_state(state_to_follow, current_state)
        return self.reactive_control(current_state, self.desired_state,  dt)

    def reactive_control(self, current_state, desired_state,  dt, integral=True):
        heading_error = desired_state[2] - current_state[2] 
        angle = desired_state[2]

        diff = desired_state[:2] - current_state[:2]
        diff_angle = np.arctan2(diff[1], diff[0])
        dist = np.linalg.norm(diff)

        path_error = dist * np.sin(diff_angle - angle) 
        long_error = dist * np.cos(diff_angle - angle)
        # Proportional term
        proportional_term = self.kp * long_error

        # Integral term
        if integral:
            self.long_error_integral += long_error * dt
            integral_term = self.ki * self.long_error_integral
        else: integral_term = 0

        # Derivative term
        derivative_term = self.kd * (desired_state[3] - current_state[3]) / dt

        accelleration = proportional_term + integral_term + derivative_term
    
        steering_angle = heading_error * self.k_heading + path_error * self.k_path

        return self.clip_control(accelleration, steering_angle)



    def compute_predictive_optimal_control(self, other_states, current_state, last_control, dt):
        # self.add_follow_state(other_states[0])
        # calculate where I want to be in dt
        desired_states = self.calculate_desired_states(other_states, current_state)
        self.desired_state = desired_states[0] * 1.0
        horizon = len(other_states)
        initial_guess = np.array([last_control for i in range(horizon)], dtype='float64') 
        initial_guess = initial_guess.flatten()
        acc_bounds = [(-self.max_acc,self.max_acc) for i in range(horizon) ]
        steer_bounds = [(-self.max_steer,self.max_steer) for i in range(horizon) ]
        bounds = list(chain.from_iterable(zip(acc_bounds,steer_bounds)))
        res = minimize(self.to_minimise, initial_guess, (desired_states, current_state, dt), method="Powell", tol=0.001, bounds=bounds)
        res = np.array(res.x, dtype='float64')
        res = res.reshape((horizon,2))
        acc = res[:,0]
        steer = res[:,1]
        return acc, steer

    
    
    def to_minimise(self, all_x, all_desired, start_state, dt ):
        current_state = start_state * 1.0
        # x = self.clip_control(x[0], x[1])
        total_error = 0.0
        current_discount = 1.0
        all_x = np.array(all_x, dtype='float64')
        n_ctrl = int(len(all_x) / 2)
        all_x = all_x.reshape((n_ctrl, 2))
        for i, x in enumerate(all_x):
            new_state = self.simple_single_track(current_state, x, dt)
            desired = all_desired[i]
            diff = desired - new_state
            dist_error = (diff[0] **2 + diff[1] **2) * self.dist_weight
            angle_error = diff[2] **2 * self.angle_weight
            vel_error = diff[3] ** 2 * self.vel_weight
            acc_cost = x[0] ** 2 * self.acc_weight
            steer_cost = x[1] **2 * self.steer_weight
            total_cost = dist_error + angle_error + vel_error + steer_cost + acc_cost
            total_error += total_cost * current_discount
            current_discount *= self.discount_factor
            current_state = new_state * 1.0
        return total_error

    def simple_single_track(self, state,  controls, dt):
        wheelbase = 2.5
        new_state = state * 1.0
        new_state[2] += (new_state[3] * np.tan(controls[1])) / wheelbase * dt
        new_state[3] += controls[0] * dt
        new_state[0] += new_state[3] * dt * np.cos(new_state[2])
        new_state[1] += new_state[3] * dt * np.sin(new_state[2])
        return new_state
