import numpy as np
from helpers import *
from copy import deepcopy
from scipy.linalg import inv
from numpy import dot

class VehicleEstimator:
    def __init__(self, initial_state, initial_covariance, wheelbase=2.5):
        self.state = np.array(initial_state, dtype='float64')  # [x, y, theta, speed]
        self.covariance = np.array(initial_covariance, dtype='float64')
        self.wheelbase = wheelbase
        self.gate_measurements = False

    def prediction(self, state, cov, steering_angle, acceleration, dt, replace = True):
        # Motion model
        new_state = deepcopy(state)
        new_covariance = deepcopy(cov)
        new_state[3] += acceleration * dt
        new_state[2] += (new_state[3] * np.tan(steering_angle)) / self.wheelbase * dt
        new_state[2] = mod2pi(new_state[2])
        new_state[3] += acceleration * dt

        new_state[0] += new_state[3] * dt * np.cos(new_state[2])
        new_state[1] += new_state[3] * dt * np.sin(new_state[2])

        # Jacobian of motion model
        F = self.compute_prediction_jacobian(self._prediction_model, new_state, steering_angle, acceleration, dt)

        # Process noise covariance # TODO : get better idea of this...s
        pn = 0.05
        Q = np.array([[pn, 0, 0, 0],
                      [0, pn, 0, 0],
                      [0, 0, pn/10, 0],
                      [0, 0, 0, pn]], dtype='float64')

        # Update the covariance based on motion model
        new_covariance = np.matmul(np.matmul(F, new_covariance), F.T) + Q

        if replace:
            self.covariance = new_covariance
            self.state = new_state
        
        return new_state, new_covariance

    def predictions(self, state, cov, steers, accs, dt):
        new_state = state * 1.0
        new_cov = cov* 1.0
        all_states = []
        for i in range(len(steers)):
            new_state, new_cov, = self.prediction(new_state, new_cov, steers[i], accs[i], dt, replace=False)
            all_states.append(new_state * 1.0)
        return all_states


    def measurement(self, measured_state, measurement_model, measurement_covariance):

        # Compute the Jacobian of the measurement model
        H = self.compute_measurement_jacobian(measurement_model)
        # Calculate the measurement residual
        residual = measured_state - H @ self.state
        # Calculate the Kalman gain
        S = np.matmul(np.matmul(H, self.covariance), H.T) + measurement_covariance
        K = np.matmul(np.matmul(self.covariance, H.T), np.linalg.inv(S))
        # Update the state and covariance based on measurement
        self.state += np.matmul(K, residual)
        self.covariance = np.matmul((np.eye(4) - np.matmul(K, H)), self.covariance)

    def measurement(self, measured_state, measurement_model, measurement_covariance):

        # Compute the Jacobian of the measurement model
        H = self.compute_measurement_jacobian(measurement_model)
        # Predicted measurement
        z_hat = np.matmul(H, self.state)
        # Calculate the measurement residual
        residual = measured_state - z_hat
        # Calculate the Kalman gain
        S = np.matmul(np.matmul(H, self.covariance), H.T) + measurement_covariance

        # Gate size can be tuned based on your requirements
        if self.gating(measured_state, z_hat, S, gate_size=9) or not self.gate_measurements:
            K = np.matmul(np.matmul(self.covariance, H.T), np.linalg.inv(S))
            # Update the state and covariance based on measurement
            self.state += np.matmul(K, residual)
            self.covariance = np.matmul((np.eye(4) - np.matmul(K, H)), self.covariance)

    def gnss_measurement(self, measured_position, uncertainty):
        measurement_model = lambda state: state[:2]  # Measurement model function
        measurement_covariance = np.diag([uncertainty ** 2, uncertainty ** 2])  # Measurement noise covariance
        self.measurement(measured_position, measurement_model, measurement_covariance)

    def range_sensor_measurement(self, measured_pos,   measurement_covariance): # TODO update lidar model
        measurement_model = lambda state: state[:2]  # Measurement model function
        self.measurement(measured_pos, measurement_model, self.covariance[:2,:2] + measurement_covariance)

    def speedometer_measurement(self, measured_speed, uncertainty):
        measurement_model = lambda state: np.array([state[3]])  # Measurement model function
        measurement_covariance = np.array([[uncertainty ** 2]])  # Measurement noise covariance
        self.measurement(measured_speed, measurement_model, measurement_covariance)

    def compass_measurement(self, measured_heading, uncertainty):
        measurement_model = lambda state: np.array([state[2]])  # Measurement model function
        measurement_covariance = np.array([[uncertainty ** 2]])  # Measurement noise covariance
        self.measurement(measured_heading, measurement_model, measurement_covariance)

    def incorp_others_estimate(self, est_state, covariance):
        measurement_model = lambda state: state * 1.0
        self.measurement(est_state, measurement_model, covariance)

    def compute_prediction_jacobian(self, function, state, *args, epsilon=1e-7):
        num_states = len(state)
        num_outputs = len(function(state, *args))

        H = np.zeros((num_outputs, num_states))

        for i in range(num_states):
            perturbed_state = state.copy()
            perturbed_state[i] += epsilon

            perturbed_output = function(perturbed_state, *args)
            nominal_output = function(state, *args)

            H[:, i] = (perturbed_output - nominal_output) / epsilon

        return H

    def compute_measurement_jacobian(self, function, epsilon=1e-7):
        num_states = len(self.state)
        num_outputs = len(function(self.state))

        H = np.zeros((num_outputs, num_states))

        for i in range(num_states):
            perturbed_state = self.state.copy()
            perturbed_state[i] += epsilon

            perturbed_output = function(perturbed_state)
            nominal_output = function(self.state)

            H[:, i] = (perturbed_output - nominal_output) / epsilon

        return H

    def get_state(self):
        return self.state

    def get_covariance(self):
        return self.covariance

    def _prediction_model(self, state, steering_angle, acceleration, dt):
        new_state = deepcopy(state)
        new_state[3] += acceleration * dt
        new_state[2] += (new_state[3] * np.tan(steering_angle)) / self.wheelbase * dt
        new_state[2] = mod2pi(new_state[2])
        new_state[3] += acceleration * dt

        new_state[0] += new_state[3] * dt * np.cos(new_state[2])
        new_state[1] += new_state[3] * dt * np.sin(new_state[2])
        return new_state

    def gating(self, z, z_hat, S, gate_size=9):

        D_squared = dot(dot((z - z_hat).T, inv(S)), (z - z_hat))
        if D_squared <= gate_size:
            return True
        else:
            return False