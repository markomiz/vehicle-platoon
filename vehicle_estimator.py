import numpy as np
from helpers import *

class VehicleEstimator:
    def __init__(self, initial_state, initial_covariance, wheelbase=2.5):
        self.state = np.array(initial_state)  # [x, y, theta, speed]
        self.covariance = np.array(initial_covariance)
        self.wheelbase = wheelbase

    def prediction(self, steering_angle, acceleration, dt):
        # Motion model
        self.state[3] += acceleration * dt
        self.state[2] += (self.state[3] * np.tan(steering_angle)) / self.wheelbase * dt
        self.state[2] = mod2pi(self.state[2])
        self.state[3] += acceleration * dt

        self.state[0] += self.state[3] * dt * np.cos(self.state[2])
        self.state[1] += self.state[3] * dt * np.sin(self.state[2])

        # Jacobian of motion model
        F = np.array([[0, 0, -self.state[3] * np.sin(self.state[2]) * dt, np.cos(self.state[2]) * dt],
                      [0, 0, self.state[3] * np.cos(self.state[2]) * dt, np.sin(self.state[2]) * dt],
                      [0, 0, 0, dt* np.tan(steering_angle)/self.wheelbase],
                      [0, 0, 0, 0]])

        # Process noise covariance # TODO : get better idea of this...s
        Q = np.array([[0.05, 0, 0, 0],
                      [0, 0.05, 0, 0],
                      [0, 0, 0.05, 0],
                      [0, 0, 0, 0.05]])

        # Update the covariance based on motion model
        self.covariance = np.matmul(np.matmul(F, self.covariance), F.T) + Q

    def measurement(self, measured_state, measurement_model, measurement_covariance):
        # Compute the Jacobian of the measurement model
        H = self.compute_jacobian(measurement_model)
        # Calculate the measurement residual
        residual = measured_state - H @ self.state

        # Calculate the Kalman gain
        S = np.matmul(np.matmul(H, self.covariance), H.T) + measurement_covariance
        K = np.matmul(np.matmul(self.covariance, H.T), np.linalg.inv(S))

        # Update the state and covariance based on measurement
        self.state += np.matmul(K, residual)
        self.covariance = np.matmul((np.eye(4) - np.matmul(K, H)), self.covariance)

    def gnss_measurement(self, measured_position, uncertainty):
        measurement_model = lambda state: state[:2]  # Measurement model function
        measurement_covariance = np.diag([uncertainty ** 2, uncertainty ** 2])  # Measurement noise covariance

        self.measurement(measured_position, measurement_model, measurement_covariance)

    def lidar_measurement(self, measured_distance, object_position, object_covariance, measurement_covariance): # TODO update lidar model
        measurement_model = lambda state: object_position - state[:2]  # Measurement model function

        self.measurement(measured_distance, measurement_model, object_covariance + measurement_covariance)

    def speedometer_measurement(self, measured_speed, uncertainty):
        measurement_model = lambda state: np.array([state[3]])  # Measurement model function
        measurement_covariance = np.array([[uncertainty ** 2]])  # Measurement noise covariance

        self.measurement(measured_speed, measurement_model, measurement_covariance)

    def compass_measurement(self, measured_heading, uncertainty):
        measurement_model = lambda state: np.array([state[2]])  # Measurement model function
        measurement_covariance = np.array([[uncertainty ** 2]])  # Measurement noise covariance

        self.measurement(measured_heading, measurement_model, measurement_covariance)

    def compute_jacobian(self, function, epsilon=1e-5):
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
