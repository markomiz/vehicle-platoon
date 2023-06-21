from helpers import *

gnss_uncertainty = 1.0  # Reasonable GNSS measurement uncertainty (in meters)
speedometer_uncertainty = 0.5  # Reasonable speedometer measurement uncertainty (in m/s)
compass_uncertainty = 0.01 # Reasonable compass measurement uncertainty
lidar_uncertainty = 0.1 # Reasonable lidar measurement uncertainty

class VehicleSystem:
    def __init__(self, vehicle, own_estimate, controller, other_estimates, world, id): 
        self.vehicle = vehicle
        self.estimator = own_estimate
        self.controller = controller
        self.other_estimates = other_estimates
        self.can_update = False
        self.world = world
        self.id = id

    def set_next_ctrl(self, steer, acc): 
        self.next_steer = steer
        self.next_acc = acc
        self.can_update = True

    def update(self, dt): 
        if self.can_update:
            self.vehicle.noisy_single_track_update(self.next_steer, self.next_acc, dt)
            self.estimator.prediction(self.next_steer, self.next_acc, dt)
            self.can_update = False

    def compute_follow_control(self, other_vehicle, dt):
        acc, steer = self.controller.compute_follow_control(other_vehicle, self.vehicle.state, dt )
        self.set_next_ctrl(steer, acc)

    def simulate_GNSS(self):
        measured_position = self.vehicle.get_state()[:2] + np.random.randn(2) * gnss_uncertainty
        self.estimator.gnss_measurement(measured_position, gnss_uncertainty)

    def simulate_compass(self):
        measured_heading = self.vehicle.get_state()[2] + np.random.randn() * compass_uncertainty
        self.estimator.compass_measurement(mod2pi(measured_heading), compass_uncertainty)

    def simulate_speedometer(self):
        measured_speed = self.vehicle.get_state()[3] + np.random.randn() * speedometer_uncertainty
        self.estimator.speedometer_measurement(measured_speed, speedometer_uncertainty)

    def simulate_lidar(self, vehicle_ahead):
        other_v_pos = vehicle_ahead.vehicle.get_state()[:2]
        real_diff = other_v_pos - self.vehicle.get_state()[:2]
        measured_lidar = real_diff + np.random.randn() * lidar_uncertainty
        lidar_covariance = np.diag([lidar_uncertainty ** 2, lidar_uncertainty ** 2])
        self.estimator.lidar_measurement(measured_lidar, other_v_pos, lidar_covariance, self.estimator.covariance[:2,:2])

    def recieve_control_message(self, other_id, steer, accelleration, dt):
        if len(self.other_estimates) > other_id: return
        # predict on recieving
        self.other_estimates[other_id].prediction(steer, accelleration, dt)

    def emit_control_message(self): 
        self.world.transmit_control_message(self.id, self.next_steer, self.next_acc, dt)