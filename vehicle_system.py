from helpers import *

gnss_uncertainty = 1.0  # Reasonable GNSS measurement uncertainty (in meters)
speedometer_uncertainty = 0.5  # Reasonable speedometer measurement uncertainty (in m/s)
compass_uncertainty = 0.01 # Reasonable compass measurement uncertainty
lidar_uncertainty = 0.1 # Reasonable lidar measurement uncertainty

class VehicleSystem:
    def __init__(self, vehicle, controller, world, id): 
        self.vehicle = vehicle
        self.controller = controller
        self.estimates = {}
        self.can_update = False
        self.world = world
        self.id = id
        self.next_accs = []
        self.next_steers = []

    def set_next_ctrl(self, steer, acc): 
        self.next_steers[self.id] = steer
        self.next_accs[self.id] = acc
        self.can_update = True

    def update(self, dt): 
        if self.can_update:
            self.vehicle.noisy_single_track_update(self.next_steers[self.id], self.next_accs[self.id], dt)

            self.estimates[self.id].prediction(self.next_steers[self.id], self.next_accs[self.id], dt)
            self.can_update = False

    def compute_follow_control(self, other_state, dt):
        acc, steer = self.controller.compute_follow_control(other_state, self.vehicle.state, dt )
        self.set_next_ctrl(steer, acc)
    
    def compute_predictive_control(self, dt):
        
        self.predicted_state = self.estimates[self.id - 1].prediction(self.next_steers[self.id - 1], self.next_accs[self.id - 1], dt, False)
        acc, steer = self.controller.compute_follow_control(self.estimates[self.id - 1], self.vehicle.state, dt)
        self.set_next_ctrl(steer, acc)
        self.emit_control_message()

    def simulate_GNSS(self):
        measured_position = self.vehicle.get_state()[:2] + np.random.randn(2) * gnss_uncertainty
        self.estimates[self.id].gnss_measurement(measured_position, gnss_uncertainty)

    def simulate_compass(self):
        measured_heading = self.vehicle.get_state()[2] + np.random.randn() * compass_uncertainty
        self.estimates[self.id].compass_measurement(mod2pi(measured_heading), compass_uncertainty)

    def simulate_speedometer(self):
        measured_speed = self.vehicle.get_state()[3] + np.random.randn() * speedometer_uncertainty
        self.estimates[self.id].speedometer_measurement(measured_speed, speedometer_uncertainty)

    def simulate_lidar(self, other_vehicle): # maybe update to take into account angles
        other_v_pos = other_vehicle.vehicle.get_state()[:2]
        real_diff = other_v_pos - self.vehicle.get_state()[:2]
        measured_lidar = real_diff + np.random.randn() * lidar_uncertainty
        lidar_covariance = np.diag([lidar_uncertainty ** 2, lidar_uncertainty ** 2])
        self.estimates[self.id].lidar_measurement(measured_lidar, other_v_pos, lidar_covariance, self.estimates[self.id].covariance[:2,:2])

    def recieve_control_message(self, other_id, steer, accelleration):
        self.next_steers[other_id] = steer
        self.next_accs[other_id] = accelleration

    def emit_control_message(self): 
        self.world.transmit_control_message(self.id, self.next_steers[self.id], self.next_accs[self.id])

    # def process_join_request(self, other, pos): 
    #     # add to other vehicle estimates in correct way