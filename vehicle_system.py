from helpers import *

gnss_uncertainty = 0.1  # Reasonable GNSS measurement uncertainty (in meters)
speedometer_uncertainty = 0.05  # Reasonable speedometer measurement uncertainty (in m/s)
compass_uncertainty = 0.01 # Reasonable compass measurement uncertainty
lidar_uncertainty = 0.01 # Reasonable lidar measurement uncertainty

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
            for  i, estimate in self.estimates.items():
                estimate.prediction(self.next_steers[i], self.next_accs[i], dt)
            self.can_update = False

    def compute_follow_control(self, dt):
        acc, steer = self.controller.compute_follow_control(self.estimates[self.id - 1].state, self.estimates[self.id].state, dt )
        self.set_next_ctrl(steer, acc)
        
    
    def compute_predictive_control(self, dt):
        self.predicted_state = self.estimates[self.id - 1].prediction(self.next_steers[self.id - 1], self.next_accs[self.id - 1], dt, False)
        last_control = [self.next_accs[self.id],self.next_steers[self.id]] 
        acc, steer = self.controller.compute_predictive_control(self.estimates[self.id - 1].state, self.estimates[self.id].state,last_control, dt)
        self.set_next_ctrl(steer, acc)
        
    
    def compute_averaged_control(self, dt):
        divisor = 2 # weighting of own control
        acc = self.next_accs[self.id] * 1.0
        steer = self.next_steers[self.id] * 1.0
        if self.id - 1 > 0:
            acc += self.next_accs[self.id - 1]
            steer = self.next_steers[self.id - 1]
            divisor += 1
        if self.id + 1 < len(self.next_steers):
            acc += self.next_accs[self.id + 1]
            steer += self.next_steers[self.id + 1]
            divisor += 1
        acc /= divisor
        steer /= divisor
        self.set_next_ctrl(steer, acc)


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
        measured_pos = self.estimates[self.id].state[:2] + real_diff + np.random.randn() * lidar_uncertainty
        lidar_covariance = np.diag([lidar_uncertainty ** 2, lidar_uncertainty ** 2])
        self.estimates[other_vehicle.id].lidar_measurement(measured_pos, lidar_covariance)

    def recieve_control_message(self, other_id, steer, accelleration):
        if other_id == self.id: return
        self.next_steers[other_id] = steer
        self.next_accs[other_id] = accelleration

    def emit_control_message(self): 
        self.world.transmit_control_message(self.id, self.next_steers[self.id], self.next_accs[self.id])

    def receive_estimate_message(self, id, state, covariance):
        if id == self.id: return # don't update own estimate with own estimate 
        self.estimates[id].incorp_others_estimate(state, covariance)

    def emit_estimate_message(self): 
        # own estimate
        self.world.transmit_estimate( self.id, self.estimates[self.id].state, self.estimates[self.id].covariance)
        # ahead estimate
        if self.id > 1:
            self.world.transmit_estimate( self.id - 1, self.estimates[self.id -1].state, self.estimates[self.id -1].covariance)
        # behind estimate
        if self.id + 1 < len(self.estimates):
            self.world.transmit_estimate( self.id + 1, self.estimates[self.id + 1].state, self.estimates[self.id + 1].covariance)


    # def process_join_request(self, other, pos): 
    #     # add to other vehicle estimates in correct way