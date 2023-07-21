from helpers import *

gnss_uncertainty = 0.1  # Reasonable GNSS measurement uncertainty (in meters)
speedometer_uncertainty = 0.1  # Reasonable speedometer measurement uncertainty (in m/s)
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
        self.physics_per_step = 1
        self.last_control_update_times = []
        self.current_time = 0
        self.comm_dist = 2
        self.estimate_buffer = []

    def set_next_ctrl(self, steer, acc): 
        self.next_steers[self.id] = steer
        self.next_steers = np.array(self.next_steers,dtype='float64')
        self.next_accs[self.id] = acc
        self.next_accs = np.array(self.next_accs, dtype='float64')
        self.can_update = True

    def update(self, dt): 
        if self.can_update:
            for i in range(self.physics_per_step):
                self.vehicle.noisy_single_track_update(self.next_steers[self.id,0], self.next_accs[self.id,0], dt/float(self.physics_per_step))
            for  i, estimate in self.estimates.items():
                if len(self.next_steers[0]) > 1:
                    last_est = int(self.current_time - self.last_control_update_times[i])
                    last_est = min(last_est, len(self.next_steers[i]) - 1)
                    estimate.prediction(estimate.state, estimate.covariance, self.next_steers[i,last_est], self.next_accs[i,last_est], dt)
                else:
                    estimate.prediction(estimate.state, estimate.covariance, self.next_steers[i], self.next_accs[i], dt)
            self.can_update = False
            self.current_time += 1

    def compute_follow_control(self, dt):
        acc, steer = self.controller.compute_follow_control(self.estimates[self.id - 1].state * 1.0, self.estimates[self.id].state * 1.0, dt )
        self.set_next_ctrl(steer, acc)

    def compute_follow_road_control(self,speed, dt):
        acc, steer = self.controller.compute_follow_road_control(speed, self.estimates[self.id].state * 1.0, dt )
        self.set_next_ctrl(steer, acc)
        
    
    def compute_predictive_control(self, dt):
        idf = self.id -1
        predicted_states = self.estimates[idf].predictions(self.estimates[idf].state, self.estimates[idf].covariance,  self.next_steers[idf], self.next_accs[idf], dt)
        last_control = [self.next_accs[self.id,0],self.next_steers[self.id,0]] 
        acc, steer = self.controller.compute_predictive_control(predicted_states, self.estimates[self.id].state ,last_control, dt)
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

    def recieve_control_message(self, other_id, steer, accelleration, from_id):
        if (abs(from_id - self.id) > self.comm_dist): return
        self.next_steers[other_id] = steer
        self.next_accs[other_id] = accelleration
        self.last_control_update_times[other_id] = self.current_time

    def emit_control_message(self): 
        for i in range(len(self.next_steers)):
            self.world.transmit_control_message(i, self.next_steers[i], self.next_accs[i], self.id)

    def receive_estimate_message(self, id, state, covariance, from_id):
        if from_id == self.id: return
        if abs(self.id - from_id) > self.comm_dist: return 
        message = (id, state, covariance)
        self.estimate_buffer.append(message)

    def process_estimate_buffer(self):
        for message in self.estimate_buffer:
            self.estimates[message[0]].incorp_others_estimate(message[1], message[2])
        self.estimate_buffer = []

    def emit_estimate_message(self):
        # print(len(self.estimates), '  est len')
        for i in range(len(self.estimates)):
            est = self.estimates[i]
            # if (abs(self.id - int(i)) > 1): return 
            self.world.transmit_estimate(i, est.state, est.covariance, self.id)