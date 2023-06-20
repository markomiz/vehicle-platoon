
class VehicleSystem:
    def __init__(self, vehicle, own_estimate, controller, other_vehicles): 
        self.vehicle = vehicle
        self.estimator = own_estimate
        self.controller = controller
        self.other_estimates = other_vehicles
        self.can_update = False

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

    