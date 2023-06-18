
class VehicleSystem:
    def __init__(self, vehicle, own_estimate, controller, other_vehicles): 
        self.vehicle = vehicle
        self.estimator = own_estimate
        self.controller = controller
        self.other_estimates = other_vehicles