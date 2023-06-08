import carla
import numpy as np


class VehicleCommand:
    def __init__(self, throttle: float = 0.0, steer: float = 0.0, brake: float = 0.0,
                 hand_brake: bool = False, reverse: bool = False):
        self.time_stamp = None
        self.throttle = throttle
        self.steer = steer
        self.brake = brake
        self.hand_brake = hand_brake
        self.reverse = reverse

    def send_control(self, vehicle: carla.Vehicle):
        throttle = np.clip(self.throttle, 0.0, 1.0)
        steer = np.clip(self.steer, -1.0, 1.0)  # +1 is right, -1 is left
        brake = np.clip(self.brake, 0.0, 1.0)
        control = carla.VehicleControl(throttle, steer, brake, self.hand_brake, self.reverse)
        vehicle.apply_control(control)
