import time

import carla

from src.simulator_handler import SimulatorHandler
from src.utils.vehicle_command import VehicleCommand


class SimulationHyperParameters:
    def __init__(self, simulation_time: float = 30.0, town_name: str = "Town04",
                 ego_spawn_index: int = 13, weather: carla.WeatherParameters = carla.WeatherParameters.ClearNoon,
                 use_autopilot: bool = True):
        self.simulation_time = simulation_time
        self.town_name = town_name
        self.ego_spawn_index = ego_spawn_index
        self.weather = weather
        # potential weather choices are [ClearNoon, ClearSunset, CloudyNoon, CloudySunset,
        # WetNoon, WetSunset, MidRainyNoon, MidRainSunset, HardRainNoon, HardRainSunset,
        # SoftRainNoon, SoftRainSunset]
        self.use_autopilot = use_autopilot


if __name__ == "__main__":
    # Simulation hyperparameters
    sim_parameters = SimulationHyperParameters()

    simulator_handler = SimulatorHandler(town_name=sim_parameters.town_name)
    simulator_handler.spawn_vehicle(spawn_index=sim_parameters.ego_spawn_index)
    simulator_handler.set_weather(weather=sim_parameters.weather)

    # add sensors
    rgb_cam = simulator_handler.rgb_cam()
    gnss_sensor = simulator_handler.gnss()
    imu_sensor = simulator_handler.imu()

    # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    if sim_parameters.use_autopilot:
        simulator_handler.vehicle.set_autopilot(True)
    else:
        VehicleCommand(throttle=1.0).send_control(simulator_handler.vehicle)
    # turn on the autopilot
    # simulator_handler.vehicle.set_autopilot(True)

    print(f"[INFO] Running simulation for {sim_parameters.simulation_time} seconds")
    time.sleep(sim_parameters.simulation_time)  # time.sleep is a blocking operation for letting the simulation run
    simulator_handler.terminate()  # cleanup the script and remove actors
