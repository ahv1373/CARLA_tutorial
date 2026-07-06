import glob
import os
import sys
import random
import time

try:
    sys.path.append(glob.glob(
        r'F:\carla\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg'
    )[0])
except IndexError:
    raise RuntimeError("Couldn't find the CARLA egg.")

import carla
from src.simulator_handler import SimulatorHandler
from src.utils.vehicle_command import VehicleCommand

class SimulationHyperParameters:
    def __init__(self, simulation_time: float = 30, town_name: str = "Town01",
                 ego_spawn_index: int = random.randint(0, 30), weather: carla.WeatherParameters = carla.WeatherParameters.ClearNoon,
                 use_autopilot: bool = True):  # Town choice are Mine_01 and Town10HD_Opt
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
    instance_segmentation_cam = simulator_handler.instance_segmentation_cam()
    gnss_sensor = simulator_handler.gnss()
    imu_sensor = simulator_handler.imu()
    lidar_sensor = simulator_handler.lidar()
    radar_sensor = simulator_handler.radar()
    collision_sensor = simulator_handler.collision()


    # listen to sensor data
    lidar_sensor.listen(lambda data: simulator_handler.lidar_callback(data))
    radar_sensor.listen(lambda data: simulator_handler.radar_callback(data))
    collision_sensor.listen(lambda event: simulator_handler.collision_callback(event))
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    instance_segmentation_cam.listen(lambda image: simulator_handler.instance_segmentation_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    if sim_parameters.use_autopilot is True:
        simulator_handler.vehicle.set_autopilot(True)
    else:
        VehicleCommand(throttle=1.0).send_control(simulator_handler.vehicle)
    # turn on the autopilot
    # simulator_handler.vehicle.set_autopilot(True)

    print(f"[INFO] Running simulation for {sim_parameters.simulation_time} seconds")

    # while True:
    #     # tick the world
    #     simulator_handler.world.tick()
    #
    #     # get the current simulation time
    #     current_time = simulator_handler.world.get_snapshot().timestamp.elapsed_seconds
    #     # check if the simulation time has elapsed
    #     if current_time >= sim_parameters.simulation_time:
    #         break

    time.sleep(sim_parameters.simulation_time)  # time.sleep is a blocking operation for letting the simulation run
    # at the very end of src/__main__.py
    import os

    simulator_handler.terminate()
    print("Exiting.")
    os._exit(0)   # hard-exit: skips interpreter teardown that triggers the fatal error

