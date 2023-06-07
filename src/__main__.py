import time

import carla

from src.simulator_handler import SimulatorHandler
from utils.vehicle_command import VehicleCommand
from path_following_handler import PathFollowingHandler
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    simulator_handler = SimulatorHandler(town_name="Town10HD_Opt")

    simulator_handler.set_weather(weather=carla.WeatherParameters.ClearNoon)

    # potential weather choices are [ClearNoon, ClearSunset, CloudyNoon, CloudySunset,
    # WetNoon, WetSunset, MidRainyNoon, MidRainSunset, HardRainNoon, HardRainSunset,
    # SoftRainNoon, SoftRainSunset]
    path_following_handler = PathFollowingHandler(client=simulator_handler.client)
    ego_spawn_point = path_following_handler.ego_spawn_point
    ego_vehicle = simulator_handler.spawn_ego_vehicles(road_id=ego_spawn_point["road_id"],
                                                       filtered_points_index=ego_spawn_point["filtered_points_index"])
    # add sensors
    rgb_cam = simulator_handler.rgb_cam()
    gnss_sensor = simulator_handler.gnss()
    imu_sensor = simulator_handler.imu()

    # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))


    if path_following_handler.debug_mode:
        path_following_handler.start()
    else:
        ego_pid_controller = path_following_handler.pid_controller(ego_vehicle,
                                                                   path_following_handler.pid_values_lateral,
                                                                   path_following_handler.pid_values_longitudinal)
        path_following_handler.vehicle_and_controller_inputs(ego_vehicle, ego_pid_controller)
        path_following_handler.start()

