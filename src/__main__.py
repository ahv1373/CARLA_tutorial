import os
import sys
import glob
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append(r"D:\Term8\CARLA_0.9.8\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")

import carla
import time

sys.path.append(r"D:\CARLA_tutorial-main\src")
sys.path.append(r"D:\CARLA_tutorial-main\src\utils")
sys.path.append(r"D:\CARLA_tutorial-main\utils")

from simulator_handler import SimulatorHandler
from vehicle_command import VehicleCommand
from path_following_handler import PathFollowingHandler

if __name__ == "__main__":
    simulator_handler = SimulatorHandler(town_name="Town01")
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
    LIDAR_sensor = simulator_handler.LIDAR()
    radar_sensor = simulator_handler.radar()
    collision_sensor = simulator_handler.collision()

    # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    LIDAR_sensor.listen(lambda LIDAR: simulator_handler.LIDAR_callback(LIDAR))
    radar_sensor.listen(lambda radar: simulator_handler.radar_callback(radar))
    collision_sensor.listen(lambda collision: simulator_handler.collision_callback(collision))
    # VehicleCommand(throttle=1.0).send_control(simulator_handler.vehicle)
    # time.sleep(20.0)

    if path_following_handler.debug_mode:
        path_following_handler.start()
    else:
        ego_pid_controller = path_following_handler.pid_controller(ego_vehicle,
                                                                   path_following_handler.pid_values_lateral,
                                                                   path_following_handler.pid_values_longitudinal)
        
        path_following_handler.vehicle_and_controller_inputs(ego_vehicle, ego_pid_controller)
        path_following_handler.start()

