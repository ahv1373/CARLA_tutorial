import time

import carla
import glob
import os
import sys

from src.simulator_handler import SimulatorHandler
from utils.vehicle_command import VehicleCommand
from utils.carla_utils import draw_waypoints, filter_waypoints, TrajectoryToFollow, InfiniteLoopThread
import warnings

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    simulator_handler = SimulatorHandler(town_name="Town03")
    simulator_handler.spawn_vehicle(spawn_index=30)
    simulator_handler.set_weather(weather=carla.WeatherParameters.ClearNoon)

    # potential weather choices are [ClearNoon, ClearSunset, CloudyNoon, CloudySunset,
    # WetNoon, WetSunset, MidRainyNoon, MidRainSunset, HardRainNoon, HardRainSunset,
    # SoftRainNoon, SoftRainSunset]
    
    ego_spawn_point = path_following_handler.ego_spawn_point
    filtered_waypoints = filter_waypoints(path_following_handler.waypoints, road_id=ego_spawn_point["road_id"])
    spawn_point = filtered_waypoints[ego_spawn_point["filtered_points_index"]].transform
    spawn_point.location.z += 2
    vehicle = client.get_world().spawn_actor(vehicle_blueprint, spawn_point)
    actor_list.append(vehicle)

    # add sensors
    
    LIDAR_sensor = simulator_handler.LIDAR()
    radar_sensor = simulator_handler.radar()
    collision_sensor = simulator_handler.collision()
    rgb_cam = simulator_handler.rgb_cam()
    gnss_sensor = simulator_handler.gnss()
    imu_sensor = simulator_handler.imu()
    rgb_cam = simulator_handler.rgb_cam(vehicle)
    gnss_sensor = simulator_handler.gnss(vehicle)
    imu_sensor = simulator_handler.imu(vehicle)
    lidar = simulator_handler.lidar(vehicle)
    radar = simulator_handler.radar(vehicle)
    collision = simulator_handler.collision(vehicle)

    # listen to sensor data
    
    LIDAR_sensor.listen(lambda LIDAR: simulator_handler.LIDAR_callback(LIDAR))
    lidar.listen(lambda data: simulator_handler.lidar_callback(data))
    radar.listen(lambda data: simulator_handler.radar_callback(data))
    collision.listen(lambda event: simulator_handler.collision_callback(event))
    radar_sensor.listen(lambda radar: simulator_handler.radar_callback(radar))
    collision_sensor.listen(lambda collision: simulator_handler.collision_callback(collision))
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    VehicleCommand(throttle=1.0).send_control(simulator_handler.vehicle)
    
     if path_following_handler.debug_mode:
        path_following_handler.start()
    else:        
        ego_pid_controller = path_following_handler.pid_controller(vehicle,
                                                                   path_following_handler.pid_values_lateral,
                                                                   path_following_handler.pid_values_longitudinal)

        path_following_handler.vehicle_and_controller_inputs(vehicle, ego_pid_controller)
        path_following_handler.start()
    
   time.sleep(1000.0)



