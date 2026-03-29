#!/usr/bin/env python
# coding: utf-8

# In[1]:


import carla
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from utils.vehicle_command import VehicleCommand
import time
from path_following_handler import PathFollowingHandler
class SimulatorHandler:

    def __init__(self, town_name):
        self.spawn_point = None
        self.vehicle = None
        self.world = client.get_world()
            if os.path.basename(self.world.get_map().name) != town_name:
                self.world: carla.World = client.load_world(town_name)
                    
from src.simulator_handler import SimulatorHandler
from utils.vehicle_command import VehicleCommand
from simulator_handler import SimulatorHandler
from vehicle_command import VehicleCommand

if __name__ == "__main__":
    simulator_handler = SimulatorHandler(town_name="Town04")
    simulator_handler.spawn_vehicle(spawn_index=13)
    simulator_handler = SimulatorHandler(town_name="Town01")
    simulator_handler.set_weather(weather=carla.WeatherParameters.ClearNoon)
    #SENSOR
    rgb_cam = simulator_handler.rgb_cam()
    gnss_sensor = simulator_handler.gnss()
    imu_sensor = simulator_handler.imu()
    LIDAR_sensor = simulator_handler.LIDAR()
    radar_sensor = simulator_handler.radar()
    collision_sensor = simulator_handler.collision()
    
self.spawn_point = self.world.get_map().get_spawn_points()[spawn_index]
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, self.spawn_point)
        self.actor_list.append(self.vehicle)
        global vehicle2
        vehicle2 = self.vehicle    
def set_weather(self, weather=carla.WeatherParameters.ClearNoon):
        self.world.set_weather(weather)
 gnss_dict["altitude"] = gnss.altitude
        self.gnss_dataframe = self.gnss_dataframe.append(gnss_dict, ignore_index=True)
        self.gnss_dataframe.to_csv(os.path.join(self.save_dir, "gnss.csv"), index=False)

    def lidar(self):
        lidar_cam = None
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(32))
        lidar_bp.set_attribute('points_per_second', str(90000))
        lidar_bp.set_attribute('rotation_frequency', str(40))
        lidar_bp.set_attribute('range', str(20))
        lidar_location = carla.Location(0, 0, 2)
        lidar_rotation = carla.Rotation(0, 0, 0)
        lidar_transform = carla.Transform(lidar_location, lidar_rotation)
        lidar_sen = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle,
                                           attachment_type=carla.AttachmentType.Rigid)
        return lidar_sen

    def radar(self):
        rad_cam = None
        rad_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        rad_bp.set_attribute('horizontal_fov', str(35))
        rad_bp.set_attribute('vertical_fov', str(20))
        rad_bp.set_attribute('range', str(20))
        rad_location = carla.Location(x=2.0, z=1.0)
        rad_rotation = carla.Rotation(pitch=5)
        rad_transform = carla.Transform(rad_location, rad_rotation)
        rad_ego = self.world.spawn_actor(rad_bp, rad_transform, attach_to=self.vehicle,
                                         attachment_type=carla.AttachmentType.Rigid)
        return rad_ego

    def rad_callback(self, radar_data):
        velocity_range = 7.5  # m/s
        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)   
def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.world.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))
def spawn_vehicle(self, spawn_index: int = 90):
        self.vehicle_blueprint = self.blueprint_library.filter("model3")[0]  # choosing the car
        self.spawn_point = self.world.get_map().get_spawn_points()[spawn_index]
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, self.spawn_point)
        self.waypoints: list = self.client.get_world().get_map().generate_waypoints(distance=1.0)

   
    def spawn_ego_vehicles(self, road_id: int, filtered_points_index: int) -> Any:
        print("spawning ego vehicle at road_id={} filtered_points_index={}".format(road_id,
                                                                                   filtered_points_index))
        vehicle_blueprint =             self.client.get_world().get_blueprint_library().filter("model3")[0]
        filtered_waypoints = filter_waypoints(self.waypoints, road_id=road_id)
        spawn_point = filtered_waypoints[filtered_points_index].transform
        spawn_point.location.z += 2
        self.vehicle = self.client.get_world().spawn_actor(vehicle_blueprint, spawn_point)
        self.actor_list.append(self.vehicle)
        return self.vehicle            
            
def infinite_trajectory_tracking(self, vehicle, ego_pid_controller_):
        curr_waypoint_index = {'road_id': 13, 'filtered_points_index': 0}
        curr_waypoint = filter_waypoints(self.waypoints,
                                         road_id=curr_waypoint_index['road_id'])[curr_waypoint_index['filtered_points_index']]
        print(curr_waypoint)
        while True:
            way_points = curr_waypoint.next_until_lane_end(2)
            draw_waypoints(self.world, self.waypoints, road_id=curr_waypoint.road_id , life_time=30)
            for target_waypoint in way_points:
                self.__follow_target_waypoints__(vehicle, target_waypoint, ego_pid_controller_)
            curr_waypoint = target_waypoint.next(3.5)[0]    
            
def vehicle_and_controller_inputs(self, ego_vehicle_, ego_pid_controller_):
Self.ego_vehicle = ego_vehicle_
self.ego_pid_controller = ego_pid_controller_
def colision(self):
        blueprint_library = self.world.get_blueprint_library()

        collision_sensor = self.world.spawn_actor(blueprint_library.find('sensor.other.collision'),
                                                  carla.Transform(), attach_to=self.vehicle,
                                                  attachment_type=carla.AttachmentType.Rigid)
        return collision_sensor
def clearing(self):
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for actor in self.world.get_actors().filter('*sensor*'):
            actor.destroy()
def colision_callback(event):
        print("COLLISION")         
        


# In[ ]:




