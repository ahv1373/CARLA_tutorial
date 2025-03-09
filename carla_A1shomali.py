#!/usr/bin/env python
# coding: utf-8

# In[2]:

import time
import sys

import carla

import cv2

import pandas as pd

SimulatorHandler(town_name="Town04")
    simulator_handler.spawn_vehicle(spawn_index=13)

    
      rgb_cam = simulator_handler.rgb_cam()
        
    gnss_sensor = simulator_handler.gnss()
    
    
    imu_sensor = simulator_handler.imu()
    lidar = simulator_handler.lidar()
    
    
    radar = simulator_handler.radar()
    
    collision = simulator_handler.collision()
    
     # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    
    
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    
    
    lidar_sensor.listen(lambda lidar_sensor: simulator_handler.lidar_callback(lidar_sensor))
    
    
    radar_sensor.listen(lambda radar_sensor: simulator_handler.radar_callback(radar_sensor))
    
    
    colision_sensor.listen(lambda colision_sensor: simulator_handler.colision_callback(colision_sensor))
    VehicleCommand(throttle=1.0).send_control(simulator_handler.vehicle)
    time.sleep(20.0)
    
           client.set_timeout(8.0)
        
            self.world = client.get_world()
            
            if os.path.basename(self.world.get_map().name) != town_name:
                
                self.world: carla.World = client.load_world(town_name)
                    
                self.world: carla.World = client.load_world(town_name, reset_settings=True)

            self.blueprint_library = self.world.get_blueprint_library()
            
            self.actor_list = []
            
            self.vehicle_list = []


# In[ ]:




