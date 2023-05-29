import os

import carla
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from utils.vehicle_command import VehicleCommand
import time


class SimulatorHandler:

    def __init__(self, town_name):
        self.radar_dataframe = None
        self.spawn_point = None
        self.vehicle = None
        self.rgb_cam_sensor = None
        self.vehicle_blueprint = None

        # create data save directories (if they don't exist)
        self.save_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(os.path.join(self.save_dir, "rgb_cam")):
            os.makedirs(os.path.join(self.save_dir, "rgb_cam"))

        try:
            print("Trying to communicate with the client...")
            client = carla.Client("127.0.0.1", 2000)
            client.set_timeout(38.0)
            self.world = client.get_world()
            if os.path.basename(self.world.get_map().name) != town_name:
                self.world: carla.World = client.load_world(town_name)

            self.blueprint_library = self.world.get_blueprint_library()
            self.actor_list = []
            self.vehicle_list = []
            self.IM_WIDTH = 1024  # Ideally a config file should be defined for such parameters
            self.IM_HEIGHT = 720
            print("Successfully connected to CARLA client")
        except Exception as error:
            raise Exception(f"Error while initializing the simulator: {error}")

        self.imu_dataframe = pd.DataFrame({})
        self.gnss_dataframe = pd.DataFrame({})

    def spawn_vehicle(self, spawn_index: int = 90):
        self.vehicle_blueprint = self.blueprint_library.filter("model3")[0]  # choosing the car
        self.spawn_point = self.world.get_map().get_spawn_points()[spawn_index]
        self.vehicle = self.world.spawn_actor( self.vehicle_blueprint, self.spawn_point)
        self.actor_list.append(self.vehicle)

    def set_weather(self, weather=carla.WeatherParameters.ClearNoon):
        self.world.set_weather(weather)

    def rgb_cam(self):
        rgb_camera = self.blueprint_library.find("sensor.camera.rgb")
        rgb_camera.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        rgb_camera.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        rgb_camera.set_attribute("fov", "110")
        rgb_camera.set_attribute('sensor_tick', '0.0')
        spawn_point_rgb = carla.Transform(carla.Location(x=2.5, y=0, z=0.9),
                                          carla.Rotation(pitch=-5, roll=0, yaw=0))

        self.rgb_cam_sensor = self.world.spawn_actor(rgb_camera, spawn_point_rgb, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_cam_sensor)
        return self.rgb_cam_sensor

    def gnss(self):
        gnss_sensor = self.blueprint_library.find("sensor.other.gnss")
        gnss_sensor.set_attribute("sensor_tick", str(0.0))
        gnss_location = carla.Location(0, 0, 0)
        gnss_rotation = carla.Rotation(0, 0, 0)
        gnss_transform = carla.Transform(gnss_location, gnss_rotation)
        ego_gnss = self.world.spawn_actor(gnss_sensor, gnss_transform, attach_to=self.vehicle,
                                          attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(ego_gnss)
        return ego_gnss

    def imu(self):
        imu_sensor = self.blueprint_library.find("sensor.other.imu")
        imu_location = carla.Location(0, 0, 0)
        imu_rotation = carla.Rotation(0, 0, 0)
        imu_transform = carla.Transform(imu_location, imu_rotation)
        ego_imu = self.world.spawn_actor(imu_sensor, imu_transform, attach_to=self.vehicle,
                                         attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(ego_imu)
        return ego_imu

    def rgb_cam_callback(self, image):
        image.save_to_disk("data/rgb_cam/%06d.jpg" % image.frame)
        raw_image = np.array(image.raw_data)

        rgba_image = raw_image.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))  # because carla rgb cam is rgba
        rgb_image = rgba_image[:, :, :3]
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.show()
        # cv2.imshow("rgb camera", rgb_image)  # FixMe: Replace with pygame visualization
        # cv2.waitKey(1)

    def imu_callback(self, imu):  # accelerometer is m/s^2 and gyroscope data is rad/sec
        imu_dict = {}
        imu_dict["timestamp"] = imu.timestamp
        imu_dict["accelerometer_x"] = imu.accelerometer.x
        imu_dict["accelerometer_y"] = imu.accelerometer.y
        imu_dict["accelerometer_z"] = imu.accelerometer.z
        imu_dict["gyroscope_x"] = imu.gyroscope.x
        imu_dict["gyroscope_y"] = imu.gyroscope.y
        imu_dict["gyroscope_z"] = imu.gyroscope.z
        # create a pandas dataframe
        self.imu_dataframe = self.imu_dataframe.append(imu_dict, ignore_index=True)
        # save the dataframe to a csv file
        self.imu_dataframe.to_csv(os.path.join(self.save_dir, "imu.csv"), index=False)

    def gnss_callback(self, gnss):
        gnss_dict = {}
        gnss_dict["timestamp"] = gnss.timestamp
        gnss_dict["latitude"] = gnss.latitude
        gnss_dict["longitude"] = gnss.longitude
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
        for index, detect in enumerate(radar_data):
            radar_dict = {"index": index, "altitude": detect.altitude, "azimuth": detect.azimuth, "depth": detect.depth,
                          "velocity": detect.velocity}
        #     self.radar_dataframe = self.radar_dataframe.append(radar_dict, ignore_index=True)
        #
        # self.radar_dataframe.to_csv(os.path.join(self.save_dir, "radar.csv"), index=False)
        return radar_dict

    def colision(self):
        collision_sensor = self.world.spawn_actor(self.blueprint_library.find('sensor.other.collision'),
                                                  carla.Transform(), attach_to=self.vehicle,
                                                  attachment_type=carla.AttachmentType.Rigid)
        return collision_sensor

    def colision_callback(self, event):
        print(f"COLLISION: {event}")
