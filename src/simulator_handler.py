import os
import sys
import carla
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import glob

class SimulatorHandler:
    def __init__(self, town_name):
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
            client = carla.Client("localhost", 2000)
            client.set_timeout(8.0)
            self.world = client.get_world()
            if os.path.basename(self.world.get_map().name) != town_name:
                self.world: carla.World = client.load_world(town_name)

            self.blueprint_library = self.world.get_blueprint_library()
            self.actor_list = []
            self.vehicle_list = []
            self.IM_WIDTH = 800  # Ideally a config file should be defined for such parameters
            self.IM_HEIGHT = 600
            print("Successfully connected to CARLA client")
        except Exception as error:
            raise Exception(f"Error while initializing the simulator: {error}")

        self.imu_dataframe = pd.DataFrame({})
        self.gnss_dataframe = pd.DataFrame({})
        self.LIDAR_dataframe = pd.DataFrame({})
        self.radar_dataframe = pd.DataFrame({})
        self.collision_dataframe = pd.DataFrame({})

        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for actor in self.world.get_actors().filter('*sensor*'):
            actor.destroy()

    def spawn_vehicle(self, spawn_index: int = 90):
        self.vehicle_blueprint = self.blueprint_library.filter("model3")[0]  # choosing the car
        self.spawn_point = self.world.get_map().get_spawn_points()[spawn_index]
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, self.spawn_point)
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
    def LIDAR(self):
        LIDAR_sensor = self.blueprint_library.find("sensor.lidar.ray_cast")
        LIDAR_sensor.set_attribute('range', '100.0')
        LIDAR_sensor.set_attribute('upper_fov', '15.0')
        LIDAR_sensor.set_attribute('lower_fov', '-25.0')
        LIDAR_sensor.set_attribute('channels', '64.0')
        LIDAR_sensor.set_attribute('rotation_frequency', '20.0')
        LIDAR_sensor.set_attribute('points_per_second', '500000')

        LIDAR_transform = carla.Transform(carla.Location(z=2))
        ego_LIDAR = self.world.spawn_actor(LIDAR_sensor, LIDAR_transform, attach_to=self.vehicle,
                                          attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(ego_LIDAR)
        return ego_LIDAR

    def radar(self):
        radar_sensor = self.blueprint_library.find("sensor.other.radar")
        radar_sensor.set_attribute('horizontal_fov', str(30))
        radar_sensor.set_attribute('vertical_fov', str(30))
        radar_sensor.set_attribute('points_per_second', '10000')
        radar_transform = carla.Transform(carla.Location(z=2))
        ego_radar = self.world.spawn_actor(radar_sensor, radar_transform, attach_to=self.vehicle,
                                          attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(ego_radar)
        return ego_radar

    def collision(self):
        collision_sensor = self.blueprint_library.find('sensor.other.collision')
        collision_location = carla.Location(0, 0, 0)
        collision_rotation = carla.Rotation(0, 0, 0)
        collision_transform = carla.Transform(collision_location, collision_rotation)
        ego_collision =self.world.spawn_actor(collision_sensor, collision_transform, attach_to=self.vehicle,
                                          attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(ego_collision)
        return ego_collision
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
    def collision_callback(self, collision):
        collision_dict = {}
        collision_dict["Collision Time"] = collision.timestamp
        collision_dict["collision_actor"] = collision.other_actor  # The actor collided with
        collision_dict["collision_impulse"] = collision.normal_impulse  # The impulse of the hit
        self.collision_dataframe = self.collision_dataframe.append(collision_dict, ignore_index=True)
        self.collision_dataframe.to_csv(os.path.join(self.save_dir, "collision.csv"), index=False)

    def LIDAR_callback(self, LIDAR):
        # LIDAR_dict = {}
        points = np.frombuffer(LIDAR.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(LIDAR), 3))
        # LIDAR_dict["timestamp"] = LIDAR.timestamp
        # LIDAR_dict["horizontal_angle"] = LIDAR.horizontal_angle
        # LIDAR_dict["raw_data"] = LIDAR.raw_data
        # LIDAR_dict["location"] = LIDAR.transform.location
        # LIDAR_dict["rotation"] = LIDAR.transform.rotation
        df = pd.DataFrame(points, columns=["X", "Y", "Z"])
        self.LIDAR_dataframe = self.LIDAR_dataframe.append(df, ignore_index=True)
        self.LIDAR_dataframe.to_csv(os.path.join(self.save_dir, "LIDAR.csv"), index=False)

    def radar_callback(self,radar):
        points = np.frombuffer(radar.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar), 4))
        df = pd.DataFrame(points, columns=["altitude","azimuth","depth","velocity"])
        self.radar_dataframe = self.radar_dataframe.append(df, ignore_index=True)
        self.radar_dataframe.to_csv(os.path.join(self.save_dir, "radar.csv"), index=False)
    def gnss_callback(self, gnss):
        gnss_dict = {}
        gnss_dict["timestamp"] = gnss.timestamp
        gnss_dict["latitude"] = gnss.latitude
        gnss_dict["longitude"] = gnss.longitude
        gnss_dict["altitude"] = gnss.altitude
        self.gnss_dataframe = self.gnss_dataframe.append(gnss_dict, ignore_index=True)
        self.gnss_dataframe.to_csv(os.path.join(self.save_dir, "gnss.csv"), index=False)
