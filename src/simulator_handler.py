import os

import carla
import cv2
import numpy as np
import pandas as pd


class SimulatorHandler:
    def __init__(self, town_name):
        self.spawn_point = None
        self.vehicle = None
        self.rgb_cam_sensor = None
        self.bp = None
        if not os.path.exists(os.path.join("data", "rgb_cam")):
            os.makedirs(os.path.join("data", "rgb_cam"))

        try:
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
        except Exception as error:
            raise Exception(f"Error while initializing the simulator: {error}")

        self.imu_dict = {"timestamp": [],
                         "accelerometer_x": [],
                         "accelerometer_y": [],
                         "accelerometer_z": [],
                         "gyroscope_x": [],
                         "gyroscope_y": [],
                         "gyroscope_z": []}
        self.gnss_dict = {"timestamp": [],
                          "latitude": [],
                          "longitude": [],
                          "altitude": []}

    def spawn_vehicle(self, spawn_index: int = 90):
        self.bp = self.blueprint_library.filter("model3")[0]  # choosing the car
        self.spawn_point = self.world.get_map().get_spawn_points()[spawn_index]
        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)
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
        # imu_sensor.set_attribute("sensor_tick", str(3.0))
        ego_imu = self.world.spawn_actor(imu_sensor, imu_transform, attach_to=self.vehicle,
                                         attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(ego_imu)
        return ego_imu

    def rgb_cam_callback(self, image):
        image.save_to_disk("data/rgb_cam/%06d.jpg" % image.frame)
        raw_image = np.array(image.raw_data)

        rgba_image = raw_image.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))  # because carla rgb cam is rgba
        rgb_image = rgba_image[:, :, :3]
        # cv2.imshow("rgb camera", rgb_image)  # FixMe: Replace with pygame visualization
        # cv2.waitKey(1)

    def imu_callback(self, imu):  # accelerometer is m/s^2 and gyroscope data is rad/sec
        self.imu_dict["timestamp"].append(imu.timestamp)
        self.imu_dict["accelerometer_x"].append(imu.accelerometer.x)
        self.imu_dict["accelerometer_y"].append(imu.accelerometer.y)
        self.imu_dict["accelerometer_z"].append(imu.accelerometer.z)
        self.imu_dict["gyroscope_x"].append(imu.gyroscope.x)
        self.imu_dict["gyroscope_y"].append(imu.gyroscope.y)
        self.imu_dict["gyroscope_z"].append(imu.gyroscope.z)
        # create a pandas dataframe
        imu_df = pd.DataFrame(self.imu_dict)
        # save the dataframe to a csv file
        imu_df.to_csv("data/imu.csv", index=False)

    def gnss_callback(self, gnss):
        self.gnss_dict["timestamp"].append(gnss.timestamp)
        self.gnss_dict["latitude"].append(gnss.latitude)
        self.gnss_dict["longitude"].append(gnss.longitude)
        self.gnss_dict["altitude"].append(gnss.altitude)
        gnss_df = pd.DataFrame(self.gnss_dict)
        gnss_df.to_csv("data/gnss.csv", index=False)
