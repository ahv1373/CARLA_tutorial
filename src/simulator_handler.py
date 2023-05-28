import os
import sys
sys.path.append("E:\Education\Ph.D\Semester 2\Courses\Machine "
                "Learning\Carla_TA\CARLA_0.9.8\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")
import carla
import cv2
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt


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
            self.IM_WIDTH = 1220  # Ideally a config file should be defined for such parameters
            self.IM_HEIGHT = 710
            print("Successfully connected to CARLA client")
        except Exception as error:
            raise Exception(f"Error while initializing the simulator: {error}")

        self.imu_dataframe = pd.DataFrame({})
        self.gnss_dataframe = pd.DataFrame({})
        self.Lidar_dataframe = pd.DataFrame({})
        self.Radar_dataframe = pd.DataFrame({})
        self.Collision_dataframe = pd.DataFrame({})

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
    def LIDAR_sensor(self):
        lidar_sensor = self.blueprint_library.find("sensor.lidar.ray_cast")
        lidar_sensor.set_attribute("channels", "25")
        lidar_sensor.set_attribute("range", "8")
        lidar_sensor.set_attribute("points_per_second", "20000")
        lidar_sensor.set_attribute('sensor_tick', '0.0')
        spawn_point_Lidar = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0),
                                            carla.Rotation(pitch=0, roll=0, yaw=0))

        self.lidar_Sensor = self.world.spawn_actor(lidar_sensor, spawn_point_Lidar, attach_to=self.vehicle)
        self.actor_list.append(self.lidar_Sensor)
        return self.lidar_Sensor

    def RADAR_sensor(self):
        Radar_sensor = self.blueprint_library.find("sensor.other.radar")
        Radar_sensor.set_attribute("horizontal_fov", "35.0")
        Radar_sensor.set_attribute("points_per_second", "1000")
        Radar_sensor.set_attribute("range", "90.0")
        Radar_sensor.set_attribute('sensor_tick', '0.0')
        Radar_sensor.set_attribute('vertical_fov', '30.0')
        spawn_point_Radar = carla.Transform(carla.Location(x=3.5, y=0, z=0.7),
                                            carla.Rotation(pitch=-5, roll=0, yaw=0))

        self.Radar_Sensor = self.world.spawn_actor(Radar_sensor, spawn_point_Radar, attach_to=self.vehicle)
        self.actor_list.append(self.Radar_Sensor)
        return self.Radar_Sensor

    def COLLISION_sensor(self):
        Collision_sensor = self.blueprint_library.find("sensor.other.radar")
        spawn_point_Collision = carla.Transform(carla.Location(x=0, y=0, z=0),
                                                carla.Rotation(pitch=0, roll=0, yaw=0))

        self.Collision_Sensor = self.world.spawn_actor(Collision_sensor, spawn_point_Collision, attach_to=self.vehicle)
        self.actor_list.append(self.Collision_Sensor)
        return self.Collision_Sensor
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
    def LIDAR_callback(self, Lidar):
        Lidar_dict = {}
        Lidar_dict["timestamp"] = Lidar.timestamp
        Lidar_dict["transform"] = Lidar.transform
        Lidar_dict["horizontal_angle"] = Lidar.horizontal_angle
        Lidar_dict["channels"] = Lidar.channels
        Lidar_dict["raw_data"] = Lidar.raw_data
        self.Lidar_dataframe = self.Lidar_dataframe.append(Lidar_dict, ignore_index=True)
        self.Lidar_dataframe.to_csv(os.path.join(self.save_dir, "Lidar.csv"), index=False)

    def RADAR_callback(self, Radar):
        Radar_dict = {}
        Radar_dict["raw_data"] = Radar.raw_data

        self.Radar_dataframe = self.Radar_dataframe.append(Radar_dict, ignore_index=True)
        self.Radar_dataframe.to_csv(os.path.join(self.save_dir, "Radar.csv"), index=False)

    def COLLISION_callback(self, Collision):
        Collision_dict = {}
        Collision_dict["timestamp"] = Collision.timestamp
        Collision_dict["transform"] = Collision.transform
        Collision_dict["actor"] = Collision.actor
        Collision_dict["other_actor"] = Collision.other_actor
        Collision_dict["normal_impulse"] = Collision.normal_impulse
        self.Collision_dataframe = self.Collision_dataframe.append(Collision_dict, ignore_index=True)
        self.Collision_dataframe.to_csv(os.path.join(self.save_dir, "Collision.csv"), index=False)

    def clearing(self):
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for actor in self.world.get_actors().filter('*sensor*'):
            actor.destroy()