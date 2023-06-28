import os
import sys
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm

sys.path.append(r"D:\CARLA_0.9.8_2\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")
import carla
from carla import World


# sys.path.append(r"D:\CARLA_Code\integrate two session\src")


class SimulatorHandler:
    def __init__(self, client: carla.Client, actor_list):
        self.raw_data = None
        self.COOL = None
        self.COOL_RANGE = None
        self.viridis = None
        self.spawn_point = None
        self.rgb_cam_sensor = None
        self.vehicle_blueprint = None


@ @-21

, 36 + 32, 79 @ @


def __init__(self, town_name):
    if not os.path.exists(os.path.join(self.save_dir, "rgb_cam")):
        os.makedirs(os.path.join(self.save_dir, "rgb_cam"))
 self.imu_dataframe = pd.DataFrame({})
        self.gnss_dataframe = pd.DataFrame({})
      self.client = client
        self.world: World = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.IM_WIDTH = 800  # Ideally a config file should be defined for such parameters
        self.IM_HEIGHT = 600

        self.actor_list = actor_list

    # Add LIDAR and RADAR, and collision sensors

    def lidar(self, vehicle):
        lidar_sensor = self.blueprint_library.find("sensor.lidar.ray_cast")
        # lidar_sensor.set_attribute("sensor_tick", str(0.0))
        # lidar_sensor.set_attribute('range', '100.0')
        # lidar_sensor.set_attribute('upper_fov', '15.0')
        # lidar_sensor.set_attribute('lower_fov', '-25.0')
        # lidar_sensor.set_attribute('channels', '64.0')
        # lidar_sensor.set_attribute('rotation_frequency', '20.0')
        # lidar_sensor.set_attribute('points_per_second', '500000')
        with open("src/data/lidar.csv", 'a') as csvfile:
            ww = np.array(["X", "Y", "Z"])
            ww = np.reshape(ww, (1, 3))
            np.savetxt(csvfile, ww, delimiter=',', fmt=['%s', '%s', '%s'], comments='')
        self.viridis = np.array(cm.get_cmap('plasma').colors)
        lidar_location = carla.Location(0, 0, 0)
        lidar_rotation = carla.Rotation(0, 0, 0)
        lidar_transform = carla.Transform(lidar_location, lidar_rotation)
        lidar_sen = self.world.spawn_actor(lidar_sensor, lidar_transform, attach_to=vehicle,
                                           attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(lidar_sen)
        return lidar_sen

    def radar(self, vehicle):
        radar_sensor = self.blueprint_library.find("sensor.other.radar")
        radar_sensor.set_attribute('horizontal_fov', '30.0')
        radar_sensor.set_attribute('vertical_fov', '30.0')
        radar_sensor.set_attribute('points_per_second', '10000')
        with open('src/data/radar.csv', 'a') as csvfile:
            ww = np.array(["altitude", "azimuth", "depth", "velocity"])
            ww = np.reshape(ww, (1, 4))
            np.savetxt(csvfile, ww, delimiter=',', fmt=['%s', '%s', '%s', '%s'], comments='')
        self.viridis = np.array(cm.get_cmap('plasma').colors)
        # VID_RANGE = np.linspace(0.0, 1.0, self.viridis.shape[0])
        self.COOL_RANGE = np.linspace(0.0, 1.0, self.viridis.shape[0])
        self.COOL = np.array(cm.get_cmap('winter')(self.COOL_RANGE))
        self.COOL = self.COOL[:, :3]
        radar_location = carla.Location(0, 0, 0)
        radar_rotation = carla.Rotation(0, 0, 0)
        radar_transform = carla.Transform(radar_location, radar_rotation)
        radar_sen = self.world.spawn_actor(radar_sensor, radar_transform, attach_to=vehicle,
                                           attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(radar_sen)
        return radar_sen

    def collision(self, vehicle):
        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        with open('src/data/collision.csv', 'a') as csvfile:
            ww = np.array(["time stamp", "situation"])
            ww = np.reshape(ww, (1, 2))
            np.savetxt(csvfile, ww, delimiter=',', fmt=['%s', '%s'], comments='')
        collision_location = carla.Location(0, 0, 0)
        collision_rotation = carla.Rotation(0, 0, 0)
        collision_transform = carla.Transform(collision_location, collision_rotation)
        collision_sen = self.world.spawn_actor(collision_sensor, collision_transform, attach_to=vehicle,
                                               attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(collision_sen)
        return collision_sen

    def rgb_cam(self, vehicle):
        rgb_camera = self.blueprint_library.find("sensor.camera.rgb")
        rgb_camera.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        rgb_camera.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
@@ -59,40 +113,43 @@ def rgb_cam(self):
        spawn_point_rgb = carla.Transform(carla.Location(x=2.5, y=0, z=0.9),
                                          carla.Rotation(pitch=-5, roll=0, yaw=0))
 self.rgb_cam_sensor = self.world.spawn_actor(rgb_camera, spawn_point_rgb, attach_to=vehicle)
        self.actor_list.append(self.rgb_cam_sensor)
        return self.rgb_cam_sensor


def gnss(self, vehicle):
    gnss_sensor = self.blueprint_library.find("sensor.other.gnss")
    gnss_sensor.set_attribute("sensor_tick", str(0.0))
    gnss_location = carla.Location(0, 0, 0)
    gnss_rotation = carla.Rotation(0, 0, 0)
    gnss_transform = carla.Transform(gnss_location, gnss_rotation)
    ego_gnss = self.world.spawn_actor(gnss_sensor, gnss_transform, attach_to=vehicle,
                                      attachment_type=carla.AttachmentType.Rigid)
    self.actor_list.append(ego_gnss)
    return ego_gnss

   def imu(self, vehicle):
        imu_sensor = self.blueprint_library.find("sensor.other.imu")
        imu_location = carla.Location(0, 0, 0)
        imu_rotation = carla.Rotation(0, 0, 0)
        imu_transform = carla.Transform(imu_location, imu_rotation)
        ego_imu = self.world.spawn_actor(imu_sensor, imu_transform, attach_to=vehicle,
                                         attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(ego_imu)
        return ego_imu

        # callbacks


def rgb_cam_callback(self, image):
    image.save_to_disk("src/data/rgb_cam/%06d.jpg" % image.frame)
    # raw_image = np.array(image.raw_data)

    # rgba_image = raw_image.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))  # because carla rgb cam is rgba
    # rgb_image = rgba_image[:, :, :3]
    # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    # plt.imshow(rgb_image)
    # plt.show()

    # cv2.imshow("rgb camera", rgb_image)  # FixMe: Replace with pygame visualization
    # cv2.waitKey(1)


@ @-118

, 3 + 175, 25 @ @


def gnss_callback(self, gnss):
    gnss_dict["altitude"] = gnss.altitude
    self.gnss_dataframe = self.gnss_dataframe.append(gnss_dict, ignore_index=True)
    self.gnss_dataframe.to_csv(os.path.join(self.save_dir, "gnss.csv"), index=False)


def lidar_callback(self, point_cloud):
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    # print("#########", len(point_cloud.raw_data), "#########")
    data = np.reshape(data, (int(data.shape[0] / 3), 3))

    with open('src/data/lidar.csv', 'a') as csvfile:
        np.savetxt(csvfile, data, delimiter=',', fmt=['%f', '%f', '%f'], comments='')


def collision_callback(self, event):
    ar = np.array([str(event.timestamp), "collision detected!"])
    ar = np.reshape(ar, (1, 2))
    with open('src/data/collision.csv', 'a') as csvfile:
        np.savetxt(csvfile, ar, delimiter=',', fmt=['%s', '%s'], comments='')


def radar_callback(self, data):
    mydata = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
    mydata = np.reshape(mydata, (int(mydata.shape[0] / 4), 4))
    with open('src/data/radar.csv', 'a') as csvfile:
        np.savetxt(csvfile, mydata, delimiter=',', fmt=['%f', '%f', '%f', '%f'], comments='')
        import sys
        import threading
        from typing import Tuple, Any, Dict, Union
        sys.path.append(r"D:\CARLA_0.9.8_2\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")

        import carla

        @ @-27

        , 12 + 29, 14 @ @

        def __init__(self, trajectory_index: int) -> None:
            def get_trajectory_data(self) -> Tuple[Any, list, list]:
                if self.trajectory_index == 0:
                    carla_map = "Town05"
                    # road_id_list: list = [17,1000, 1,2000, 1000, 5,2000, 1000, 7,2000,1000, 17,2000]
                    # filtered_point_index_list: list = [-4,3, 0,1, 9, -1, 1,3, -1,1,8, 0,1]
                    road_id_list: list = [1, 1, 2000, 1000, 5, 5, 2000, 1000, 7, 7, 2000, 1000, 17, 17, 2000, 1000]
                    filtered_point_index_list: list = [0, 28, 1, 5, -1, -29, 1, 4, -1, -29, 1, 8, 0, 28, 1, 5]
                elif self.trajectory_index == 1:
                    carla_map = "Town05"
                    road_id_list: list = [1, 1, 1, 2, 1000, 21, 20, 20, 1000, 15, 22, 6, 2000]
                    filtered_point_index_list: list = [-300, 1, -4, -4, 25, -1, -1, 3, 10, 0, 3, -1, 1]
                else:
                    raise NotImplementedError('A trajectory with index {} has not been implemented.'
                                              @ @ -41, 7 + 45, 7 @ @

                    def get_trajectory_data(self) -> Tuple[Any, list, list]:

                def get_ego_vehicle_spwan_point(self) -> Union[Dict[str, int], Dict[str, int]]:
                    if self.trajectory_index == 0:
                        ego_spawn_point: Union = {'road_id': 1, 'filtered_points_index': 0}
                        elif self.trajectory_index == 1:
                        ego_spawn_point: Union = {'road_id': 1, 'filtered_points_index': 4}
                    else:

                        #
                        # This work is licensed under the terms of the MIT license.
                        # For a copy, see <https://opensource.org/licenses/MIT>.

                        import sys
                        """ This module contains PID controllers to perform lateral and longitudinal control. """
                        sys.path.append(
                            r"D:\CARLA_0.9.8_2\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")

                        from collections import deque
                        import math
                        @ @-170

                        , 7 + 171, 7 @ @

                        def _pid_control(self, target_speed, current_speed):

                            return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

                        def change_parameters(self, K_P, K_I, K_D, dt):
                            def change_parameters(self, K_P, K_I, K_D, dt=0.03):
                                """Changes the PID parameters"""
                                self._k_p = K_P
                                self._k_i = K_I

                        @ @-261

                        , 7 + 262, 7 @ @

                        def _pid_control(self, waypoint, vehicle_transform):

                            return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

                        def change_parameters(self, K_P, K_I, K_D, dt):
                            def change_parameters(self, K_P, K_I, K_D, dt=0.03):
                                """Changes the PID parameters"""
                                self._k_p = K_P
                                self._k_i = K_I
