import os
import queue
import threading

import carla
import cv2
import numpy as np
import pandas as pd
import pygame


class SimulatorHandler:
    def __init__(self, town_name: str):
        self.spawn_point = None
        self.vehicle = None
        self.rgb_cam_sensor = None
        self.vehicle_blueprint = None
        self.image_saving_index = 0

        # create data save directories (if they don't exist)
        self.save_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "rgb_cam"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "instance_segmentation_cam"), exist_ok=True)

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
            self.IM_WIDTH = 1280  # Ideally a config file should be defined for such parameters
            self.IM_HEIGHT = 720

            # For visualization purposes using the pygame library
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.IM_WIDTH, self.IM_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            # Attributes are: Hardware surface and double buffer which mean that the display is
            # rendered in the GPU and the display is double buffered

            print("Successfully connected to CARLA client")
        except Exception as error:
            raise Exception(f"Error while initializing the simulator: {error}")

        self.imu_dataframe = pd.DataFrame({})
        self.gnss_dataframe = pd.DataFrame({})

        self.rgb_image_queue = queue.Queue()
        self.instance_segmentation_image_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._process_image_saving, daemon=True)
        self.save_thread.start()

    def spawn_vehicle(self, spawn_index: int = 90):
        self.vehicle_blueprint = self.blueprint_library.filter("Lincoln")[0]  # choosing the car
        self.spawn_point = self.world.get_map().get_spawn_points()[spawn_index]
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, self.spawn_point)

        # Visualizing the spawn point of the ego vehicle on the CARLA Client
        self.world.debug.draw_string(self.spawn_point.location, 'O',
                                     color=carla.Color(r=255, g=0, b=0),
                                     life_time=20)

        self.actor_list.append(self.vehicle)

    def set_weather(self, weather=carla.WeatherParameters.ClearNoon):
        self.world.set_weather(weather)

    def rgb_cam(self):
        rgb_camera = self.blueprint_library.find("sensor.camera.rgb")
        rgb_camera.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        rgb_camera.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        rgb_camera.set_attribute("fov", "110")
        rgb_camera.set_attribute('sensor_tick', str(1 / 30))  # 30 FPS
        spawn_point_rgb = carla.Transform(carla.Location(x=2.5, y=0, z=0.9),
                                          carla.Rotation(pitch=-5, roll=0, yaw=0))

        self.rgb_cam_sensor = self.world.spawn_actor(rgb_camera, spawn_point_rgb, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_cam_sensor)
        return self.rgb_cam_sensor

    def instance_segmentation_cam(self):
        instance_segmentation_camera = self.blueprint_library.find("sensor.camera.instance_segmentation")
        instance_segmentation_camera.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        instance_segmentation_camera.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        instance_segmentation_camera.set_attribute("fov", "110")
        instance_segmentation_camera.set_attribute('sensor_tick', str(1 / 30))  # 30 FPS
        spawn_point_instance_segmentation = carla.Transform(carla.Location(x=2.5, y=0, z=0.9),
                                                            carla.Rotation(pitch=-5, roll=0, yaw=0))

        self.rgb_cam_sensor = self.world.spawn_actor(instance_segmentation_camera,
                                                     spawn_point_instance_segmentation, attach_to=self.vehicle)
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

    def _process_image_saving(self):
        while True:
            # Pulls the next image from the queue and saves it
            rgb_img = self.rgb_image_queue.get()
            instance_segmentation_img = self.instance_segmentation_image_queue.get()
            if rgb_img is None or instance_segmentation_img is None: # Sentinel value to stop the thread
                break
            rgb_img.save_to_disk("data/rgb_cam/%06d.jpg" % rgb_img.frame)
            instance_segmentation_img.save_to_disk("data/instance_segmentation_cam/%06d.png" % instance_segmentation_img.frame)
            # mark the task as done for both queues
            self.rgb_image_queue.task_done()
            self.instance_segmentation_image_queue.task_done()

    def rgb_cam_callback(self, image):
        # Save the image to disk
        # image.save_to_disk("data/rgb_cam/%06d.jpg" % image.frame)
        self.rgb_image_queue.put(image)
        # Visualize the image using pygame
        # Convert raw data to numpy array and reshape to (H, W, 4)
        img_bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_bgra = np.reshape(img_bgra, (image.height, image.width, 4))

        # Fast numpy slicing: Drops the Alpha channel and reverses BGR to RGB in one step
        img_rgb = img_bgra[:, :, 2::-1]

        # Render to Pygame
        image_surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
        self.display.blit(image_surface, (0, 0))
        pygame.display.flip()

    def instance_segmentation_callback(self, image):
        self.instance_segmentation_image_queue.put(image)

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
        self.imu_dataframe = pd.concat([self.imu_dataframe, pd.DataFrame([imu_dict])], ignore_index=True)
        # save the dataframe to a csv file
        self.imu_dataframe.to_csv(os.path.join(self.save_dir, "imu.csv"), index=False)

    def gnss_callback(self, gnss):
        gnss_dict = {}
        gnss_dict["timestamp"] = gnss.timestamp
        gnss_dict["latitude"] = gnss.latitude
        gnss_dict["longitude"] = gnss.longitude
        gnss_dict["altitude"] = gnss.altitude
        # append the dictionary to the dataframe
        self.gnss_dataframe = pd.concat([self.gnss_dataframe, pd.DataFrame([gnss_dict])], ignore_index=True)
        self.gnss_dataframe.to_csv(os.path.join(self.save_dir, "gnss.csv"), index=False)

    def terminate(self):
        print(
            f"Waiting for {self.rgb_image_queue.qsize()} RGB frames and {self.instance_segmentation_image_queue.qsize()} Seg frames to finish saving...")

        # 1. Block the main thread until the queues are completely empty
        # Do this BEFORE destroying anything!
        self.rgb_image_queue.join()
        self.instance_segmentation_image_queue.join()
        print("All frames saved successfully!")

        # 2. Now it is safe to destroy the actors
        for actor in self.actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()

        # 3. Clean up the rest
        pygame.quit()
        self.world = None
        self.actor_list = []
        self.vehicle_list = []
        self.vehicle = None
        self.rgb_cam_sensor = None
        self.vehicle_blueprint = None
        self.spawn_point = None
        self.imu_dataframe = pd.DataFrame({})
        self.gnss_dataframe = pd.DataFrame({})

        print("Simulation terminated.")
