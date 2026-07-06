import os
import queue
import threading
import sys
import glob
try:
    sys.path.append(glob.glob(
        r'F:\carla\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg'
    )[0])
except IndexError:
    raise RuntimeError("Couldn't find the CARLA egg.")
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
        os.makedirs(os.path.join(self.save_dir, "lidar"), exist_ok=True)
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
            for bp in self.blueprint_library.filter("sensor.camera.*"):
                print("bd:", bp.id)
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
        self.radar_dataframe = pd.DataFrame({})
        self.collision_dataframe = pd.DataFrame({})


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
        rgb_camera.set_attribute('sensor_tick', str(1 / 15))  # 30 FPS
        spawn_point_rgb = carla.Transform(carla.Location(x=2.5, y=0, z=0.9),
                                          carla.Rotation(pitch=-5, roll=0, yaw=0))

        self.rgb_cam_sensor = self.world.spawn_actor(rgb_camera, spawn_point_rgb, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_cam_sensor)
        return self.rgb_cam_sensor

    def instance_segmentation_cam(self):
        instance_segmentation_camera = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        instance_segmentation_camera.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        instance_segmentation_camera.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        instance_segmentation_camera.set_attribute("fov", "110")
        instance_segmentation_camera.set_attribute('sensor_tick', str(1 / 20))  # 30 FPS
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
    def lidar(self):
        lidar_bp = self.blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "100")
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("points_per_second", "500000")
        transform = carla.Transform(carla.Location(x=0, y=0, z=2.4))
        ego_lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=self.vehicle)
        self.actor_list.append(ego_lidar)
        return ego_lidar

    def lidar_callback(self, point_cloud):
        point_cloud.save_to_disk(os.path.join(self.save_dir, "lidar", "%06d.ply" % point_cloud.frame))
        
    def radar(self):
        radar_bp = self.blueprint_library.find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", "30")
        radar_bp.set_attribute("vertical_fov", "10")
        radar_bp.set_attribute("range", "100")
        transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        ego_radar = self.world.spawn_actor(radar_bp, transform, attach_to=self.vehicle)
        self.actor_list.append(ego_radar)
        return ego_radar

    def radar_callback(self, radar_data):
        for detection in radar_data:
            row = {"timestamp": radar_data.timestamp, "depth": detection.depth,
                "azimuth": detection.azimuth, "altitude": detection.altitude,
                "velocity": detection.velocity}
            self.radar_dataframe = pd.concat([self.radar_dataframe, pd.DataFrame([row])], ignore_index=True)
        self.radar_dataframe.to_csv(os.path.join(self.save_dir, "radar.csv"), index=False)
        
    def collision(self):
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        ego_collision = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(ego_collision)
        return ego_collision

    def collision_callback(self, event):
        row = {"timestamp": event.timestamp,
            "other_actor": event.other_actor.type_id,
            "impulse_x": event.normal_impulse.x,
            "impulse_y": event.normal_impulse.y,
            "impulse_z": event.normal_impulse.z}
        self.collision_dataframe = pd.concat([self.collision_dataframe, pd.DataFrame([row])], ignore_index=True)
        self.collision_dataframe.to_csv(os.path.join(self.save_dir, "collision.csv"), index=False)
        
    def _process_image_saving(self):
        while True:
            rgb_img = self.rgb_image_queue.get()
            instance_segmentation_img = self.instance_segmentation_image_queue.get()
            if rgb_img is None or instance_segmentation_img is None:  # Sentinel value to stop the thread
                break
            rgb_img.save_to_disk(
                os.path.join(self.save_dir, "rgb_cam", "%06d.jpg" % rgb_img.frame))
            instance_segmentation_img.save_to_disk(
                os.path.join(self.save_dir, "instance_segmentation_cam", "%06d.png" % instance_segmentation_img.frame))
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
        # 1. Stop all sensors FIRST so no more callbacks fire
        for actor in self.actor_list:
            if actor is not None and actor.is_alive and isinstance(actor, carla.Sensor):
                if actor.is_listening:
                    actor.stop()

        # 2. Wait for queued images to finish saving
        print(f"Waiting for {self.rgb_image_queue.qsize()} RGB frames and "
            f"{self.instance_segmentation_image_queue.qsize()} Seg frames to finish saving...")
        self.rgb_image_queue.join()
        self.instance_segmentation_image_queue.join()

        # 3. Send the sentinel so the saver thread exits cleanly, then join it
        self.rgb_image_queue.put(None)
        self.instance_segmentation_image_queue.put(None)
        self.save_thread.join(timeout=5)
        print("All frames saved successfully!")

        # 4. Now destroy the actors
        for actor in self.actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()

        # 5. Clean up the rest
        pygame.quit()
        self.world = None
        self.actor_list = []
        # ... (rest of the resets as before)
        print("Simulation terminated.")
