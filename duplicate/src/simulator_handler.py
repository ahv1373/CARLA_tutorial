# CARLA => Run (Simulation)
# Ego-Vehicle -> Sensors (Vision, IMU, GNSS)
# Data receive -> Store
# VehicleCommand (throttle, steer, brake) -> Apply Ego-Vehicle
import carla
import os
import pandas as pd


class SimulatorHandler:
    def __init__(self, town_name: str) -> None:
        self.rgb_camera_sensor = None
        self.vehicle = None
        self.spawn_point = None
        self.vehicle_blueprint = None

        self.save_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(os.path.join(self.save_dir, "rgb_cam")):
            os.makedirs(os.path.join(self.save_dir, "rgb_cam"))

        # Connect to carla client
        try:
            print("Trying to connect to carla simulator...")
            client = carla.Client("localhost", 2000)
            client.set_timeout(8.0)
            self.world = client.get_world()
            # town != town_name -> town name
            if os.path.basename(self.world.get_map().name) != town_name:
                self.world = client.load_world(town_name)

            self.blueprint_library = self.world.get_blueprint_library()
            self.map = self.world.get_map()
            self.actor_list = []
            self.vehicle_list = []

            # For rgb camera
            self.IMG_WIDTH, self.IMG_HEIGHT = 800, 600  # in pixels
            print("Successfully connected to carla client")

        except Exception as e:
            raise Exception(f"Error occurred while connecting to client: {e}")

        # For other two sensors
        self.imu_dataframe = pd.DataFrame({})
        self.gnss_dataframe = pd.DataFrame({})

    def spawn_vehicle(self, spawn_index: int = 90):
        self.vehicle_blueprint = self.blueprint_library.filter("model3")[0]
        spawn_points = self.map.get_spawn_points()
        self.spawn_point = spawn_points[spawn_index]
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, self.spawn_point)
        self.actor_list.append(self.vehicle)
        self.vehicle_list.append(self.vehicle)

    def set_weather(self, weather=carla.WeatherParameters.ClearNoon):
        self.world.set_weather(weather)

    # Add sensors
    def rgb_cam(self):
        rgb_camera = self.blueprint_library.find("sensor.camera.rgb")
        rgb_camera.set_attribute("image_size_x", f"{self.IMG_WIDTH}")
        rgb_camera.set_attribute("image_size_y", f"{self.IMG_HEIGHT}")
        rgb_camera.set_attribute("fov", "110")  # in deg
        rgb_camera.set_attribute('sensor_tick', '0.0')
        camera_location = carla.Location(x=2.5, y=0, z=0.9)  # relative to vehicle location
        camera_rotation = carla.Rotation(roll=0, pitch=-5, yaw=0)

        spawn_rgb_camera = carla.Transform(camera_location, camera_rotation)
        self.rgb_camera_sensor = self.world.spawn_actor(rgb_camera, spawn_rgb_camera,
                                                        attach_to=self.vehicle)

    def gnss(self):
        pass

    def imu(self):
        pass

    def rgb_cam_callback(self, image):
        pass

    def imu_callback(self, imu):
        pass

    def gnss_callback(self, gnss):
        pass
