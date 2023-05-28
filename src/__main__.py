import time
import sys

sys.path.append("E:\Education\Ph.D\Semester 2\Courses\Machine "
                "Learning\Carla_TA\CARLA_0.9.8\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")

import carla

from src.simulator_handler import SimulatorHandler
from utils.vehicle_command import VehicleCommand

if __name__ == "__main__":
    simulator_handler = SimulatorHandler(town_name="Town04")
    simulator_handler.spawn_vehicle(spawn_index=13)
    simulator_handler.set_weather(weather=carla.WeatherParameters.ClearNoon)

    # potential weather choices are [ClearNoon, ClearSunset, CloudyNoon, CloudySunset,
    # WetNoon, WetSunset, MidRainyNoon, MidRainSunset, HardRainNoon, HardRainSunset,
    # SoftRainNoon, SoftRainSunset]

    # add sensors
    rgb_cam = simulator_handler.rgb_cam()
    gnss_sensor = simulator_handler.gnss()
    imu_sensor = simulator_handler.imu()
    lidar_sensor = simulator_handler.LIDAR_sensor()
    radar_sensor = simulator_handler.RADAR_sensor()
    collision_sensor = simulator_handler.COLLISION_sensor()

    # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    lidar_sensor.listen(lambda lidar: simulator_handler.LIDAR_callback(lidar))
    radar_sensor.listen(lambda radar: simulator_handler.RADAR_callback(radar))
    collision_sensor.listen(lambda collision: simulator_handler.COLLISION_callback(collision))
    VehicleCommand(throttle=1.0).send_control(simulator_handler.vehicle)
    time.sleep(20.0)
    simulator_handler.clearing()


