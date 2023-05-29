import time
import sys
import carla

from src.simulator_handler import SimulatorHandler
from utils.vehicle_command import VehicleCommand
import warnings

warnings.filterwarnings('ignore')

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
    lidar_sensor = simulator_handler.lidar()
    radar_sensor = simulator_handler.radar()
    colision_sensor = simulator_handler.colision()

    # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    lidar_sensor.listen(lambda lidar_sensor: simulator_handler.lidar_callback(lidar_sensor))
    radar_sensor.listen(lambda radar_sensor: simulator_handler.radar_callback(radar_sensor))
    colision_sensor.listen(lambda colision_sensor: simulator_handler.colision_callback(colision_sensor))

    VehicleCommand(throttle=1.0).send_control(simulator_handler.vehicle)
    time.sleep(20.0)





