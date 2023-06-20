import time

import carla

from src.simulator_handler import SimulatorHandler
from utils.vehicle_command import VehicleCommand

if __name__ == "__main__":
    simulator_handler = SimulatorHandler(town_name="Town03")
    simulator_handler.spawn_vehicle(spawn_index=30)
    simulator_handler.set_weather(weather=carla.WeatherParameters.ClearNoon)

    # potential weather choices are [ClearNoon, ClearSunset, CloudyNoon, CloudySunset,
    # WetNoon, WetSunset, MidRainyNoon, MidRainSunset, HardRainNoon, HardRainSunset,
    # SoftRainNoon, SoftRainSunset]

    # add sensors
    LIDAR_sensor = simulator_handler.LIDAR()
    radar_sensor = simulator_handler.radar()
    collision_sensor = simulator_handler.collision()
    rgb_cam = simulator_handler.rgb_cam()
    gnss_sensor = simulator_handler.gnss()
    imu_sensor = simulator_handler.imu()

    # listen to sensor data
    LIDAR_sensor.listen(lambda LIDAR: simulator_handler.LIDAR_callback(LIDAR))
    radar_sensor.listen(lambda radar: simulator_handler.radar_callback(radar))
    collision_sensor.listen(lambda collision: simulator_handler.collision_callback(collision))
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    VehicleCommand(throttle=1.0).send_control(simulator_handler.vehicle)
    
   time.sleep(1000.0)



