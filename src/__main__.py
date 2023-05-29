# Pooriya Sanaie, Fatemeh Aghabarari , Seyed Javad Hosseyni
import time

import carla

from src.simulator_handler import SimulatorHandler
from utils.vehicle_command import VehicleCommand

if __name__ == "__main__":
    simulator_handler = SimulatorHandler(town_name="Town05")
    simulator_handler.spawn_vehicle(spawn_index=1)
    simulator_handler.set_weather(weather=carla.WeatherParameters.ClearNoon)

    # potential weather choices are [ClearNoon, ClearSunset, CloudyNoon, CloudySunset,
    # WetNoon, WetSunset, MidRainyNoon, MidRainSunset, HardRainNoon, HardRainSunset,
    # SoftRainNoon, SoftRainSunset]

    # add sensors
    rgb_cam = simulator_handler.rgb_cam()
    gnss_sensor = simulator_handler.gnss()
    imu_sensor = simulator_handler.imu()
    lidar_sen_1 = simulator_handler.lidar()
    radar_sensor = simulator_handler.radar()
    # colision_sensor = simulator_handler.colision()

    # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    lidar_sen_1.listen(lambda point_cloud: point_cloud.save_to_disk('data/lidar_pooriya/%.6d.ply' % point_cloud.frame))
    radar_data_list=[]
    radar_sensor.listen(lambda radar_data: [radar_data_list.append(data) for data in radar_data])

    radar_sensor.listen(lambda radar_data: simulator_handler.rad_callback(radar_data))
    # colision_data_list = []
    # colision_sensor.listen(lambda event: [colision_data_list.append(event)])
    # colision_sensor.listen(lambda event: simulator_handler.colision_callback(event))
    VehicleCommand(throttle=1.0).send_control(simulator_handler.vehicle)
    time.sleep(20)
    poo = 1
