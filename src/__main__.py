import time
import sys
import carla

from src.simulator_handler import SimulatorHandler
from utils.vehicle_command import VehicleCommand
from path_following_handler import PathFollowingHandler
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    simulator_handler = SimulatorHandler(town_name="Town10HD_Opt")
    simulator_handler.set_weather(weather=carla.WeatherParameters.ClearNoon)

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


    if path_following_handler.debug_mode:
        path_following_handler.start()
    else:
        ego_pid_controller = path_following_handler.pid_controller(ego_vehicle,
                                                                   path_following_handler.pid_values_lateral,
                                                                   path_following_handler.pid_values_longitudinal)
        path_following_handler.vehicle_and_controller_inputs(ego_vehicle, ego_pid_controller)
        path_following_handler.start()




