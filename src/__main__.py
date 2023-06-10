import time

import carla
#A2-Pooriya Sanaie(401624033)-Fateme Aghabarari(401622503)-Seyed Javad Hosseini
from src.simulator_handler import SimulatorHandler
from utils.vehicle_command import VehicleCommand
from path_following_handler import PathFollowingHandler
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    simulator_handler = SimulatorHandler(town_name="Town10HD_Opt")

    simulator_handler.set_weather(weather=carla.WeatherParameters.ClearNoon)
    path_following_handler = PathFollowingHandler(client=simulator_handler.client)
    ego_spawn_point = path_following_handler.ego_spawn_point
    ego_vehicle = simulator_handler.spawn_ego_vehicles(road_id=ego_spawn_point["road_id"],
                                                       filtered_points_index=ego_spawn_point["filtered_points_index"])
    # add sensors
    rgb_cam = simulator_handler.rgb_cam()
    gnss_sensor = simulator_handler.gnss()
    imu_sensor = simulator_handler.imu()
    lidar_sen_1 = simulator_handler.lidar()
    radar_sensor = simulator_handler.radar()
    #  colision_sensor = simulator_handler.colision()

    # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    lidar_sen_1.listen(lambda point_cloud: point_cloud.save_to_disk('data/lidar_pooriya/%.6d.ply' % point_cloud.frame))
    radar_sensor.listen(lambda radar_data: simulator_handler.rad_callback(radar_data))
    # colision_sensor.listen(lambda colision_sensor: simulator_handler.colision_callback(colision_sensor))

    if path_following_handler.debug_mode:
        path_following_handler.start()
    else:
        # ego_vehicle = path_following_handler.spawn_ego_vehicles(road_id=ego_spawn_point["road_id"],
        # filtered_points_index=ego_spawn_point["filtered_points_index"])
        ego_pid_controller = path_following_handler.pid_controller(ego_vehicle,
                                                                   path_following_handler.pid_values_lateral,
                                                                   path_following_handler.pid_values_longitudinal)
        path_following_handler.vehicle_and_controller_inputs(ego_vehicle, ego_pid_controller)
        path_following_handler.start()

    # time.sleep(20.0)
    # simulator_handler.clearing()
