import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass



import time

import carla

from src.simulator_handler import SimulatorHandler
from utils.vehicle_command import VehicleCommand

if __name__ == "__main__":
    simulator_handler = SimulatorHandler(town_name="Town05")
    simulator_handler.spawn_vehicle(spawn_index=34)
    simulator_handler.set_weather(weather=carla.WeatherParameters.ClearNoon)

    # potential weather choices are [ClearNoon, ClearSunset, CloudyNoon, CloudySunset,
    # WetNoon, WetSunset, MidRainyNoon, MidRainSunset, HardRainNoon, HardRainSunset,
    # SoftRainNoon, SoftRainSunset]

    # add sensors
    rgb_cam = simulator_handler.rgb_cam()
    gnss_sensor = simulator_handler.gnss()
    imu_sensor = simulator_handler.imu()
    lidar = simulator_handler.lidar()
    radar = simulator_handler.radar()
    collision = simulator_handler.collision()

    radarList = o3d.geometry.PointCloud()
    point_list = o3d.geometry.PointCloud()

    # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    lidar.listen(lambda data: simulator_handler.lidar_callback(data, point_list))
    radar.listen(lambda data: simulator_handler.radar_callback(data, radarList))
    collision.listen(lambda event: simulator_handler.collision_callback(event))
    VehicleCommand(throttle=1.0).send_control(simulator_handler.vehicle)

    # Open3D visualiser for LIDAR and RADAR
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='Carla Lidar',
        width=960,
        height=540,
        left=480,
        top=270)
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True
    simulator_handler.add_open3d_axis(vis)

    frame = 0
    while True:
        if frame == 2:
            vis.add_geometry(point_list)
            vis.add_geometry(radarList)
        vis.update_geometry(point_list)
        vis.update_geometry(radarList)

        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.005)
        frame += 1

        # Break if user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            break



