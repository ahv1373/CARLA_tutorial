import open3d as o3d
import cv2
import time
import carla

from src.simulator_handler import SimulatorHandler
from utils1.vehicle_command import VehicleCommand
from src.path_following_handler import PathFollowingHandler

if __name__ == '__main__':
    simulator_handler = SimulatorHandler(town_name="Town10HD_Opt")
    client_ = carla.Client("localhost", 2000)
    client_.set_timeout(10.0)
    path_following_handler = PathFollowingHandler(client=client_, debug_mode=False)
    ego_spawn_point = path_following_handler.ego_spawn_point
    if path_following_handler.debug_mode:
        path_following_handler.start()
    else:
        ego_vehicle = \
            path_following_handler.spawn_ego_vehicles(road_id=ego_spawn_point["road_id"],
                                                      filtered_points_index=ego_spawn_point["filtered_points_index"])
        ego_pid_controller = path_following_handler.pid_controller(ego_vehicle,
                                                                   path_following_handler.pid_values_lateral,
                                                                   path_following_handler.pid_values_longitudinal)
        path_following_handler.vehicle_and_controller_inputs(ego_vehicle, ego_pid_controller)
        path_following_handler.start()


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
        width=1200,
        height=600,
        left=500,
        top=300)
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

        time.sleep(0.1)
        frame += 1


        # Break if user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            break
