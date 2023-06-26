import glob
import os
import sys
import time
import carla

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


sys.path.append(r"D:\CARLA_0.9.8_2\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")
sys.path.append(r"D:\CARLA_Code\integrate two session\src")
sys.path.append(r"D:\CARLA_Code\integrate two session\src")
sys.path.append(r"D:\CARLA_Code\integrate two session\utils")

from simulator_handler import SimulatorHandler
from path_following_handler import PathFollowingHandler
from vehicle_command import VehicleCommand

if __name__ == '__main__':
    client = carla.Client("localhost", 2000)
    client.set_timeout(8.0)

    town_name = "Town05"


    try:
        print("Trying to communicate with the client...")
        world = client.get_world()
        if os.path.basename(world.get_map().name) != town_name:
            world: carla.World = client.load_world(town_name)

        blueprint_library = world.get_blueprint_library()
        actor_list = []
        print("Successfully connected to CARLA client")
    except Exception as error:
        raise Exception(f"Error while initializing the simulator: {error}")

    simulator_handler = SimulatorHandler(client=client, actor_list=actor_list)

    weather = [carla.WeatherParameters(cloudiness=20.0, sun_altitude_angle=90.0, fog_density=0.0),  # day
               carla.WeatherParameters(cloudiness=20.0, sun_altitude_angle=-90.0, fog_density=0.0),  # night
               carla.WeatherParameters(cloudiness=20.0, sun_altitude_angle=90.0, fog_density=60.0),  # fog
               carla.WeatherParameters(cloudiness=85.0, sun_altitude_angle=90.0, fog_density=0.0)]  # cloud

    world.set_weather(weather[3])


    path_following_handler = PathFollowingHandler(client=client, debug_mode=False)

    vehicle_blueprint = blueprint_library.filter("model3")[0]

    ego_spawn_point = path_following_handler.ego_spawn_point

if __name__ == "__main__":
    simulator_handler = SimulatorHandler(town_name="Town04")
    simulator_handler.spawn_vehicle(spawn_index=13)
    simulator_handler.set_weather(weather=carla.WeatherParameters.ClearNoon)
    filtered_waypoints = filter_waypoints(path_following_handler.waypoints, road_id=ego_spawn_point["road_id"])
    spawn_point = filtered_waypoints[ego_spawn_point["filtered_points_index"]].transform
    spawn_point.location.z += 2
    vehicle = client.get_world().spawn_actor(vehicle_blueprint, spawn_point)
    actor_list.append(vehicle)

    rgb_cam = simulator_handler.rgb_cam(vehicle)

    # add sensors
    rgb_cam = simulator_handler.rgb_cam()
    gnss_sensor = simulator_handler.gnss()
    imu_sensor = simulator_handler.imu()

    # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    VehicleCommand(throttle=1.0).send_control(simulator_handler.vehicle)
    time.sleep(20.0)

    if path_following_handler.debug_mode:
        path_following_handler.start()
    else:
        ego_pid_controller = path_following_handler.pid_controller(vehicle,
                                                                   path_following_handler.pid_values_lateral,
                                                                   path_following_handler.pid_values_longitudinal)
        path_following_handler.vehicle_and_controller_inputs(vehicle, ego_pid_controller)
        path_following_handler.start()


