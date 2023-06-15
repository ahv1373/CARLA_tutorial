
import glob
import os
import sys
import random
import cv2
# import skimage.measure as measure

#in synchronous mode, sensor data must be added to a queue
import queue
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append(r"D:\CARLA_0.9.8_2\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")

sys.path.append(r"D:\CARLA_Code\integrate two session\src")
# from utils.carla_utils import draw_waypoints, filter_waypoints, TrajectoryToFollow, InfiniteLoopThread

# import time

import carla


sys.path.append(r"D:\CARLA_Code\integrate two session\src")
sys.path.append(r"D:\CARLA_Code\integrate two session\utils")

from simulator_handler import SimulatorHandler
# from path_following_handler import PathFollowingHandler
# from vehicle_command import VehicleCommand





if __name__ == '__main__':
    client = carla.Client("localhost", 2000)
    client.set_timeout(8.0)

    town_name="Town05"
    # spawn_index = 2

    try:
        print("Trying to communicate with the client...")
        world = client.get_world()
        if os.path.basename(world.get_map().name) != town_name:
            world: carla.World = client.load_world(town_name)

        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05 # must be less than 0.1, or else physics will be noisy
        #must use fixed delta seconds and synchronous mode for python api controlled sim, or else 
        #camera and sensor data may not match simulation properly and will be noisy 
        settings.synchronous_mode = True 
        world.apply_settings(settings)
        

        blueprint_library = world.get_blueprint_library()
        actor_list = []
        print("Successfully connected to CARLA client")
    except Exception as error:
        raise Exception(f"Error while initializing the simulator: {error}")

    simulator_handler = SimulatorHandler(client=client,actor_list=actor_list)
    
    weather = carla.WeatherParameters(
    cloudiness=20.0,
    precipitation=20.0,
    sun_altitude_angle=110.0)

   

    # vehicle_blueprint = blueprint_library.filter("model3")[0]  # choosing the car
    
    blueprints = blueprint_library.filter('*') ##################

    blueprint_library = world.get_blueprint_library()
    bp = random.choice(blueprint_library.filter('vehicle')) # lets choose a vehicle at random

    # lets choose a random spawn point
    transform = random.choice(world.get_map().get_spawn_points()) 

    #spawn a vehicle
    vehicle = world.spawn_actor(bp, transform) 
    actor_list.append(vehicle)

    # vehicle.set_autopilot(True)


    # vehicle = client.get_world().spawn_actor(vehicle_blueprint, 5)
   
    m= world.get_map()
    waypoint = m.get_waypoint(transform.location)

# #lets add more vehicles
#     for _ in range(0, 200):
#         transform = random.choice(m.get_spawn_points())

#     bp_vehicle = random.choice(blueprint_library.filter('vehicle'))

#     # This time we are using try_spawn_actor. If the spot is already
#     # occupied by another object, the function will return None.
#     other_vehicle = world.try_spawn_actor(bp_vehicle, transform)
#     if other_vehicle is not None:
#         #print(npc)
#         # other_vehicle.set_autopilot(True)
#         actor_list.append(other_vehicle)
    


    # # add wierd thing
    # blueprint_library = world.get_blueprint_library()
    # weirdobj_bp = blueprint_library.find('static.prop.fountain')
    # weirdobj_transform = random.choice(world.get_map().get_spawn_points())
    # weirdobj_transform = carla.Transform(carla.Location(x=230, y=195, z=40), carla.Rotation(yaw=180))
    # weird_obj = world.try_spawn_actor(weirdobj_bp, weirdobj_transform)
    # actor_list.append(weird_obj)



    


    rgb_cam = simulator_handler.rgb_cam(vehicle)

    # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))

    world.tick()
    

