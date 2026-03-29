import carla
import random
import time
import os

output_folder = "A1_sensor_data"
os.makedirs(output_folder, exist_ok=True)

client = carla.Client('localhost', 2000)
client.set_timeout(5.0) 
world = client.get_world()
blueprint_library = world.get_blueprint_library() 

vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
print("Spawned Vehicle ID:", vehicle.id)
print("Vehicle spawned:", vehicle.type_id)

obstacle_bp = blueprint_library.find('static.prop.streetbarrier')  # Example prop
obstacle_transform = carla.Transform(
    spawn_point.location + carla.Location(x=5),
    spawn_point.rotation
)
obstacle = world.spawn_actor(obstacle_bp, obstacle_transform)
print("Obstacle spawned in front of the vehicle.")



lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('channels', '32')
lidar_bp.set_attribute('range', '50')
lidar_sensor = world.spawn_actor(lidar_bp, carla.Transform(), attach_to=vehicle)

def process_lidar_data(point_cloud):
    lidar_file = os.path.join(output_folder, "lidar_output.txt") 
    with open(lidar_file, "a") as file:
        file.write(str(point_cloud) + "\n")

lidar_sensor.listen(lambda data: process_lidar_data(data))

radar_bp = blueprint_library.find('sensor.other.radar')
radar_bp.set_attribute('horizontal_fov', '35')
radar_bp.set_attribute('vertical_fov', '20')

radar_sensor = world.spawn_actor(radar_bp, carla.Transform(), attach_to=vehicle)

def process_radar_data(data):
    radar_file = os.path.join(output_folder, "radar_output.txt")
    with open(radar_file, "a") as file:
        file.write(str(data) + "\n")

radar_sensor.listen(lambda data: process_radar_data(data))

collision_bp = blueprint_library.find('sensor.other.collision')

collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

def on_collision(event):
    collision_file = os.path.join(output_folder, "collision_log.txt")
    with open(collision_file, "a") as file:
        file.write("Collision detected: " + str(event) + "\n")

collision_sensor.listen(lambda event: on_collision(event))


vehicle.apply_control(carla.VehicleControl(throttle=0.5))
time.sleep(10)
vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

