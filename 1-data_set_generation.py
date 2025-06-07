import carla
import random
import time
import os
import numpy as np
import cv2
from datetime import datetime
from collections import defaultdict
import queue

# Initial settings
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(os.path.join(DATA_DIR, "rgb"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "semantics"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "analysis"), exist_ok=True)

# Weather conditions
weather_presets = {
    
    'Clear_Day': carla.WeatherParameters.ClearNoon,
    'Clear_Night': carla.WeatherParameters.ClearNight,
    'Heavy_Rain': carla.WeatherParameters.HardRainNoon,
    'Foggy': carla.WeatherParameters(fog_density=80.0)
        
}

# Class mapping for semantic segmentation
class_mapping =                                                                                                                                  {0: "Pedestrian", 1: "Traffic Light", 2: "Car", 3: "Bus"}

def save_image(image, path):
    """Save image data to file (RGB or semantic class IDs)"""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]

    if 'semantics' in path:
        # Save only class IDs from red channel
        class_ids = array[:, :, 2]
        np.save(path.replace('.png', '.npy'), class_ids)
    else:
        # Save RGB image
        cv2.imwrite(path, array)

def create_dense_traffic_scene(world, traffic_manager, blueprint_library, zone_center, radius=40):
    """Create a dense scene with cars, buses, pedestrians and traffic lights"""
    vehicles_list = []
    walkers_list = []
    lights_list = []

    vehicle_types = {
        # Cars
        "vehicle.tesla.model3": 4,
        "vehicle.audi.a2": 3,
        "vehicle.dodge.charger_police": 2,
        "vehicle.toyota.prius": 3,
        # Buses
        "vehicle.volkswagen.t2": 1,
        "vehicle.mercedes.sprinter": 2,
        "vehicle.carlamotors.carlacola": 1
    }

    # Use pre-defined spawn points from the map
    all_spawn_points = world.get_map().get_spawn_points()
    random.shuffle(all_spawn_points)
    used_spawn_points = set()

    # Spawn vehicles
    for vehicle_type, weight in vehicle_types.items():
        is_bus = any(bus_id in vehicle_type for bus_id in ["t2", "sprinter", "carlacola"])
        spawn_count = weight * (2 if is_bus else 3)
        for _ in range(spawn_count):
            try:
                # Find the spawn point closest to zone_center
                spawn_point = min(
                    (sp for sp in all_spawn_points if sp not in used_spawn_points),
                    key=lambda sp: sp.location.distance(zone_center.location)
                )
                used_spawn_points.add(spawn_point)
                bp = blueprint_library.find(vehicle_type)
                if bp.has_attribute('color'):
                    color = random.choice(bp.get_attribute('color').recommended_values)
                    bp.set_attribute('color', color)
                vehicle = world.try_spawn_actor(bp, spawn_point)
                if vehicle is not None:
                    vehicles_list.append(vehicle)
                    vehicle.set_autopilot(True, traffic_manager.get_port())
                    if is_bus:
                        traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(4.0, 6.0))
                        traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-10, 0))
                    else:
                        traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(1.0, 3.0))
                        traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-20, 20))
                    traffic_manager.update_vehicle_lights(vehicle, True)
            except Exception as e:
                print(f"Vehicle spawn error: {e}")
                continue

    # Create pedestrian groups
    pedestrian_groups = 4
    pedestrians_per_group = 6
    for _ in range(pedestrian_groups):
        group_center = carla.Location(
            x=zone_center.location.x + random.uniform(-radius/2, radius/2),
            y=zone_center.location.y + random.uniform(-radius/2, radius/2),
            z=zone_center.location.z
        )
        for _ in range(pedestrians_per_group):
            try:
                offset = random.uniform(1, 4)
                angle = random.uniform(0, 2 * np.pi)
                x = group_center.x + offset * np.cos(angle)
                y = group_center.y + offset * np.sin(angle)
                spawn_point = carla.Transform(
                    carla.Location(x=x, y=y, z=group_center.z + 1.0)
                )
                bp = random.choice(blueprint_library.filter("walker.pedestrian.*"))
                walker = world.try_spawn_actor(bp, spawn_point)
                if walker is not None:
                    walkers_list.append(walker)
                    walker_control = carla.WalkerControl()
                    walker_control.speed = random.uniform(0.6, 2.0)
                    direction = carla.Vector3D(
                        x=group_center.x - x,
                        y=group_center.y - y,
                        z=0
                    )
                    direction.make_unit_vector()
                    walker_control.direction = direction
                    walker.apply_control(walker_control)
            except Exception as e:
                print(f"Walker spawn error: {e}")
                continue

    # Configure traffic lights
    try:
        traffic_lights = world.get_actors().filter('traffic.traffic_light*')
        for traffic_light in traffic_lights:
            if traffic_light.get_location().distance(zone_center.location) < radius:
                states = [carla.TrafficLightState.Red,
                         carla.TrafficLightState.Yellow,
                         carla.TrafficLightState.Green]
                traffic_light.set_state(random.choice(states))
                traffic_light.set_green_time(random.uniform(4.0, 8.0))
                traffic_light.set_red_time(random.uniform(2.0, 6.0))
                traffic_light.set_yellow_time(random.uniform(1.0, 3.0))
                lights_list.append(traffic_light)
    except Exception as e:
        print(f"Traffic light error: {e}")

    return vehicles_list, walkers_list, lights_list

def analyze_dataset(data_dir):
    """Analyze dataset and generate reports"""
    print("\n=== Dataset Analysis ===")
    
    # Count samples
    rgb_files = os.listdir(os.path.join(data_dir, "rgb"))
    semantic_files = [f for f in os.listdir(os.path.join(data_dir, "semantics")) 
                     if f.endswith('.npy')]
    
    print(f"Total Samples: {len(rgb_files)}\n")
    
    # Count samples per weather
    weather_counts = defaultdict(int)
    for f in rgb_files:
        weather = f.split('_')[0]
        weather_counts[weather] += 1
        
    print("Samples per weather condition:")
    for weather, count in weather_counts.items():
        print(f"{weather}: {count} samples")
    
    print("\nTotal Annotated Instances:")
    # Analyze semantic segmentation data
    class_pixels = defaultdict(int)
    for sem_file in semantic_files:
        sem_data = np.load(os.path.join(data_dir, "semantics", sem_file))
        unique, counts = np.unique(sem_data, return_counts=True)
        for cls_id, count in zip(unique, counts):
            if cls_id in class_mapping:
                class_pixels[cls_id] += count
    
    for cls_id, pixels in class_pixels.items():
        print(f"{class_mapping[cls_id]} ({cls_id}): {pixels} pixels")

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Initialize variables
    ego_vehicle = None
    camera_rgb = None
    camera_seg = None
    vehicles_list = []
    walkers_list = []
    lights_list = []

    try:
        # World settings
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Create ego vehicle
        blueprint_library = world.get_blueprint_library()
        ego_bp = blueprint_library.find("vehicle.tesla.model3")

        # Try to spawn ego vehicle
        max_tries = 20
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for attempt in range(max_tries):
            try:
                spawn_point = spawn_points[attempt % len(spawn_points)]
                ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
                print(f"Successfully spawned ego vehicle on attempt {attempt+1}")
                break
            except RuntimeError as e:
                print(f"Failed attempt {attempt+1}/{max_tries} to spawn ego vehicle: {e}")
                if attempt == max_tries - 1:
                    raise RuntimeError("Could not spawn ego vehicle")

        # Set up traffic manager
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(1.5)
        traffic_manager.global_percentage_speed_difference(10.0)
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        # Set up cameras
        camera_rgb = world.spawn_actor(
            blueprint_library.find("sensor.camera.rgb"),
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=ego_vehicle
        )
        camera_seg = world.spawn_actor(
            blueprint_library.find("sensor.camera.semantic_segmentation"),
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=ego_vehicle
        )

        # Set up image capture
        global rgb_data, seg_data
        rgb_data = None
        seg_data = None

        def rgb_callback(image):
            global rgb_data
            rgb_data = image

        def seg_callback(image):
            global seg_data
            seg_data = image

        camera_rgb.listen(rgb_callback)
        camera_seg.listen(seg_callback)

        # Data collection loop
        frame_count = {w: 0 for w in weather_presets.keys()}
        all_spawn_points = world.get_map().get_spawn_points()

        for weather_name, weather in weather_presets.items():
            world.set_weather(weather)
            print(f"\nWeather set to: {weather_name}")
            while frame_count[weather_name] < 100: 
                # Clean up previous actors
                for vehicle in vehicles_list:
                    if vehicle and vehicle.is_alive:
                        vehicle.destroy()
                for walker in walkers_list:
                    if walker and walker.is_alive:
                        walker.destroy()
                vehicles_list.clear()
                walkers_list.clear()

                # Select a valid spawn point
                valid_spawn_points = [
                    sp for sp in all_spawn_points
                    if sp.location.z > 0 
                ]
                if not valid_spawn_points:
                    print("No valid spawn points available!")
                    break

                zone_center = random.choice(valid_spawn_points)

                # Move ego vehicle to the selected spawn point
                ego_vehicle.set_transform(zone_center)

                # Create new scene
                new_vehicles, new_walkers, new_lights = create_dense_traffic_scene(
                    world, traffic_manager, blueprint_library, zone_center
                )
                vehicles_list.extend(new_vehicles)
                walkers_list.extend(new_walkers)
                lights_list.extend(new_lights)

                # Let the scene stabilize
                for _ in range(30):
                    world.tick()

                # Capture and save images
                if rgb_data and seg_data:
                    filename = f"{weather_name}_{frame_count[weather_name]:05d}"
                    save_image(rgb_data, os.path.join(DATA_DIR, "rgb", f"rgb_{filename}.png"))
                    save_image(seg_data, os.path.join(DATA_DIR, "semantics", f"seg_{filename}.png"))
                    frame_count[weather_name] += 1
                    print(f"Saved: {filename} | Count: {frame_count[weather_name]}")

                # Delay between captures
                time.sleep(3.0)

        print("Data collection completed")
        analyze_dataset(DATA_DIR)

    finally:
        print("Cleaning up actors...")
        if camera_rgb:
            camera_rgb.stop()
        if camera_seg:
            camera_seg.stop()
        if ego_vehicle:
            ego_vehicle.set_autopilot(False)
            ego_vehicle.destroy()
        for vehicle in vehicles_list:
            if vehicle and vehicle.is_alive:
                vehicle.destroy()
        for walker in walkers_list:
            if walker and walker.is_alive:
                walker.destroy()
        world.apply_settings(original_settings)

if __name__ == "__main__":
    main()