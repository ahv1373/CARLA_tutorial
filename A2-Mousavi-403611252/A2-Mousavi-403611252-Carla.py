import carla
import random
import os
import numpy as np
import cv2
import time

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

blueprints = world.get_blueprint_library()
vehicle_bp = random.choice(blueprints.filter('vehicle.*'))
spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)


camera_bp = blueprints.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

output_dir = 'carla_dataset_a2'
os.makedirs(output_dir, exist_ok=True)

def save_image(image, weather_name, img_id):
    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img_array = np.reshape(img_array, (image.height, image.width, 4))[:, :, :3]
    filename = os.path.join(output_dir, f'{weather_name}_{img_id}.png')
    cv2.imwrite(filename, img_array)

weathers = {
    "day": carla.WeatherParameters.ClearNoon,
    "night": carla.WeatherParameters(
        cloudiness=0.0, 
        precipitation=0.5, 
        sun_altitude_angle=-90.0
    ) ,
    "rain": carla.WeatherParameters.HardRainSunset,
    "fog": carla.WeatherParameters(
        cloudiness=20.0, 
        precipitation=0.0, 
        sun_altitude_angle=20.0, 
        fog_density=50.0
    )
}

try:
    img_id = 0 
    for weather_name, weather in weathers.items():
        world.set_weather(weather)
        time.sleep(5)  
        
        captured_images = [0]  

        def process_image(image):
            """ Callback function to process images """
            if captured_images[0] < 50: 
                save_image(image, weather_name, img_id + captured_images[0])
                captured_images[0] += 1
                time.sleep(3)
            else:
                camera.stop()  
        
        camera.listen(process_image)

        while captured_images[0] < 50:
            time.sleep(0.1)

        img_id += 50  
        camera.stop()  

finally:
    camera.destroy()
    vehicle.destroy()
    