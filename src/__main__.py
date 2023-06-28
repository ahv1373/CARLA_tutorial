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

sys.path.append(r"D:\CARLA_0.9.8_2\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")

sys.path.append(r"D:\CARLA_Code\integrate two session\src")
from utils.carla_utils import draw_waypoints, filter_waypoints, TrajectoryToFollow, InfiniteLoopThread

import time

import carla
sys.path.append(r"D:\CARLA_Code\integrate two session\src")
sys.path.append(r"D:\CARLA_Code\integrate two session\utils")

from simulator_handler import SimulatorHandler
from path_following_handler import PathFollowingHandler
from vehicle_command import VehicleCommand

if __name__ == '__main__':
    client = carla.Client("localhost", 2000)
    client.set_timeout(8.0)

    town_name = "Town05"
    # spawn_index = 2

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

    # weather = carla.WeatherParameters(cloudiness=100.0,sun_altitude_angle=165.0,fog_density=0.0)
    # world.set_weather(weather)

    # carla.WeatherParameters(cloudiness=20.0,
    #         sun_altitude_angle=100.0,fog_density=60.0)

    path_following_handler = PathFollowingHandler(client=client, debug_mode=False)

    vehicle_blueprint = blueprint_library.filter("model3")[0]  # choosing the car
    # spawn_point = world.get_map().get_spawn_points()[spawn_index]
    # vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)

    ego_spawn_point = path_following_handler.ego_spawn_point
    filtered_waypoints = filter_waypoints(path_following_handler.waypoints, road_id=ego_spawn_point["road_id"])
    spawn_point = filtered_waypoints[ego_spawn_point["filtered_points_index"]].transform
    spawn_point.location.z += 2
    vehicle = client.get_world().spawn_actor(vehicle_blueprint, spawn_point)
    actor_list.append(vehicle)
    rgb_cam = simulator_handler.rgb_cam(vehicle)
  # gnss_sensor = simulator_handler.gnss(vehicle)
    # imu_sensor = simulator_handler.imu(vehicle)
    # lidar = simulator_handler.lidar(vehicle)
    # radar = simulator_handler.radar(vehicle)
    # collision = simulator_handler.collision(vehicle)
    # listen to sensor data
    rgb_cam.listen(lambda image: simulator_handler.rgb_cam_callback(image))
    # imu_sensor.listen(lambda imu: simulator_handler.imu_callback(imu))
    # gnss_sensor.listen(lambda gnss: simulator_handler.gnss_callback(gnss))
    # lidar.listen(lambda data: simulator_handler.lidar_callback(data))
    # radar.listen(lambda data: simulator_handler.radar_callback(data))
    # collision.listen(lambda event: simulator_handler.collision_callback(event))

    if path_following_handler.debug_mode:
        path_following_handler.start()
    else:
        ego_pid_controller = path_following_handler.pid_controller(vehicle,
                                                                   path_following_handler.pid_values_lateral,
                                                                   path_following_handler.pid_values_longitudinal)

        path_following_handler.vehicle_and_controller_inputs(vehicle, ego_pid_controller)
        path_following_handler.start()

        from tensorflow.keras.callbacks import ModelCheckpoint
        from mock import Mock
        import matplotlib.pyplot as plt
        from sklearn.metrics import classification_report, confusion_matrix

        num_of_test_samples = 0
        for root_dir, cur_dir, files in os.walk(r"D:\CARLA_Code\trainSet\test"):
            num_of_test_samples += len(files)
        print('num_of_test_samples count:', num_of_test_samples)


        class TrainHyperParameters:
            def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3), number_of_classes: int = 4,
                         learning_rate: float = 0.001, batch_size: int = 32, number_of_epochs: int = 5) -> None:
                self.hyperparameters = Mock()
                self.hyperparameters.input_shape = input_shape
                self.hyperparameters.number_of_classes = number_of_classes


        @ @-60

        , 12 + 66, 23 @ @


        def model_builder(self):
            self.model = keras.models.Sequential([
                keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.hyperparameters.input_shape),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.2),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.2),
                keras.layers.Conv2D(128, (3, 3), activation='relu'),
                keras.layers.Conv2D(128, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.2),
                keras.layers.Conv2D(512, (3, 3), activation='relu'),
                keras.layers.Conv2D(512, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.2),
                keras.layers.Flatten(),
                keras.layers.Dense(1024, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(self.hyperparameters.number_of_classes, activation='softmax')
            ])


        @ @-106

        , 6 + 123, 15 @ @


        def train(self, train_generator, test_generator):
            # plot loss and accuracy on train and validation set
            self.plot_history(history)

            Y_pred = self.model.predict_generator(test_generator,
                                                  num_of_test_samples // self.hyperparameters.batch_size + 1)
            y_pred = np.argmax(Y_pred, axis=1)
            print('Confusion Matrix')
            print(confusion_matrix(test_generator.classes, y_pred))
            print('Classification Report')
            target_names = ['fog ', 'day', 'cloud', 'night']
            print(classification_report(test_generator.classes, y_pred, target_names=target_names))


        def plot_history(self, history):
            matplotlib.use('Agg')
            plt.figure(figsize=(10, 5))


        @ @-128

        , 6 + 154, 7 @ @


        def exec(self):

            if __name__ == '__main__':
                data_dir_ = r"D:\CARLA_Code\trainSet"
                train_custom_cnn = TrainCustomCNN(data_dir_)
                train_custom_cnn.exec()

                self.model = None
                self.model_path = model_path
                self.model_input_size = model_input_size
                self.class_labels = ['day', 'night']
                self.class_labels = ['fog ', 'day', 'night', 'cloud']

            def load(self):
                start_time = time.time()


        @ @-40

        , 8 + 40, 8 @ @


        def exec(self, frame: np.ndarray) -> str:


        if __name__ == "__main__":
            img_dir = r"D:\CARLA_Code\trainSet\finalTest"
            model_path_ = r"D:\CARLA_Code\output\checkpoints\best_model.h5"
            adverse_weather_classifier = AdverseWeatherClassifier(model_path_)
            adverse_weather_classifier.load()
            for root, dirs, files in os.walk(img_dir):
