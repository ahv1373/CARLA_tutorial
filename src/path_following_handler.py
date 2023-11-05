import math
import os
import time
from typing import Any, Union, Dict, List, Optional

import carla
import matplotlib
from carla import World
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src.utils.carla_utils import draw_waypoints, filter_waypoints, TrajectoryToFollow, PIDControllerProperties
from src.utils.controller import VehiclePIDController

matplotlib.interactive(True)


class PathFollowingHandler:
    def __init__(self, client: carla.Client, trajectory_index: int = 0, **kwargs) -> None:
        # desired speed is in m/s
        self.trajectory_to_follow_handler = TrajectoryToFollow(trajectory_index=trajectory_index)
        self.actor_list = []
        self.ego_vehicle = None
        self.ego_pid_controller = None
        carla_map, road_id_list, filtered_point_index_list = self.trajectory_to_follow_handler.get_trajectory_data()
        self.ego_spawn_point: Union[Dict[str, int], Dict[str, int]] = \
            self.trajectory_to_follow_handler.get_ego_vehicle_spwan_point()
        self.client = client
        self.trajectory_to_follow: Union[Dict[str, list], Dict[str, list]] = \
            {'road_id': road_id_list, 'filtered_points_index': filtered_point_index_list}

        self.world: World = self.client.get_world()
        if os.path.basename(self.world.get_map().name) != carla_map:
            self.world: World = client.load_world(carla_map)
        self.waypoints: list = self.client.get_world().get_map().generate_waypoints(distance=1.0)

        self.lateral_pid_props, self.longitudinal_pid_props = PIDControllerProperties(), PIDControllerProperties()
        self.lateral_pid_props.set_pid_gains(k_p=1.0, k_i=0.05, k_d=0.07)
        self.longitudinal_pid_props.set_pid_gains(k_p=1.0, k_i=0.05, k_d=0.07)

        self.pid_values_lateral: Union[Dict[str, float], Dict[str, float], Dict[str, float]] = \
            {'K_P': self.lateral_pid_props.k_p,
             'K_D': self.lateral_pid_props.k_d,
             'K_I': self.lateral_pid_props.k_i}
        self.pid_values_longitudinal: Union[Dict[str, float], Dict[str, float], Dict[str, float]] = \
            {'K_P': self.longitudinal_pid_props.k_p,
             'K_D': self.longitudinal_pid_props.k_d,
             'K_I': self.longitudinal_pid_props.k_i}

        # Receive keyword arguments
        self.vehicle_to_target_distance_threshold: float = kwargs.get('vehicle_to_target_distance_threshold', 2.5)
        self.desired_speed: float = kwargs.get('desired_speed', 20.0)

        self.reached_destination: bool = False
        self.previous_waypoint: Optional[carla.Waypoint] = None

        # initialize live plotting
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        self.axes_list = [ax[0], ax[1]]
        ax[0].set_title('Throttle commands')
        ax[1].set_title('Steering commands')
        self.throttle_commands, self.steering_commands, self.timestamp = [], [], []
        self.throttle_plot, = ax[0].plot([], [], color='blue')
        self.steering_plot, = ax[1].plot([], [], color='red')
        self.live_plot = FuncAnimation(fig, self.update_live_plots, init_func=self.plot_initializer)
        self.initial_simulation_time = time.time()

    def plot_initializer(self):
        for ax in self.axes_list:
            ax.set_xlim(0, 50)
            ax.set_ylim(-1, 1)
            ax.grid(True)
        return self.throttle_plot, self.steering_plot

    def update_live_plots(self, timestamp):
        self.throttle_plot.set_data(self.timestamp, self.throttle_commands)
        self.steering_plot.set_data(self.timestamp, self.steering_commands)

        # set x axis (timestamp) interactively
        self.axes_list[0].set_xlim(max(0, self.timestamp[-1] - 50),
                                   max(self.timestamp[-1], 50))
        self.axes_list[1].set_xlim(max(0, self.timestamp[-1] - 50),
                                   max(self.timestamp[-1], 50))
        return self.throttle_plot, self.steering_plot

    def __follow_target_waypoints__(self, vehicle: Any, target_waypoint, ego_pid_controller_) -> None:
        self.client.get_world().debug.draw_string(target_waypoint.transform.location, 'O',
                                                  draw_shadow=False,
                                                  color=carla.Color(r=255, g=0, b=0),
                                                  life_time=20,
                                                  persistent_lines=True)
        while True:
            vehicle_loc = vehicle.get_location()
            vehicle_to_target_distance = math.sqrt((target_waypoint.transform.location.x - vehicle_loc.x) ** 2
                                                   + (target_waypoint.transform.location.y - vehicle_loc.y) ** 2)

            if vehicle_to_target_distance < self.vehicle_to_target_distance_threshold:
                break
            else:
                control_signal = ego_pid_controller_.run_step(self.desired_speed, target_waypoint)
                vehicle.apply_control(control_signal)

                # append the control signals to the live plotting lists

                self.throttle_commands.append(control_signal.throttle)
                self.steering_commands.append(control_signal.steer)
                self.timestamp.append(time.time() - self.initial_simulation_time)
                print(f"throttle: {control_signal.throttle}, steer: {control_signal.steer},"
                      f"timestamp: {time.time() - self.initial_simulation_time}")
                plt.show()  # Block execution until the plot is closed
                plt.pause(1e-9)  # Give time for the plot to update

    def visualize_road_id(self, road_id: int, filtered_points_index: int, life_time: int = 5) -> None:
        # For debugging purposes.
        filtered_waypoints = filter_waypoints(self.waypoints, road_id=road_id)
        draw_waypoints(self.world, self.waypoints, road_id=road_id, life_time=life_time)
        target_waypoint = filtered_waypoints[filtered_points_index]
        self.client.get_world().debug.draw_string(target_waypoint.transform.location, 'O',
                                                  draw_shadow=False,
                                                  color=carla.Color(r=255, g=0, b=0),
                                                  life_time=life_time,
                                                  persistent_lines=True)

    def spawn_ego_vehicles(self, road_id: int, filtered_points_index: int) -> Any:
        print("spawning ego vehicle at road_id={} filtered_points_index={}".format(road_id,
                                                                                   filtered_points_index))
        vehicle_blueprint = \
            self.client.get_world().get_blueprint_library().filter("model3")[0]
        filtered_waypoints = filter_waypoints(self.waypoints, road_id=road_id)
        spawn_point = filtered_waypoints[filtered_points_index].transform
        spawn_point.location.z += 2
        vehicle = self.client.get_world().spawn_actor(vehicle_blueprint, spawn_point)
        return vehicle

    @staticmethod
    def set_pid_controller(vehicle, args_lateral: dict, args_longitudinal: dict) -> VehiclePIDController:
        ego_pid_controller_ = VehiclePIDController(vehicle, args_lateral=args_lateral,
                                                   args_longitudinal=args_longitudinal)
        return ego_pid_controller_

    def follow_trajectory(self, vehicle: Any, ego_pid_controller_: VehiclePIDController) -> None:
        for trajectory_point_index in range(len(self.trajectory_to_follow['road_id'])):
            current_road_id, current_filtered_point_index = \
                self.trajectory_to_follow['road_id'][trajectory_point_index], \
                    self.trajectory_to_follow['filtered_points_index'][trajectory_point_index]
            draw_waypoints(self.world, self.waypoints, road_id=current_road_id, life_time=30)
            print("Following point: {}/{}".format(trajectory_point_index,
                                                  len(self.trajectory_to_follow['road_id']) - 1))
            print('current_road_id: {}, current_filtered_point_index: {}'.format(current_road_id,
                                                                                 current_filtered_point_index))
            if current_road_id == 1000:  # 1000 means using waypoint.next
                target_waypoint = self.previous_waypoint.next(float(
                    current_filtered_point_index))[0]
            elif current_road_id == 2000:  # 2000 means using waypoint.next_until_lane_end
                target_waypoints: List[carla.Waypoint] = self.previous_waypoint.next_until_lane_end(float(
                    current_filtered_point_index))
                for target_waypoint in target_waypoints:
                    self.__follow_target_waypoints__(vehicle, target_waypoint, ego_pid_controller_)
                    self.previous_waypoint = target_waypoint
            else:
                filtered_waypoints = filter_waypoints(self.waypoints, road_id=current_road_id)
                target_waypoint = filtered_waypoints[current_filtered_point_index]

            self.__follow_target_waypoints__(vehicle, target_waypoint, ego_pid_controller_)
            self.previous_waypoint = target_waypoint

    def set_vehicle_and_controller_inputs(self, ego_vehicle_, ego_pid_controller_):
        self.ego_vehicle = ego_vehicle_
        self.ego_pid_controller = ego_pid_controller_
        self.actor_list.append(self.ego_vehicle)

    def exec(self):
        if not self.reached_destination:
            self.follow_trajectory(self.ego_vehicle, self.ego_pid_controller)
            self.reached_destination = True
            print("Destination has been reached.")
        else:
            print("Destination is already reached. Skipping the path following algorithm.")

    def terminate(self):
        print("Terminating trajectory following handler")
        for actor in self.actor_list:
            actor.destroy()
        print("All actors are destroyed. Path following handler is terminated.")


if __name__ == '__main__':
    client_ = carla.Client("localhost", 2000)
    client_.set_timeout(8.0)
    path_following_handler = PathFollowingHandler(client=client_)

    # to visualize a road id region:
    # import sys
    # path_following_handler.visualize_road_id(road_id=10, filtered_points_index=50, life_time=30)
    # sys.exit(1)

    ego_spawn_point = path_following_handler.ego_spawn_point
    ego_vehicle = \
        path_following_handler.spawn_ego_vehicles(road_id=ego_spawn_point["road_id"],
                                                  filtered_points_index=ego_spawn_point["filtered_points_index"])
    ego_pid_controller = path_following_handler.set_pid_controller(ego_vehicle,
                                                                   path_following_handler.pid_values_lateral,
                                                                   path_following_handler.pid_values_longitudinal)
    path_following_handler.set_vehicle_and_controller_inputs(ego_vehicle, ego_pid_controller)
    path_following_handler.exec()
    # terminate the path following handler as well as the actor list
    path_following_handler.terminate()
