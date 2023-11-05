import math
import os

import carla

from src.utils.carla_utils import TrajectoryToFollow, PIDControllerProperties, draw_waypoints, filter_waypoints
from src.utils.controller import VehiclePIDController


class PathFollowingControl:
    def __init__(self, client) -> None:

        self.trajectory_to_follow_handler = TrajectoryToFollow(trajectory_index=0)
        self.actor_list = []
        self.ego_vehicle = None
        self.ego_pid_controller = None
        carla_map, road_id_list, filtered_point_index_list = self.trajectory_to_follow_handler.get_trajectory_data()
        self.ego_vehicle_spawn_point = self.trajectory_to_follow_handler.get_ego_vehicle_spwan_point()
        self.client = client
        self.trajectory_to_follow = {"road_id": road_id_list, "filtered_point_index": filtered_point_index_list}

        self.world = self.client.load_world(carla_map)
        self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=1.0)

        self.lateral_pid_props, self.longitudinal_pid_props = PIDControllerProperties(), PIDControllerProperties()
        self.lateral_pid_props.set_pid_gains(k_p=1.0, k_i=0.05, k_d=0.07)
        self.longitudinal_pid_props.set_pid_gains(k_p=1.0, k_i=0.05, k_d=0.07)

        self.pid_values_lateral = \
            {'K_P': self.lateral_pid_props.k_p,
             'K_D': self.lateral_pid_props.k_d,
             'K_I': self.lateral_pid_props.k_i}

        self.pid_values_longitudinal = \
            {'K_P': self.longitudinal_pid_props.k_p,
             'K_D': self.longitudinal_pid_props.k_d,
             'K_I': self.longitudinal_pid_props.k_i}

        self.vehicle_to_target_distance_threshold = 2.5
        self.desired_speed = 20.0  # m/s

        self.reached_destination = False
        self.previous_waypoint = None

        # there are some additional lines for plotting

    def plot_initializer(self):
        pass

    def update_live_plots(self, timestamp):
        pass

    def __follow_target_waypoints__(self, vehicle, target_waypoint, ego_pid_controller_) -> None:
        # visualize red point
        self.client.get_world().debug.draw_string(target_waypoint.transform.location,
                                                  '0', color=carla.Color(r=255, g=0, b=0), life_time=50.0)

        while True:
            # we should know our ego vehicle's location
            ego_vehicle_location = vehicle.get_location()
            vehicle_to_target_distance = math.sqrt(
                (ego_vehicle_location.x - target_waypoint.transform.location.x) ** 2 +
                (ego_vehicle_location.y - target_waypoint.transform.location.y) ** 2)
            if vehicle_to_target_distance < self.vehicle_to_target_distance_threshold:
                break
            else:
                control_signal = ego_pid_controller_.run_step(self.desired_speed, target_waypoint)
                vehicle.apply_control(control_signal)

    def visualize_road_id(self, road_id: int, filtered_point_idx: int, life_time: float) -> None:
        pass

    def spawn_ego_vehicle(self, road_id: int, filtered_point_idx: int) -> carla.Actor:
        vehicle_blueprint = self.client.get_world().get_blueprint_library().find('model3')[0]
        #
        filtered_waypoint: carla.Waypoint = filter_waypoints(self.waypoints, road_id=road_id)[filtered_point_idx]
        spawn_point = filtered_waypoint.transform
        spawn_point.location.z += 2.0
        vehicle = self.client.get_world().spawn_actor(vehicle_blueprint, spawn_point)
        return vehicle

    @staticmethod
    def set_pid_controller(vehicle, args_longitudinal: dict, args_lateral: dict) -> VehiclePIDController:
        ego_pid_controller = VehiclePIDController(vehicle, args_longitudinal=args_longitudinal,
                                                  args_lateral=args_lateral)
        return ego_pid_controller

    def follow_trajectory(self, vehicle, ego_pid_controller):
        # for m in range(0, 6):
        #     pass
        for trajectory_point_index in range(len(self.trajectory_to_follow["road_id"])):
            current_road_id, current_filtered_point_index = \
                self.trajectory_to_follow["road_id"][trajectory_point_index], \
                    self.trajectory_to_follow["filtered_point_index"][trajectory_point_index]

            draw_waypoints(self.world, self.waypoints, road_id=current_road_id, life_time=50.0)  # visualization

            filtered_waypoints = filter_waypoints(self.waypoints, road_id=current_road_id)
            target_waypoint: carla.Waypoint = filtered_waypoints[current_filtered_point_index]

            self.__follow_target_waypoints__(vehicle, target_waypoint, ego_pid_controller)
            self.previous_waypoint = target_waypoint

    def set_vehicle_and_controller_inputs(self, ego_vehicle_, ego_pid_controller_):
        self.ego_vehicle = ego_vehicle_
        self.ego_pid_controller = ego_pid_controller_
        self.actor_list.append(self.ego_vehicle)

    def exec(self):
        if not self.reached_destination:
            self.follow_trajectory(self.ego_vehicle, self.ego_pid_controller)
            self.reached_destination = True
            print("The vehicle has reached the destination")
            self.terminate()
        else:
            print("The vehicle has already reached the destination")

    def terminate(self):
        print("Starting the thread termination")
        for actor in self.actor_list:
            actor.destroy()
        print("All actors have been destroyed")


if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    path_following_handler = PathFollowingControl(client)

    ego_spawn_point = path_following_handler.ego_vehicle_spawn_point
    ego_vehicle = path_following_handler.spawn_ego_vehicle(road_id=ego_spawn_point["road_id"],
                                                           filtered_point_idx=ego_spawn_point["filtered_points_index"])
    ego_pid_controller = path_following_handler.set_pid_controller(ego_vehicle,
                                                                   args_longitudinal=path_following_handler.pid_values_longitudinal,
                                                                   args_lateral=path_following_handler.pid_values_lateral)
    path_following_handler.set_vehicle_and_controller_inputs(ego_vehicle, ego_pid_controller)

    path_following_handler.exec()
