import math
import os
import sys
from abc import ABC


from typing import Any, Union, Dict, List

import carla


from carla import World

from src.utils.carla_utils import draw_waypoints, filter_waypoints, TrajectoryToFollow, InfiniteLoopThread

from src.utils.controller import VehiclePIDController



class PathFollowingHandler(InfiniteLoopThread, ABC):
    def __init__(self, client: carla.Client, debug_mode: bool = False,
                 trajectory_index: int = 0) -> None:
        super(PathFollowingHandler, self).__init__(name="CARLA trajectory following handler")
        self.trajectory_to_follow_handler = TrajectoryToFollow(trajectory_index=trajectory_index)
        self.ego_vehicle = None
        self.ego_pid_controller = None
        carla_map, road_id_list, filtered_point_index_list = self.trajectory_to_follow_handler.get_trajectory_data()
        self.ego_spawn_point: Union[Dict[str, int], Dict[str, int]] = \
            self.trajectory_to_follow_handler.get_ego_vehicle_spwan_point()
        self.client = client
        self.trajectory_to_follow: Union[Dict[str, list], Dict[str, list]] = \
            {'road_id': road_id_list, 'filtered_points_index': filtered_point_index_list}

        self.world: World = self.client.get_world()
        self.spectator = self.world.get_spectator()
        if os.path.basename(self.world.get_map().name) != carla_map:
            self.world: World = client.load_world(carla_map)
        self.waypoints: list = self.client.get_world().get_map().generate_waypoints(distance=1.0)
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.waypoints_to_visualize = {'road_id': [4], 'filtered_points_index': [0]}
        self.pid_values_lateral: Union[Dict[str, float], Dict[str, float], Dict[str, float]] = \
            {'K_P': 1,
             'K_D': 0.08,
             'K_I': 0.03}  # control steering
        self.pid_values_longitudinal: Union[Dict[str, float], Dict[str, float], Dict[str, float]] = \
            {'K_P': 1.4,
             'K_D': 0.09,
             'K_I': 0.07}  # control speed
        self.vehicle_to_target_distance_threshold: float = 2.5

        self.desired_speed: int = 20  # meter per second
        self.reached_destination: bool = False
        self.previous_waypoint: Union[carla.Waypoint, None] = None

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
            desired_speed = self.desired_speed

            if vehicle_to_target_distance < self.vehicle_to_target_distance_threshold:
                break
            else:
                control_signal = ego_pid_controller_.run_step(desired_speed, target_waypoint)
                vehicle.apply_control(control_signal)

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
    def pid_controller(vehicle, args_lateral: dict, args_longitudinal: dict) -> VehiclePIDController:
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

    def infinite_trajectory_tracking(self, vehicle, ego_pid_controller_):
        curr_waypoint_index = {'road_id': 13, 'filtered_points_index': 0} #get first way point
        curr_waypoint = filter_waypoints(self.waypoints,
                                         road_id=curr_waypoint_index['road_id'])[curr_waypoint_index['filtered_points_index']]
        print(curr_waypoint)
        while True:
            way_points = curr_waypoint.next_until_lane_end(2)
            draw_waypoints(self.world, self.waypoints, road_id=curr_waypoint.road_id , life_time=30)
            for target_waypoint in way_points:
                self.__follow_target_waypoints__(vehicle, target_waypoint, ego_pid_controller_)
            curr_waypoint = target_waypoint.next(3.5)[0]



    def vehicle_and_controller_inputs(self, ego_vehicle_, ego_pid_controller_):
        self.ego_vehicle = ego_vehicle_
        self.ego_pid_controller = ego_pid_controller_

    def __step__(self):
        if self.debug_mode:
            self.visualize_road_id(road_id=self.waypoints_to_visualize['road_id'][0],
                                   filtered_points_index=self.waypoints_to_visualize['filtered_points_index'][0])
            sys.exit(1)

        if not self.reached_destination:
            self.infinite_trajectory_tracking(self.ego_vehicle, self.ego_pid_controller)

        else:
            sys.exit(1)

    def terminate(self):
        print("Terminating trajectory following handler")
        self.join()


if __name__ == '__main__':
    client_ = carla.Client("localhost", 2000)
    client_.set_timeout(8.0)
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
