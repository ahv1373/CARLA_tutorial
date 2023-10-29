import threading
from typing import Tuple, Any, Dict, Union

import carla


def draw_waypoints(world_, waypoints_, road_id=None, life_time=50.0):
    for waypoint in waypoints_:

        if waypoint.road_id == road_id:
            world_.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                     color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                     persistent_lines=True)


def filter_waypoints(waypoints_, road_id):
    filtered_waypoints_ = []
    for waypoint in waypoints_:
        if waypoint.road_id == road_id:
            filtered_waypoints_.append(waypoint)
    return filtered_waypoints_


class TrajectoryToFollow:
    def __init__(self, trajectory_index: int) -> None:
        self.trajectory_index = trajectory_index

    def get_trajectory_data(self) -> Tuple[Any, list, list]:
        if self.trajectory_index == 0:
            carla_map = "Town10HD_Opt"
            road_id_list: list = [13, 12, 12, 11, 2, 10]
            filtered_point_index_list: list = [-4, -1, 11, 1, 0, 0]
        elif self.trajectory_index == 1:
            carla_map = "Town10HD_Opt"
            road_id_list: list = [1, 2000, 1, 2, 1000, 21, 20, 20, 1000, 15, 22, 6, 2000]
            filtered_point_index_list: list = [-300, 1, -4, -4, 25, -1, -1, 3, 10, 0, 3, -1, 1]
        else:
            raise NotImplementedError('A trajectory with index {} has not been implemented.'
                                      .format(self.trajectory_index))
        return carla_map, road_id_list, filtered_point_index_list

    def get_ego_vehicle_spwan_point(self) -> Union[Dict[str, int], Dict[str, int]]:
        if self.trajectory_index == 0:
            ego_spawn_point: Union = {'road_id': 13, 'filtered_points_index': 0}
        elif self.trajectory_index == 1:
            ego_spawn_point: Union = {'road_id': 1, 'filtered_points_index': 4}
        else:
            raise NotImplementedError('A trajectory with index {} has not been implemented.')
        return ego_spawn_point


class PIDControllerProperties:
    def __init__(self) -> None:
        self.k_p = None
        self.k_i = None
        self.k_d = None

    def set_pid_gains(self, k_p: float, k_i: float, k_d: float) -> None:
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

    def get_pid_gains(self) -> Tuple[float, float, float]:
        return self.k_p, self.k_i, self.k_d