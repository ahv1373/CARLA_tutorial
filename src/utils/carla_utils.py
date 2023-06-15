import sys
import threading
from typing import Tuple, Any, Dict, Union
sys.path.append(r"D:\CARLA_0.9.8_2\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")

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
            carla_map = "Town05"
            # road_id_list: list = [17,1000, 1,2000, 1000, 5,2000, 1000, 7,2000,1000, 17,2000]
            # filtered_point_index_list: list = [-4,3, 0,1, 9, -1, 1,3, -1,1,8, 0,1]
            road_id_list: list = [1,1,2000, 1000, 5,5,2000, 1000, 7,7,2000,1000, 17,17,2000,1000]
            filtered_point_index_list: list = [ 0,28,1, 5, -1,-29, 1,4, -1,-29,1,8, 0,28,1,5]
        elif self.trajectory_index == 1:
            carla_map = "Town05"
            road_id_list: list = [1, 1, 1, 2, 1000, 21, 20, 20, 1000, 15, 22, 6, 2000]
            filtered_point_index_list: list = [-300, 1, -4, -4, 25, -1, -1, 3, 10, 0, 3, -1, 1]
        else:
            raise NotImplementedError('A trajectory with index {} has not been implemented.'
                                      .format(self.trajectory_index))
        return carla_map, road_id_list, filtered_point_index_list

    def get_ego_vehicle_spwan_point(self) -> Union[Dict[str, int], Dict[str, int]]:
        if self.trajectory_index == 0:
            ego_spawn_point: Union = {'road_id': 1, 'filtered_points_index': 0}
        elif self.trajectory_index == 1:
            ego_spawn_point: Union = {'road_id': 1, 'filtered_points_index': 4}
        else:
            raise NotImplementedError('A trajectory with index {} has not been implemented.')
        return ego_spawn_point


class InfiniteLoopThread(threading.Thread):
    def __init__(self, name=None):
        threading.Thread.__init__(self, name=name)
        self._terminate = False

    def join(self, **kwargs):
        self._terminate = True
        super().join(**kwargs)

    def run(self) -> None:
        while not self._terminate:
            self.__step__()
        print("Thread [{}] terminated.".format(self.name))

    def __step__(self, *args, **kwargs):  # the only function needs to be implemented by inherited classes
        raise NotImplementedError
