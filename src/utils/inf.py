import itertools
carla_map = "Town10HD_Opt"
road = [13, 12, 12, 11, 11, 1, 1, 1, 1, 1, 1, 4, 4, 13]
filtered = [-4, -1, 1, -1, 5, -1, -17, -45, -153, -201, 3, 0, -4, 0]
infinite_road = itertools.cycle(road)
infinite_filtered = itertools.cycle(filtered)
print(road+road, infinite_filtered)
# road_id_list: list = infinite_road
# filtered_point_index_list: list = infinite_filtered