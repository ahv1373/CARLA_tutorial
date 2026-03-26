# Overview
This assignment involves integrating LiDAR, RADAR, and collision sensors into a CARLA vehicle and saving their output locally. A static obstacle is placed in the car’s path to test sensor readings and collision detection.

# Components Added
LiDAR Sensor: Captures point cloud data.

RADAR Sensor: Detects objects within a certain field of view.

Collision Sensor: Logs collision events during vehicle motion.

# Procedure
Vehicle Initialization: A random vehicle is spawned at a random spawn point in the map.

Obstacle Placement: A static object (street barrier) is placed 5 meters in front of the vehicle.

# Sensor Attachment:

LiDAR with 32 channels and 50m range.

RADAR with horizontal FOV 35° and vertical FOV 20°.

Collision sensor listens for any crash during movement.

Vehicle Motion: The vehicle moves forward for 10 seconds and then stops.

# Output
Sensor outputs are saved in the A1_sensor_data/ folder:

lidar_output.txt: LiDAR point cloud data

radar_output.txt: RADAR object detection readings

collision_log.txt: Any collision event logs