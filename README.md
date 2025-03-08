# CARLA_tutorial
This repository provides useful tutorials for the CARLA simulator

## Assignment #1
Add LIDAR and RADAR, and collision sensors to the vehicle and store the outputs in the local system.
Please only sent your codes via GitHub (Do not email me the updated codes). To do so, please first clone the repository
and create new branch. The branch name should be:

```
A1-FAMILYNAME-STUDENTNUMBER
```

Submit a pull request to the repository and make any changes you want in your own branch.
Commit your changes and make sure to push them as well.

The due date will be April 5, 2025.


[//]: # (## Assignment #2)

[//]: # (Task #1: Integrate what you have learned in session 1 and 2 by adding RGB camera, IMU, and LIDAR sensor)

[//]: # (to the vehicle and control it to follow the path by taking advantage of PID controller.)

[//]: # (Store the sensor's data on your system.)

[//]: # ()
[//]: # (Task #2: Tune the PID gains for both longitudinal and later control of the vehicle.)

[//]: # (What are the important criteria for appropriate tuning of control parameters?)

[//]: # ()
[//]: # (Task #3: Instead of simply selecting one specific waypoint and passing it to the controller,)

[//]: # (update the codes to automatically select all subsequent waypoints in each road_id of the carla HD map.)

[//]: # (To do so, utilize waypoint.next_until_lane_end which returns the list of the subsequent waypoints.)

[//]: # (Your objective should be creating a looped trajectory for the vehicle and ensure that your)

[//]: # (autonomous car is capable of following this loop infinitely.)

[//]: # ()
[//]: # (Please only sent your codes via GitHub &#40;Do not email me the updated codes&#41;. To do so, please first clone the repository)

[//]: # (and create new branch. The branch name should be:)

[//]: # ()
[//]: # (```)

[//]: # (A2-FAMILYNAME-STUDENTNUMBER)

[//]: # (```)

[//]: # (The deadline to deliver the assignment is 10-June-2023.)

## Assignment #2
Generate an image dataset from the CARLA simulator containing samples from different weather conditions including day,
night, rain, and fog. Train a deep convolutional neural network to classify the images based on the weather conditions.

Evaluate the trained model on the test set utilizing relevant criteria in the classification task
and form the confusion matrix.

The due date will be April 5, 2025.