# Virtual car driving using hand gestures


## Environment used - VS code

<br />

## Requirements

ROS 1
Installation instructions - http://wiki.ros.org/noetic/Installation/Ubuntu

Turtlebot3 packages
Installation instructions - https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#gazebo-simulation

## Installation of Dependecies

### `Numpy`

```
pip install numpy
```

### `TensorFlow`

```
pip install tensorflow==2.12
```

### `OpenCV`

```
pip install opencv-python
```

### `Mediapipe`

```
pip install mediapipe
```


## Additionally Install this for running the training script


### `Scikit-learn`

```
pip install scikit-learn
```

## To Run the program

- Clone the repository to the scripts folder of turtlebot 3_bringup package in the src folder of the ROS1 catkin workspace.
- Go to terminal and type the below commands in the terminal

```
cd ~/catkin_ws && catkin_make
```
```
export TURTLEBOT3_MODEL=burger
```
```
source ~/catkin_ws/devel/setup.bash
```

```
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
```
- Open another terminal and type the below commands
```
cd ~/catkin_ws/src/turtlebot3/turtlebot3_bringup/scripts/
```
```
rosrun turtlebot3_bringup f.py
```

## To train the model 

- Go to the main folder and open model_training_script.ipynb
- If you want to modify the gestures go the model folder and change the text_classifier label file inside the keypoint_classifer folder accordingly
- Then change the number of classes in the model_training_script to the number of labels present in the text_classifier_label file
- ```Run all the cells of the model_training_script to train the model```
- Based on the confusion matrix, precision and recall score, improve the model's accuracy by adding more data or removing some data to prevent overfitting issues



