#!/usr/bin/env python3

# Import statements
import csv
import copy
import argparse
import itertools
import cv2
import math
import numpy as np
import mediapipe as mp
from geometry_msgs.msg import Twist
import rospy


from collections import Counter
from collections import deque
from model import KeyPointClassifier
from model import PointHistoryClassifier


# Defining Global Variables

speed = 0 # Vehicle's initial speed
angular_velocity = 0 # Vehicle's initial angular velocity

# Detection hands and drawing the keypoints
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Lists for storing the data related to hands and movement
hand_detect = []
move_detect = []

# Rospy Node initialization
rospy.init_node('ROS_move', anonymous=True)

# Function for publishing velocity commands
def publish_velocities(velo, ang):
    vel_msg = Twist()
    rate = rospy.Rate(5)
    vel_publisher_ = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    vel_msg = Twist()
    vel_msg.linear.x = velo/20
    vel_msg.angular.z = -ang/5
    vel_publisher_.publish(vel_msg)
    rate.sleep()
    return


# Function for Argument parsing
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


# Function for calculating the angle between the detected two hands
def calculate_angle(landmark_list):
    global angular_velocity
    l_wrist = landmark_list[0]
    r_wrist = landmark_list[1]
    if r_wrist[1] >= l_wrist[1]:
        radians1 = math.atan2(r_wrist[1] - l_wrist[1], r_wrist[0] - l_wrist[0])
    else:
        radians1 = math.atan2(l_wrist[1] - r_wrist[1], l_wrist[0] - r_wrist[0])
    deg = math.degrees(radians1) % 360
    if deg > 90:
        deg = deg - 180
    if -15 <= deg <= 15:
        stri = 'Going straight'
        angular_velocity = 0
    elif deg < -15:
        stri = 'Turning left'
        angular_velocity = np.deg2rad(deg)
    else:
        stri = 'Turning right'
        angular_velocity = np.deg2rad(deg)
    return stri, angular_velocity


# Function to select a mode for the opencv feed
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


# Fucntion to calculate the bounding rectangle for hands
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


# Function to calculate the handlandmarks
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


# Function for pre-processing the list of hand landmarks
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


# Function for pre-processing the point history
def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


# Function for writing the logs while capturing images
def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


# Function for drawing the bounding rectangles
def draw_bounding_rect(use_brect, image, brect):
    if use_brect:

        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image


# Function for writing the text on the opencv live window
def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    global speed
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    move_detect.append(hand_sign_text)
    hand_detect.append(info_text)
    if len(move_detect) == 2:
        if move_detect[0] == move_detect[1]:
            if hand_detect[0] != hand_detect[1]:
                if hand_sign_text == 'Decrease Speed':
                    if speed > 0:
                        speed -= 0.25
                elif hand_sign_text == 'Increase Speed':
                    if speed < 100:
                        speed += 0.25
                elif hand_sign_text == 'Reverse':
                    if speed > -15:
                        speed -= 0.25
                elif hand_sign_text == 'Brake':
                    speed = 0
                    publish_velocities(0, 0)
            hand_detect.clear()
            move_detect.clear()
        else:
            del move_detect[0]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, 'Speed =' + str(np.round(speed/20)), (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return image


# Main Function
def main():

    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # OpenCV data feed
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Loading the model

    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels 
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]


    # Coordinates history 
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history 
    finger_gesture_history = deque(maxlen=history_length)
    mode = 0

    while True:

        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Reading live feed
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image_ang = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == "Not applicable": 
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id],point_history_classifier_labels[most_common_fg_id[0][0]],)
            landmark_ang = []
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(debug_image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

                x1 = tuple(np.multiply(np.array(
                    (hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                    [640, 480]).astype(int))
                
                # Turning left or right detection
                landmark_ang.append(x1)
                if len(landmark_ang) == 2:
                    direction, angle = calculate_angle(landmark_ang)
                    cv2.putText(debug_image, 'Angular Speed =' + str(np.round(-angular_velocity/5, 2)), (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(debug_image, str(direction), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        else:
            point_history.append([0, 0])

        # Display output
        cv2.imshow('Hand Gesture Recognition', debug_image)
        publish_velocities(speed, np.round(angular_velocity, 2))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
