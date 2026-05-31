

import dobotArm
import numpy as np
import mediapipe as mp
import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)
H_matrix = np.load("HomographyMatrix.npy")
data = np.load("./camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

# Compute undistort maps once
ret, frame = cap.read()
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1.0])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]

def is_closer(last_position, current_position):
    if last_position is None or current_position is None:
        return None

    last_x, last_y = last_position
    last_position_distance = np.sqrt(last_x**2 + last_y**2)
    current_x, current_y = current_position
    current_position_distance = np.sqrt(current_x**2 + current_y**2)


    if current_position_distance < last_position_distance:
        return True
    else:
        return False


def move_robot_arm_to_safe_position(distance_vector,threshold=50):
    # Move the robot arm to a safe position (e.g., home position)
    x,y = distance_vector
    current_pose = dobotArm.get_pose(dobotArm.api)
    x_position = current_pose[0]
    y_position = current_pose[1]
    z_position = current_pose[2]

    nparray = np.array(distance_vector)
    nparray = nparray / np.linalg.norm(nparray) * threshold

    if is_closer((x,y), (x_position, y_position)) and np.sqrt(x**2 + y**2) < threshold:
      return False
      while np.sqrt(x**2 + y**2) < threshold:
              dobotArm.move_to_xyz(dobotArm.api, x_position, y_position, z_position) #stops
              time.sleep(1)  
              current_pose = dobotArm.get_pose(dobotArm.api)
              x_position = current_pose[0]
              y_position = current_pose[1]
              z_position = current_pose[2]
    return True
    '''if is_closer((x,y), (x_position, y_position)) and np.sqrt(x**2 + y**2) < warning_threshold:
      while np.sqrt(x**2 + y**2) < threshold:
              dobotArm.move_to_xyz(dobotArm.api, x_position-nparray[0], y_position-nparray[1], z_position)  
              time.sleep(1)  # Wait for the arm to move
              current_pose = dobotArm.get_pose(dobotArm.api)
              x_position = current_pose[0]
              y_position = current_pose[1]
              z_position = current_pose[2]'''
    '''else:
      while np.sqrt(x**2 + y**2) < threshold:
              dobotArm.move_to_xyz(dobotArm.api, x_position+nparray[0], y_position+nparray[1], z_position)  
              time.sleep(1)  # Wait for the arm to move
              current_pose = dobotArm.get_pose(dobotArm.api)
              x_position = current_pose[0]
              y_position = current_pose[1]
              z_position = current_pose[2]'''


def detect_human_hand_distance(camera_frame):
    """
    Detects a human hand in camera_frame and returns the XY-plane distance (mm)
    between the detected wrist and the robot arm's current position.
    Returns None if no hand is detected.
    """
    camera_frame = cv2.remap(camera_frame, map1, map2, cv2.INTER_LINEAR)
    h, w = camera_frame.shape[:2]

    # MediaPipe Tasks API requires an mp.Image in RGB format
    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    detection_result = detector.detect(mp_image)

    if not detection_result.hand_landmarks:
        return None

    # Use the wrist landmark (index 0) of the first detected hand
    wrist = detection_result.hand_landmarks[0][0]
    px = int(wrist.x * w)
    py = int(wrist.y * h)

    # Convert pixel coordinates to robot XY coordinates via homography
    hand_x, hand_y = pixel_to_robot(px, py, H_matrix)

    # Get the robot arm's current XY position
    pose = dobotArm.get_pose(dobotArm.api)
    arm_x, arm_y = pose[0], pose[1]

    distance = float((hand_x - arm_x) + (hand_y - arm_y))
    return distance