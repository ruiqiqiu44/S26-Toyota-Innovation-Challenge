import dobotArm
import lib.DobotDllType as dType
import numpy as np
import mediapipe as mp
import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

H_matrix = np.load("HomographyMatrix.npy")


def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1.0])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]

def detect_human_hand_distance(camera_frame):
    """
    Detects a human hand in camera_frame and returns the XY-plane distance (mm)
    between the detected wrist and the robot arm's current position.
    Returns None if no hand is detected.
    """
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
    pose = dType.GetPose(dobotArm.api)
    arm_x, arm_y = pose[0], pose[1]

    distance = float(np.sqrt((hand_x - arm_x) ** 2 + (hand_y - arm_y) ** 2))
    return distance