#This code is a simplified implementation of a collaborative robotics system that detects plates and targets using computer vision, 
#and then commands a Dobot robotic arm to pick and place objects accordingly. The system operates in three phases: scanning for plates, 
#scanning for targets, and executing the pick/place operations. 
#Stability checks are implemented to ensure reliable detection before proceeding to the next phase.

# Note: there are parameters that are useful to the successful operation of the robot arm. Read through the code before running the program.

# How to use: 
# 1. Ensure you have the Dobot robotic arm set up and connected to your computer.
# 2. Place the plates (drop zones) and targets (red blocks) within the camera's
# field of view.
# 3. Run the script. The system will first scan for plates, then targets, and finally execute the pick/place operations based on the detected positions.
# 4. Monitor the console output and the video feed for feedback on the system's status and operations

#Other Useful Codes you can use:
#dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE, rHead): moves the robot to the specified (x, y, z) coordinates with a specified rotation for the end effector (rHead). Z_SAFE is a predefined constant that ensures the robot maintains a safe height to avoid collisions when moving horizontally.



import dobotArm
import lib.DobotDllType as dType
import numpy as np
import mediapipe as mp
import cv2
import time
import threading
import handDetection

"""CONSTANTS"""

Z_SAFE = 40 #what is the clearance distance for the robot arm to avoid collisions when moving horizontally?
Z_PICK = -25 #what is the  height for the robot claw to successfully pick up the target?
STABILITY_LIMIT = 60  #how many consecutive frames of stable detection before we "lock in" the positions and move to the next phase? (at 30fps, 60 frames is about 2 seconds)
PIXEL_TOLERANCE = 10  #object can move at most this # of pixels to be considered stationary

machine_state = "scanning plate" 

# --- INITIALIZATION FOR CAMERA TRANSFORMATION ---
# MAKE SURE THAT YOU HAVE RAN calibrateCamera.py FIRST TO GENERATE THE camera_params.npz FILE
api = dType.load()
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
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]


# State machine logic to control the flow of the program through the three phases: scanning for plates, scanning for targets, and executing pick/place operations.
# THIS STATE MACHINE IS TOO SIMPLE. Can you think of logics that should change the robot's sequnece of actions?
# Ex: what if the robot fails to pick up a target? should it retry? should it go back to scanning for targets in case the target was moved? what if a new plate is added during the pick/place phase?
# What if a human's hand is in sight during pick/place phase? (safety first!)

_hand_distance = None
_hand_lock = threading.Lock()

def _update_hand_position():
    global _hand_distance
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        dist = handDetection.detect_human_hand_distance(frame)
        with _hand_lock:
            _hand_distance = dist

threading.Thread(target=_update_hand_position, daemon=True).start()

def _hand_too_close(threshold=50):
    with _hand_lock:
        dist = _hand_distance
    return dist is not None and abs(dist) < threshold

def move_with_safety(api, tx, ty, tz, threshold=50, steps=20):
    pose = dobotArm.get_pose(api)
    sx, sy, sz = pose[0], pose[1], pose[2]
    for i in range(1, steps + 1):
        if _hand_too_close(threshold):
            return False
        t = i / steps
        dobotArm.move_to_xyz(api, sx + t * (tx - sx), sy + t * (ty - sy), sz + t * (tz - sz))
    return True

def pick_up_object_from_point(api, position_vec, drop_zone=None):
    rx, ry = pixel_to_robot(*position_vec, H_matrix)

    if not move_with_safety(api, rx, ry, Z_SAFE): return
    if not move_with_safety(api, rx, ry, Z_PICK): return

    dobotArm.close_gripper(api)
    move_with_safety(api, rx, ry, Z_SAFE)

    if drop_zone is not None:
        dx, dy = drop_zone
        if not move_with_safety(api, dx, dy, Z_SAFE): return
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)
        move_with_safety(api, dx, dy, Z_SAFE)

def cleanup():
    cap.release()
    cv2.destroyAllWindows()