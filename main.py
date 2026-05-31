import asyncio
import math
import websockets
import json
import threading
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import cv2

from scipy.spatial.transform import Rotation as R
import time

# Global dictionary to store the latest pose from the Android app
latest_pose = {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
}

r_calib = None
s_calib = None
t_calib = None
k_calib = None
# --- WEBSOCKET SERVER LOGIC ---

p = np.array([
    [0, 0, 0],
    [1512, 0, 0],
    [0, 982, 0],
    [1512, 982, 0],
])

def solve_similarity_lines(p, l, v):
    """
    Solve

        p_i = s R (l_i + k_i v_i) + t

    Inputs
    ------
    p : (N,3) array
    l : (N,3) array
    v : (N,3) array (should be unit vectors)

    Returns
    -------
    R : (3,3)
    s : float
    t : (3,)
    k : (N,)
    """

    p = np.asarray(p, dtype=float)
    l = np.asarray(l, dtype=float)
    v = np.asarray(v, dtype=float)

    n = len(p)

    # Normalize v for safety
    v = v / np.linalg.norm(v, axis=1, keepdims=True)

    # Initial guess
    omega0 = np.zeros(3)
    log_s0 = 0.0
    t0 = np.mean(p - l, axis=0)
    k0 = np.zeros(n)

    x0 = np.concatenate([
        omega0,
        [log_s0],
        t0,
        k0
    ])

    def residuals(x):
        omega = x[0:3]
        log_s = x[3]
        t = x[4:7]
        k = x[7:]

        rot = R.from_rotvec(omega).as_matrix()
        s = np.exp(log_s)

        pred = s * (rot @ (l + k[:, None] * v).T).T + t

        return (pred - p).ravel()

    result = least_squares(
        residuals,
        x0,
        method='trf',
        loss='linear',
        x_scale='jac',
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=1000
    )

    x = result.x

    omega = x[0:3]
    log_s = x[3]
    t = x[4:7]
    k = x[7:]

    rot = R.from_rotvec(omega).as_matrix()
    s = np.exp(log_s)

    return rot, s, t, k

calibration_points = []
calibration_vectors = []

def rotate_x_axis(w, x, y, z):
    """
    Rotates the X-axis basis vector [1, 0, 0] by a quaternion (w, x, y, z).
    Automatically normalizes the quaternion to prevent scaling errors.
    """
    # Normalize the quaternion
    mag = math.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w / mag, x / mag, y / mag, z / mag
    
    # Optimized formula for multiplying [1, 0, 0]
    vx = 1 - 2 * (y**2 + z**2)
    vy = 2 * (x * y + w * z)
    vz = 2 * (x * z - w * y)
    
    return [vx, vy, vz]


async def handle_connection(websocket):
    global r_calib, s_calib, t_calib, k_calib
    async for message in websocket:
        data = json.loads(message)
        
        # Check if it's the special button message
        if data.get("type") == "special_button":
            print(f"Special Button Pressed: {data['button_id']}")
            pos = latest_pose["position"]
            rot = latest_pose["rotation"]
            vx, vy, vz = rotate_x_axis(rot["w"], rot["x"], rot["y"], rot["z"])
            calibration_points.append([pos["x"], pos["y"], pos["z"]])
            calibration_vectors.append([vx, vy, vz])
            print(f"Collected Calibration Point: {calibration_points[-1]}, Vector: {calibration_vectors[-1]}")
            if len(calibration_points) >= 4:
                print("Performing calibration with collected points and vectors...")
                r_calib, s_calib, t_calib, k_calib = solve_similarity_lines(p, calibration_points, calibration_vectors)
                print("Calibration results:")
                print("Rotation Matrix:\n", r_calib)
                print("Scale:", s_calib)
                print("Translation:", t_calib)
                # Clear calibration data after processing
                calibration_points.clear()
                calibration_vectors.clear()
            # Handle your special logic here
        else:
            # Handle your standard pose data
            latest_pose["position"] = data["position"]
            latest_pose["rotation"] = data["rotation"]

async def start_websocket_server():
    # Bind to 0.0.0.0 to accept connections from other devices on the local network
    async with websockets.serve(handle_connection, "0.0.0.0", 8080):
        print("WebSocket Server running on ws://0.0.0.0:8080")
        await asyncio.Future()  # Run forever

def run_server_in_thread():
    # Set up a new asyncio event loop for the background thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_websocket_server())




def main():
    # 1. Start the WebSocket server in a background thread
    server_thread = threading.Thread(target=run_server_in_thread, daemon=True)
    server_thread.start()

    # 2. Setup Matplotlib 3D plotting in interactive mode
    #plt.ion()
    #fig = plt.figure(figsize=(8, 8))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.set_title("Real-Time ARCore Camera Z-Axis")

    print("Opening visualization window. Close the window or press Ctrl+C in terminal to exit.")

    try:
        while True:
            

            pos = latest_pose["position"]
            rot = latest_pose["rotation"]

            tx, ty, tz = pos["x"], pos["y"], pos["z"]
            qx, qy, qz, qw = rot["x"], rot["y"], rot["z"], rot["w"]
            x_axis = rotate_x_axis(qw, qx, qy, qz)
            

            #print(tx, ty, tz)
            #print(qx, qy, qz, qw)
            
            # generate an all white full screen image and show it with opencv, and overlay the text of the position and rotation on it, at 1512 x 982 resolution
            img = np.ones((982, 1512, 3), dtype=np.uint8)



            if r_calib is not None:
                # apply the transformation to the position vector and print it
                calibrated_pos = s_calib * (r_calib @ np.array([tx, ty, tz])) + t_calib
                # apply the rotation to the direction vector and print it
                calibrated_dir = s_calib * (r_calib @ np.array(x_axis))
                # find where the line intersects the plane z=0 in the calibrated coordinate system
                if calibrated_dir[2] != 0:
                    t_intersect = -calibrated_pos[2] / calibrated_dir[2]
                    intersection_point = calibrated_pos + t_intersect * calibrated_dir
                    print("Calibrated Position:", calibrated_pos)
                    print("Intersection with z=0 plane:", intersection_point)
                else:
                    print("Calibrated Position:", calibrated_pos)
                    print("Direction vector is parallel to z=0 plane, no intersection.")
                    img = np.ones((982, 1512, 3), dtype=np.uint8)
                    cv2.putText(img, f"Position: ({tx:.2f}, {ty:.2f}, {tz:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(img, f"Rotation (quat): ({qx:.2f}, {qy:.2f}, {qz:.2f}, {qw:.2f})", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    # draw a red circle at the intersection point                
                    cv2.circle(img, (int(intersection_point[0]), int(intersection_point[1])), 5, (0, 0, 255), -1)

            cv2.imshow("ARCore Visualization", img)
            cv2.waitKey(1) 

            
            # Clear previous frame

            
    except KeyboardInterrupt:
        print("\nExiting program.")
    finally:
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    main()
