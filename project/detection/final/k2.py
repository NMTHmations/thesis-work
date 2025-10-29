#!/usr/bin/env python3
"""
kalman_ball_predictor.py

Single-camera, color-based ball detection + Kalman prediction prototype.

Usage:
    python kalman_ball_predictor.py --camera 0 --fps 60

Controls & workflow:
 - Click two points for START LINE (trigger) on the live image (first two clicks).
 - Click two points for IMPACT LINE (the line whose intersection we want) (next two clicks).
 - Press 'r' to reset lines, 'q' or ESC to quit.
 - Once the detected ball center crosses the start line, tracking/prediction starts.
 - The Kalman filter runs and the predicted future position line is computed. When
   the predicted trajectory intersects the impact line, the impact point is shown and
   mapped to [0,1] along the impact line.
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque

# ---------------------------
# Config / defaults
# ---------------------------
DEFAULT_HSV_LOWER = (45, 167, 0)      # white-ish (change for your ball color)
DEFAULT_HSV_UPPER = (89, 255, 255)


PLOT_HISTORY = 200  # how many predicted points to show in plot

# ---------------------------
# Helper geometry functions
# ---------------------------
def line_intersection_parametric(p, v, a, b):
    """
    Solve p + v*t = a + u*(b-a) for t and u.
    Returns (t, u) or (None, None) if degenerate.
    """
    # Solve 2x2: v * t - (b-a) * u = a - p
    A = np.column_stack((v, -(b - a)))  # 2x2
    rhs = a - p
    if np.linalg.matrix_rank(A) < 2:
        return None, None
    sol = np.linalg.solve(A, rhs)
    t, u = float(sol[0]), float(sol[1])
    return t, u

def project_point_on_segment(pt, a, b):
    """Return u in [0,1] for projection of pt onto segment a->b"""
    ab = b - a
    denom = (ab @ ab)
    if denom == 0:
        return 0.0
    u = float((pt - a) @ ab / denom)
    return np.clip(u, 0.0, 1.0)

# ---------------------------
# Kalman helper
# ---------------------------
def create_kalman(dt):
    """
    Create an OpenCV KalmanFilter for state [x, y, vx, vy] with measurement [x, y].
    """
    kf = cv2.KalmanFilter(4, 2)  # stateDim, measDim
    # State: [x, y, vx, vy]^T
    # Transition matrix A
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0 ],
        [0, 0, 0, 1 ]
    ], dtype=np.float32)

    # Measurement matrix H
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)
    # Process noise covariance Q
    q_pos = 1e-2 #0.01
    q_vel = 1e-2 #0.01
    kf.processNoiseCov = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)
    # Measurement noise R
    r_meas = 1.0
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r_meas
    # Posteriori error estimate P
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf

# ---------------------------
# HSV detector
# ---------------------------
def detect_ball_hsv(frame, hsv_lower, hsv_upper, min_area=80):
    """Return bbox center (x,y) in image coords or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    #remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, mask
    x,y,w,h = cv2.boundingRect(c)
    cx = x + w/2.0
    cy = y + h/2.0
    return (float(cx), float(cy)), mask

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="/dev/video3")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--hsv-lower", nargs=3, type=int, default=list(DEFAULT_HSV_LOWER))
    parser.add_argument("--hsv-upper", nargs=3, type=int, default=list(DEFAULT_HSV_UPPER))
    args = parser.parse_args()

    cam_idx = args.camera
    fps_target = args.fps
    hsv_lower = tuple(args.hsv_lower)
    hsv_upper = tuple(args.hsv_upper)

    cap = cv2.VideoCapture(cam_idx)

    if not cap.isOpened():
        print("Cannot open camera", cam_idx)
        return


    window_name = "Ball Predictor"

    cap.set(cv2.CAP_PROP_FPS, fps_target)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)


    dt = 1.0 / actual_fps if actual_fps > 0 else 1.0/60.0
    print(f"Camera FPS reported: {actual_fps:.2f}, using dt={dt:.5f}")

    # Kalman
    kf = create_kalman(dt)

    # State trackers
    started = False           # starts after start_line crossing
    start_line_pts = []       # two points by mouse
    impact_line_pts = []      # two points by mouse

    prev_center = None        # previous detected center (for crossing detection)
    predicted_points_history = deque(maxlen=PLOT_HISTORY)

    # Mouse callback to set lines
    cv2.namedWindow(window_name)

    def mouse_cb(event, x, y, flags, param):
        nonlocal start_line_pts, impact_line_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(start_line_pts) < 2:
                start_line_pts.append((x,y))
                print("Start line point added:", (x,y))
            elif len(impact_line_pts) < 2:
                impact_line_pts.append((x,y))
                print("Impact line point added:", (x,y))
            else:
                print("Both lines already set. Press 'r' to reset.")

    cv2.setMouseCallback(window_name, mouse_cb)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed, exiting.")
            break
        h,w = frame.shape[:2]

        # detect ball
        center, mask = detect_ball_hsv(frame, hsv_lower, hsv_upper)
        # draw mask small preview
        mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_col, (int(w*0.25), int(h*0.25)))
        cv2.rectangle(frame, (w - mask_small.shape[1] - 10, 10),
                      (w - 10, 10 + mask_small.shape[0]), (50,50,50), -1)
        frame[10 : 10 + mask_small.shape[0], w - mask_small.shape[1] - 10 : w - 10] = mask_small


        # Draw lines if available
        if len(start_line_pts) == 2:
            a,b = start_line_pts
            cv2.line(frame, a, b, (0,255,255), 2)
            cv2.putText(frame, "START LINE", (a[0], a[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        if len(impact_line_pts) == 2:
            a,b = impact_line_pts
            cv2.line(frame, a, b, (255,0,255), 2)
            cv2.putText(frame, "IMPACT LINE", (a[0], a[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        # draw measured center
        if center is not None:
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(frame, (cx,cy), 6, (0,255,0), -1)
            cv2.putText(frame, f"meas: {cx},{cy}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        # check trigger crossing (use center.y crossing the line)
        if not started and len(start_line_pts) == 2 and center is not None and prev_center is not None:
            # we'll consider crossing if the segment from prev_center to center intersects the start line segment
            p = np.array(prev_center)
            q = np.array(center)
            a = np.array(start_line_pts[0])
            b = np.array(start_line_pts[1])
            # Solve for intersection parameter - simpler: check sign of orientation
            def orient(u,v,w): return np.cross(v-u, w-u)
            # 2D cross product scalar
            o1 = np.cross(q-p, a-p)
            o2 = np.cross(q-p, b-p)
            o3 = np.cross(b-a, p-a)
            o4 = np.cross(b-a, q-a)
            # A robust segment intersection test
            def seg_intersect(p,q,a,b):
                def onseg(u,v,w):
                    return (min(u[0],v[0]) <= w[0] <= max(u[0],v[0]) and min(u[1],v[1]) <= w[1] <= max(u[1],v[1]))
                def orient2(u,v,w):
                    return (v[0]-u[0])*(w[1]-u[1]) - (v[1]-u[1])*(w[0]-u[0])
                o1 = orient2(p,q,a)
                o2 = orient2(p,q,b)
                o3 = orient2(a,b,p)
                o4 = orient2(a,b,q)
                if o1*o2 < 0 and o3*o4 < 0:
                    return True
                return False
            if seg_intersect(np.array(prev_center), np.array(center), a, b):
                started = True
                print("Trigger: start line crossed. Starting tracking/prediction.")

        # Kalman predict & update logic
        if started:
            # predict step
            pred = kf.predict()
            pred_x, pred_y = float(pred[0]), float(pred[1])
            # if we have a measurement -> update
            if center is not None:
                meas = np.array([[np.float32(center[0])], [np.float32(center[1])]])
                kf.correct(meas)
                # immediately after correct we can get posterior state
                post = kf.statePost
                px, py = float(post[0]), float(post[1])
            else:
                # no measurement: keep prediction as "current"
                px, py = pred_x, pred_y

            # store predicted point for plotting (normalized)
            predicted_points_history.append((px / w, py / h))

            # draw predicted current point
            cv2.circle(frame, (int(px), int(py)), 6, (0,0,255), -1)
            cv2.putText(frame, f"pred: {int(px)},{int(py)}", (int(px)+10, int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

            # compute future intersection with impact_line (using current posterior state and velocity)
            # state vector: [x,y,vx,vy]
            st = kf.statePost if kf.statePost is not None else pred
            state = np.array([float(st[0]), float(st[1]), float(st[2]), float(st[3])])
            p = state[0:2]
            v = state[2:4]  # pixels / second already (approx)
            if len(impact_line_pts) == 2:
                a = np.array(impact_line_pts[0], dtype=float)
                b = np.array(impact_line_pts[1], dtype=float)
                t, u = line_intersection_parametric(p, v, a, b)
                impact_pt = None
                if t is not None and t >= 0:
                    # compute impact point from prediction
                    impact_pt = p + v * t
                    # map to segment parameter u

                    # draw impact on image
                    ix, iy = int(round(impact_pt[0])), int(round(impact_pt[1]))
                    cv2.circle(frame, (ix, iy), 8, (255,0,0), -1)

                    cv2.putText(frame, f"Impact u={u:.3f}", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,200 ), 1)
                # else no valid intersection yet

        # update prev center
        if center is not None:
            prev_center = center

        # show frame
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting start/impact lines and state.")
            start_line_pts = []
            impact_line_pts = []
            started = False
            prev_center = None
            predicted_points_history.clear()
            # reset kalman state
            kf = create_kalman(dt)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
