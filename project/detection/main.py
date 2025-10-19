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
from collections import deque

from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter

from project.detection.final.fieldutils import FieldUtils
from project.detection.final.predictor import KFPredictor_2D
from project.detection.types.ODModel import ColorDetectorModel

# ---------------------------
# Config / defaults
# ---------------------------
DEFAULT_HSV_LOWER = (40, 40, 40)      # white-ish (change for your ball color)
DEFAULT_HSV_UPPER = (80, 255, 255)


PLOT_HISTORY = 200  # how many predicted points to show in plot

# ---------------------------
# Helper geometry functions
# ---------------------------

#----------------------
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

    model = ColorDetectorModel(hsv_lower, hsv_upper)

    cap = cv2.VideoCapture(cam_idx)

    if not cap.isOpened():
        print("Cannot open camera", cam_idx)
        return


    cap.set(cv2.CAP_PROP_FPS, fps_target)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)


    dt = 1.0 / actual_fps if actual_fps > 0 else 1.0/60.0
    print(f"Camera FPS reported: {actual_fps:.2f}, using dt={dt:.5f}")

    points = FieldUtils.readPoints("final/points.json")

    field = FieldUtils(
        topLeft=points["F_TOPL"],
        topRight=points["F_TOPR"],
        bottomLeft=points["F_BOTL"],
        bottomRight=points["F_BOTR"],
        startLine=(points["F_PT1"],points["F_PB1"]),
        impactLine=(points["F_GOALT"],points["F_GOALB"])
    )

    kfPredictor = KFPredictor_2D(dt=dt)



    # State trackers
    started = False           # starts after start_line crossing

    prev_center = None        # previous detected center (for crossing detection)
    predicted_points_history = deque(maxlen=PLOT_HISTORY)

    # Mouse callback to set lines
    window_name = "Ball Predictor"
    cv2.namedWindow(window_name)


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
        if points:
            field.drawField(frame)
        # draw measured center
        if center is not None:
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(frame, (cx,cy), 6, (0,255,0), -1)
            cv2.putText(frame, f"meas: {cx},{cy}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        # check trigger crossing (use center.y crossing the line)
        if not started and points and center is not None and prev_center is not None:
            a = np.array(points["F_PT1"])
            b = np.array(points["F_PB1"])
            # Solve for intersection parameter - simpler: check sign of orientation

            def seg_intersect(p,q,a,b):
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
            cv2.putText(frame, f"pred: {int(px)},{int(py)}", (int(px)+10, int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            # compute future intersection with impact_line (using current posterior state and velocity)
            # state vector: [x,y,vx,vy]
            st = kf.statePost if kf.statePost is not None else pred
            state = np.array([float(st[0]), float(st[1]), float(st[2]), float(st[3])])
            p = state[0:2]
            v = state[2:4]  # pixels / second already (approx)
            if points:
                a = np.array(points["F_GOALT"], dtype=float)
                b = np.array(points["F_GOALB"], dtype=float)
                t, u = line_intersection_parametric(p, v, a, b)
                impact_pt = None
                if t is not None and t >= 0:
                    # compute impact point from prediction
                    impact_pt = p + v * t
                    # map to segment parameter u
                    u_mapped = project_point_on_segment(impact_pt, a, b)

                    # draw impact on image
                    ix, iy = int(round(impact_pt[0])), int(round(impact_pt[1]))
                    cv2.circle(frame, (ix, iy), 8, (255,0,0), -1)
                    cv2.putText(frame, f"Impact u={u_mapped:.3f}", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
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
