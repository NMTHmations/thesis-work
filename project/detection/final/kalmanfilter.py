#!/usr/bin/env python3
"""
stereo_kalman_impact.py

Two-camera stereo prototype:
- /dev/video3 (left), /dev/video4 (right)
- load stereo calib (XML with P1,P2 or cameraMatrix/dist/R/T)
- HSV ball detection (greenish default)
- triangulate 3D point per frame, 3D Kalman filter tracking
- left image: set START line (2 clicks)
- right image: set IMPACT line (2 clicks)
- if impact line intersected by predicted projected trajectory -> compute impact and map 0..1
- if neither camera sees the impact line we can fallback to ground-plane intersection
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple

# ----------------- Config -----------------
LEFT_DEVICE = 1
RIGHT_DEVICE = 0
CALIB_FILE = "../../stereo_setup/parameters/calibration.xml"   # set your xml/yml path
HSV_LOWER = (40, 40, 40)   # greenish range given by you (modify if needed)
HSV_UPPER = (80, 255, 255)
MIN_AREA = 80
DT_MAX = 2.0   # seconds ahead to search for impact
DT_STEP = 0.01 # time step when sampling future projection
GROUND_Y = 0.0 # optional world ground plane Y (if you want fallback)
# ------------------------------------------


def load_stereo_calibration(xml_path):
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    cameraMatrix1 = fs.getNode("cameraMatrix1").mat()
    distCoeffs1 = fs.getNode("distCoeffs1").mat()
    cameraMatrix2 = fs.getNode("cameraMatrix2").mat()
    distCoeffs2 = fs.getNode("distCoeffs2").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    R1 = fs.getNode("R1").mat()
    R2 = fs.getNode("R2").mat()
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    Q = fs.getNode("Q").mat()
    fs.release()
    return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, R1, R2, P1, P2, Q

def detect_ball_hsv(frame, hsv_lower, hsv_upper, min_area=MIN_AREA):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None, mask
    x,y,w,h = cv2.boundingRect(c)
    cx = x + w/2.0
    cy = y + h/2.0
    return (float(cx), float(cy)), mask

def triangulate_point(ptL, ptR, P1, P2):
    """Triangulate single correspondence, return 3D (X,Y,Z) in camera1 coords as np.array shape (3,)"""
    pts1 = np.array([[ptL[0]], [ptL[1]]], dtype=float)
    pts2 = np.array([[ptR[0]], [ptR[1]]], dtype=float)
    pts4 = cv2.triangulatePoints(P1, P2, pts1, pts2)  # 4xN
    pts3 = pts4[:3,0] / pts4[3,0]
    return pts3

def create_kalman_3d(dt: float):
    """State: [x,y,z,vx,vy,vz], meas: [x,y,z]"""
    kf = cv2.KalmanFilter(6, 3)
    # Transition
    A = np.eye(6, dtype=np.float32)
    for i in range(3):
        A[i, i+3] = dt
    kf.transitionMatrix = A
    # Measurement matrix: map state -> measured position
    H = np.zeros((3,6), dtype=np.float32)
    H[0,0] = 1; H[1,1] = 1; H[2,2] = 1
    kf.measurementMatrix = H
    # Process noise
    q_pos = 1e-2
    q_vel = 1e-2
    Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float32)
    kf.processNoiseCov = Q
    # Measurement noise
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1.0
    # Error covariance
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf

def project_point(P, X):
    """Project 3D X (3,) using 3x4 P to image pixel (u,v)"""
    Xh = np.array([X[0], X[1], X[2], 1.0], dtype=float)
    uvw = P.dot(Xh)
    if abs(uvw[2]) < 1e-8:
        return None
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    return np.array([u, v], dtype=float)

def segment_intersect_2d(p1, p2, a, b):
    """Return True if segment p1-p2 intersects a-b (2D)."""
    def orient(a,b,c): return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    o1 = orient(p1,p2,a)
    o2 = orient(p1,p2,b)
    o3 = orient(a,b,p1)
    o4 = orient(a,b,p2)
    return (o1*o2 < 0) and (o3*o4 < 0)

def project_trajectory_and_find_impact(p0, v, P_right, impact_a, impact_b, dt_step=DT_STEP, t_max=DT_MAX):
    """
    Sample future 3D positions p(t) = p0 + v*t, project to right image by P_right,
    test for intersection with impact segment (impact_a -> impact_b) in image plane.
    Return impact 3D point and mapped u (0..1) along impact segment, or (None, None).
    """
    prev_proj = project_point(P_right, p0)
    if prev_proj is None:
        return None, None
    t = dt_step
    while t <= t_max:
        X = p0 + v * t
        proj = project_point(P_right, X)
        if proj is None:
            t += dt_step; continue
        # check segment prev_proj-proj against impact segment
        if segment_intersect_2d(prev_proj, proj, impact_a, impact_b):
            # compute impact 3D by linear interpolation between p(t-dt) and p(t)
            # refine by bisect (simple)
            lo = t - dt_step
            hi = t
            for _ in range(8):
                mid = (lo+hi)/2.0
                Xm = p0 + v*mid
                pm = project_point(P_right, Xm)
                if pm is None:
                    lo = mid
                    continue
                if segment_intersect_2d(prev_proj, pm, impact_a, impact_b):
                    hi = mid
                else:
                    lo = mid
                prev_proj = pm
            t_hit = (lo+hi)/2.0
            X_hit = p0 + v * t_hit
            # map to segment parameter u
            ab = impact_b - impact_a
            denom = (ab @ ab)
            if denom == 0:
                u = 0.0
            else:
                u = float(((project_point(P_right, X_hit) - impact_a) @ ab) / denom)
                u = np.clip(u, 0.0, 1.0)
            return X_hit, u
        prev_proj = proj
        t += dt_step
    return None, None

def intersect_with_ground(p0, v, ground_y=GROUND_Y):
    """Find t so that p0[1] + v[1]*t = ground_y, return 3D point or None"""
    vy = v[1]
    if abs(vy) < 1e-8:
        return None
    t = (ground_y - p0[1]) / vy
    if t < 0:
        return None
    return p0 + v * t

# ---------- Main run ----------
def main():
    # load calib
    P1, P2, cam1, dist1, cam2, dist2, R, T, Qmat = None, None, None, None, None, None, None, None, None
    try:
        cam1_params = load_stereo_calibration(CALIB_FILE)
        _,_, _, _, _, _, _, _, P1, P2, Qmat = cam1_params
        print("Loaded calib. P1/P2 present:", P1 is not None and P2 is not None)
    except Exception as e:
        print("Couldn't load calib:", e)
        return

    if P1 is None or P2 is None:
        print("Calibration file missing P1/P2. Please provide stereoRectify output (P1,P2).")
        return

    # open cameras
    capL = cv2.VideoCapture(LEFT_DEVICE)
    capR = cv2.VideoCapture(RIGHT_DEVICE)
    if not capL.isOpened() or not capR.isOpened():
        print("Cannot open one or both cameras")
        return

    # try set same resolution / fps if you wish (optional)
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capL.set(cv2.CAP_PROP_FPS, 60)
    capR.set(cv2.CAP_PROP_FPS, 60)
    # compute dt from one camera
    fps = capL.get(cv2.CAP_PROP_FPS) or 60.0
    dt = 1.0 / fps
    print(f"Using dt={dt:.4f} (fps={fps})")

    kf = create_kalman_3d(dt)

    # UI state
    left_pts = []   # start line points (on left image)
    right_pts = []  # impact line points (on right image)
    started = False
    prev_center_left = None

    window_name = "Left | Right (press q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def mouse_cb(event, x, y, flags, param):
        nonlocal left_pts, right_pts
        side = param  # "left" or "right"
        if event == cv2.EVENT_LBUTTONDOWN:
            if side == "left":
                if len(left_pts) < 2:
                    left_pts.append((x,y))
                    print("Left start point:", (x,y))
            else:
                if len(right_pts) < 2:
                    right_pts.append((x,y))
                    print("Right impact point:", (x,y))

    # We'll display left and right side-by-side, so set callbacks accordingly by clicking in halves
    def global_mouse(event, x, y, flags, param):
        # param not used
        # determine which half clicked
        # assume we'll show left on left half and right on right half
        nonlocal left_pts, right_pts
        # get window size (approx) by grabbing last frames; simpler: assume both frames same width w_disp
        # We'll pass the frame width in param later; here check param
        # But cv2.setMouseCallback allows only one callback; use global state to route clicks.
        pass

    # simpler: set two named windows and callbacks separately
    cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Left", 1280,720)
    cv2.resizeWindow("Right", 1280,720)
    cv2.setMouseCallback("Left", lambda e,x,y,f,p: mouse_cb(e,x,y,f,"left"))
    cv2.setMouseCallback("Right", lambda e,x,y,f,p: mouse_cb(e,x,y,f,"right"))

    # loop
    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("Frame read failed")
            break
        hL,wL = frameL.shape[:2]

        # detect centers
        centerL, maskL = detect_ball_hsv(frameL, HSV_LOWER, HSV_UPPER, MIN_AREA)
        centerR, maskR = detect_ball_hsv(frameR, HSV_LOWER, HSV_UPPER, MIN_AREA)

        # draw masks small
        #mask_s = cv2.resize(cv2.cvtColor(maskL, cv2.COLOR_GRAY2BGR), (int(wL*0.25), int(hL*0.25)))
        #frameL[10:10+mask_s.shape[0], 10:10+mask_s.shape[1]] = mask_s

        # draw centers
        if centerL is not None:
            cv2.circle(frameL, (int(centerL[0]), int(centerL[1])), 6, (0,255,0), -1)
        if centerR is not None:
            cv2.circle(frameR, (int(centerR[0]), int(centerR[1])), 6, (0,255,0), -1)

        # draw start/impact lines
        if len(left_pts) == 2:
            cv2.line(frameL, tuple(left_pts[0]), tuple(left_pts[1]), (0,255,255), 2)
            cv2.putText(frameL, "START LINE", (left_pts[0][0], left_pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        if len(right_pts) == 2:
            cv2.line(frameR, tuple(right_pts[0]), tuple(right_pts[1]), (255,0,255), 2)
            cv2.putText(frameR, "IMPACT LINE", (right_pts[0][0], right_pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        # check start trigger on left (segment intersection previous->current with start line)
        if not started and len(left_pts) == 2 and centerL is not None and prev_center_left is not None:
            if segment_intersect_2d(np.array(prev_center_left), np.array(centerL), np.array(left_pts[0]), np.array(left_pts[1])):
                started = True
                print("Start triggered")

        # if started do triangulate + kalman
        if started and centerL is not None and centerR is not None:
            # triangulate 3D
            X = triangulate_point(centerL, centerR, P1, P2)  # 3D in cam1 coords
            # measurement and update
            meas = np.array([[np.float32(X[0])],[np.float32(X[1])],[np.float32(X[2])]])
            kf.predict()
            kf.correct(meas)
            state = kf.statePost.flatten()  # x,y,z,vx,vy,vz
            p0 = state[0:3].astype(float)
            v = state[3:6].astype(float)
            # draw projected current predicted point on right image for visualization
            proj_cur = project_point(P2, p0)
            if proj_cur is not None:
                cv2.circle(frameR, (int(round(proj_cur[0])), int(round(proj_cur[1]))), 6, (0,0,255), -1)
            # compute impact by projecting future trajectory onto right image and intersect with impact segment
            impact3d, u_mapped = None, None
            if len(right_pts) == 2:
                a = np.array(right_pts[0], dtype=float)
                b = np.array(right_pts[1], dtype=float)
                impact3d, u_mapped = project_trajectory_and_find_impact(p0, v, P2, a, b)
            # fallback: if we didn't get impact from image line, intersect with ground plane in 3D
            if impact3d is None:
                ground_hit = intersect_with_ground(p0, v, GROUND_Y)
                if ground_hit is not None:
                    impact3d = ground_hit
                    # project to right image to get pixel
                    uv = project_point(P2, impact3d)
                    if uv is not None and len(right_pts)==2:
                        a = np.array(right_pts[0], dtype=float)
                        b = np.array(right_pts[1], dtype=float)
                        ab = b - a
                        denom = (ab @ ab)
                        if denom != 0:
                            u_mapped = float(((uv - a) @ ab) / denom)
                            u_mapped = float(np.clip(u_mapped, 0.0, 1.0))
                        else:
                            u_mapped = 0.0
                # else still None
            # draw impact on right
            if impact3d is not None:
                uv_impact = project_point(P2, impact3d)
                if uv_impact is not None:
                    ix, iy = int(round(uv_impact[0])), int(round(uv_impact[1]))
                    cv2.circle(frameR, (ix, iy), 8, (255,0,0), -1)
                    text = f"Impact u={u_mapped:.3f}" if u_mapped is not None else "Impact"
                    cv2.putText(frameR, text, (ix+10, iy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            # print 3D impact for debug
            if impact3d is not None:
                cv2.putText(frameR, f"Impact3D: {impact3d}, mapped: {u_mapped}",(100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # keep prev center
        if centerL is not None:
            prev_center_left = centerL

        # show windows
        cv2.imshow("Left", frameL)
        cv2.imshow("Right", frameR)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:
            break
        elif k == ord("r"):
            left_pts = []
            right_pts = []
            started = False
            kf = create_kalman_3d(dt)
            print("Reset")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
