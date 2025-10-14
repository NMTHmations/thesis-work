import cv2
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv

from project.detection.final.calc import getCenter
from project.detection.types.ODModel import ColorDetectorModel

# =====================
# --- CONFIGURATION ---
# =====================
CALIBRATION_FILE = "../../stereo_setup/parameters/calibration.xml"  # <-- XML formátum
FRAME_SIZE = (640, 480)
FPS = 60
N_MIN_POINTS = 8


# =====================
# --- LOAD CALIBRATION ---
# =====================
def load_calibration(file):
    """
    Betölti az XML kalibrációs adatokat.
    Visszatér: P1, P2, Q, stereoMapLeft, stereoMapRight
    """
    cv_file = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)

    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    stL_x = cv_file.getNode("stereoMapL_x").mat()
    stL_y = cv_file.getNode("stereoMapL_y").mat()
    stR_x = cv_file.getNode("stereoMapR_x").mat()
    stR_y = cv_file.getNode("stereoMapR_y").mat()

    cv_file.release()

    if P1 is None or P2 is None or Q is None:
        raise ValueError("❌ Hiányzó kalibrációs adatok az XML fájlban!")

    return P1, P2, Q, (stL_x, stL_y), (stR_x, stR_y)


# =====================
# --- TRIANGULATION ---
# =====================
def triangulate_point(ptL, ptR, P1, P2):
    """Kiszámolja a 3D koordinátát két képpontból"""
    pts4D = cv2.triangulatePoints(P1, P2, np.array(ptL).reshape(2, 1), np.array(ptR).reshape(2, 1))
    pts3D = pts4D[:3] / pts4D[3]
    return pts3D.flatten()


# =====================
# --- KALMAN FILTER ---
# =====================
def create_kalman():
    """Egyszerű 3D-s pozíció + sebesség Kalman-szűrő"""
    kf = cv2.KalmanFilter(6, 3)
    kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)
    kf.transitionMatrix = np.eye(6, 6, dtype=np.float32)
    for i in range(3):
        kf.transitionMatrix[i, i + 3] = 1.0
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
    return kf


# =====================
# --- DRAW 3D PATH ---
# =====================
def plot_trajectory(points, impact_point):
    points = np.array(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points[:, 0], points[:, 1], points[:, 2], label="Labda pálya", linewidth=2)
    ax.scatter(*impact_point, color="red", s=80, label="Becsapódás")

    ax.set_xlabel("X tengely")
    ax.set_ylabel("Y tengely")
    ax.set_zlabel("Z magasság")
    ax.legend()
    plt.show()


# =====================
# --- MAIN PIPELINE ---
# =====================
def main():
    print("🎬 Betöltés...")

    pointsR = {
        "P1":(1,329),
        "P2":(232,173)
    }

    pointsL = {
        "P1":(242,144),
        "P2":(520,192)
    }

    isAthaladt = False

    # Zöld szín tartománya HSV-ben
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    model = ColorDetectorModel(lower_green, upper_green)
    P1, P2, Q, stereoMapL, stereoMapR = load_calibration(CALIBRATION_FILE)
    kalman = create_kalman()

    capL = cv2.VideoCapture(1)
    capR = cv2.VideoCapture(2)

    capR.set(cv2.CAP_PROP_FPS, 60)
    capL.set(cv2.CAP_PROP_FPS, 60)

    print(capR.get(cv2.CAP_PROP_FPS))
    print(capL.get(cv2.CAP_PROP_FPS))

    boxAnnotator = sv.BoxAnnotator()

    trajectory = []
    print("✅ Rendszer inicializálva. Várakozás detektálásra...")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("⚠️ Nem sikerült képet olvasni!")
            break

        detectionsL, detectionsR = None, None
        try:
            resultsL = model.infer(frameL)
            resultsR = model.infer(frameR)
            detectionsL = model.getDetectionFromResult(resultsL, FRAME_SIZE)
            detectionsR = model.getDetectionFromResult(resultsR, FRAME_SIZE)

            # --- Tracker frissítése ---

            # --- Kirajzolás ---
            annotatedL = frameL.copy()
            annotatedL = boxAnnotator.annotate(scene=annotatedL, detections=detectionsL)

            annotatedR = frameR.copy()
            annotatedR = boxAnnotator.annotate(scene=annotatedR, detections=detectionsR)

        except:
            annotatedL = frameL.copy()
            annotatedR = frameR.copy()


        centerL, centerR = None, None
        if detectionsL is not None and detectionsR is not None:
            try:
                centerL = getCenter(detectionsL.xyxy[0])
                centerR = getCenter(detectionsR.xyxy[0])
            except:
                pass

        predicted = []

        if centerL is not None and centerR is not None:
            point3D = triangulate_point(centerL, centerR, P1, P2)
            measurement = np.array([[point3D[0]], [point3D[1]], [point3D[2]]], dtype=np.float32)
            kalman.correct(measurement)
            predicted = kalman.predict()

            trajectory.append(predicted[:3].flatten())

        cv2.circle(annotatedL, pointsL["P1"], 5, (0, 0, 255), -1)
        cv2.circle(annotatedL, pointsL["P2"], 5, (0, 0, 255), -1)
        cv2.circle(annotatedR, pointsR["P2"], 5, (0, 0, 255), -1)
        cv2.circle(annotatedR, pointsR["P2"], 5, (0, 0, 255), -1)
        cv2.line(annotatedL, pointsL["P1"], pointsL["P2"], (255, 0, 0), 2)
        cv2.line(annotatedR, pointsR["P1"], pointsR["P2"], (255, 0, 0), 2)
        cv2.imshow("Bal kamera", annotatedL)
        cv2.imshow("Jobb kamera", annotatedR)

        # --- AUTOMATIKUS MEGÁLLÁS ---
        if len(trajectory) > N_MIN_POINTS:
            recent = np.array(trajectory[-5:])
            dz = np.abs(np.diff(recent[:, 2])).mean()
            speed = np.linalg.norm(predicted[3:6])

            if (np.mean(recent[:, 2]) < 50 and dz < 1.0) or speed < 0.05:
                print("✅ Elég pont összegyűlt, számítás indul...")
                break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    if len(trajectory) < N_MIN_POINTS:
        print("⚠️ Nem gyűlt össze elég pont a becsapódás kiszámításához.")
        return

    trajectory = np.array(trajectory)

    # --- BECSAPÓDÁS: LEGALACSONYABB Z ---
    impact_idx = np.argmin(trajectory[:, 2])
    impact_point = trajectory[impact_idx]

    # --- NORMALIZÁLÁS 0–1 KÖZÉ ---
    x_min, x_max = np.min(trajectory[:, 0]), np.max(trajectory[:, 0])
    impact_normalized = (impact_point[0] - x_min) / (x_max - x_min + 1e-6)
    print(f"🎯 Becsapódás (normalizált): {impact_normalized:.3f}")

    # --- GÖRBE KIRAJZOLÁSA ---
    plot_trajectory(trajectory, impact_point)


if __name__ == "__main__":
    main()
