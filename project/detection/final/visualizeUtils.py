import cv2
import numpy as np


def createVisualizeFrame():
    # Fehér háttér (vagy szürke, mint a példádban)
    goalFrame = np.ones((360, 640, 3), dtype=np.uint8) * 100

    # PNG betöltése alfa csatornával
    goalPic = cv2.imread("goal.png", cv2.IMREAD_UNCHANGED)  # BGR + Alfa

    # Átméretezés 448x252-re
    goalPic = cv2.resize(goalPic, (448, 252))

    # Szétválasztás
    bgr_img = goalPic[:, :, :3]
    if goalPic.shape[2] == 4:  # ha van alfa csatorna
        alpha = goalPic[:, :, 3] / 255.0
    else:  # ha nincs alfa, teljesen látható
        alpha = np.ones((252, 448))

    # Középre pozicionálás
    x_offset = (640 - 448) // 2
    y_offset = (360 - 252) // 2

    # ROI kiválasztása
    roi = goalFrame[y_offset:y_offset + 252, x_offset:x_offset + 448]

    # Alkalmazzuk az alfa csatornát
    for c in range(3):  # B, G, R csatornák
        roi[:, :, c] = (bgr_img[:, :, c] * alpha + roi[:, :, c] * (1 - alpha)).astype(np.uint8)

    # Visszahelyezzük a frame-re (opcionális, mert a roi már mutat a frame-re)
    goalFrame[y_offset:y_offset + 252, x_offset:x_offset + 448] = roi


    cv2.line(goalFrame, (x_offset, y_offset-20), (640-x_offset, y_offset-20), (0,0,255), 2, 1)
    cv2.line(goalFrame,(x_offset-20, y_offset), (x_offset-20, 360-y_offset), (0,0,255), 2, 1)

    cv2.circle(goalFrame, (x_offset, y_offset-20), 5, (255,0,0), -1)
    cv2.circle(goalFrame, (640-x_offset, y_offset-20), 5, (255,0,0), -1)
    cv2.putText(goalFrame, f"{0.0}", (x_offset-20,y_offset-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    cv2.putText(goalFrame, f"{1.0}", (640-x_offset-10,y_offset-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    cv2.circle(goalFrame, (x_offset-20, y_offset), 5, (255,0,0), -1)
    cv2.circle(goalFrame, (x_offset-20, 360-y_offset), 5, (255,0,0), -1)
    cv2.putText(goalFrame, f"{0.0}", (x_offset-55, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(goalFrame, f"{1.0}", (x_offset - 55, 360-y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                2)

    return goalFrame


def placeMappedImpact(x,y):
    pass


