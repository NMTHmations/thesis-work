import cv2
import json
import argparse

from project.detection.final.calc import interpolate

points_clicked = []  # list of clicked points

def mouse_callback(event, x, y, flags, param):
    global points_clicked
    img = param
    if event == cv2.EVENT_LBUTTONDOWN and len(points_clicked) < 4:
        points_clicked.append((x, y))
        print(f"Pont kiválasztva: ({x},{y})")
        # Rajzoljuk a pontokat
        for px, py in points_clicked:
            cv2.circle(img, (px, py), 5, (0, 0, 255), -1)
        cv2.imshow("Image", img)


def main(args):
    global points_clicked

    fpath = args.fpath
    prefix = args.pref
    # Kamera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("❌ Nem sikerült megnyitni a kamerát!")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Nem sikerült képet készíteni!")
        return

    img = frame.copy()
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouse_callback, img)

    print("👉 Kattints a képen a négy sarokpontot (TOPL, BOTL, TOPR, BOTR) sorrendben. ESC a kilépéshez.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if len(points_clicked) == 4:
            break

    cv2.destroyAllWindows()

    if len(points_clicked) != 4:
        print("❌ Nem választottad ki a 4 sarokpontot!")
        return

    # Elnevezett sarokpontok
    S_TOPL, S_BOTL, S_TOPR, S_BOTR = points_clicked

    # Generáljuk a további pontokat
    points_dict = {
        prefix+"_TOPL": S_TOPL,
        prefix+"_BOTL": S_BOTL,
        prefix+"_TOPR": S_TOPR,
        prefix+"_BOTR": S_BOTR,
        prefix+"_PT1": interpolate(S_TOPL, S_TOPR, 1/6),
        prefix+"_PT2": interpolate(S_TOPL, S_TOPR, 1/3),
        prefix+"_PB1": interpolate(S_BOTL, S_BOTR, 1/6),
        prefix+"_PB2": interpolate(S_BOTL, S_BOTR, 1/3),
        prefix+"_GOALT": interpolate(S_TOPR, S_BOTR, 1/3),
        prefix+"_GOALB": interpolate(S_TOPR, S_BOTR, 2/3),
    }

    print("✅ Pontok meghatározva:")
    for k, v in points_dict.items():
        print(f"{k}: {v}")

    # Mentés JSON-be dictionary formában
    with open(fpath, "w") as f:
        json.dump(points_dict, f, indent=4)
    print(f"✅ Pontok elmentve a(z) {fpath} fájlba")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=1, help="Kamera index")
    parser.add_argument("--fpath", type=str, default="points_F_stereo.json", help="File path")
    parser.add_argument("--pref", type=str, default="F", help="Parameter prefixes")

    args = parser.parse_args()
    main(args)
