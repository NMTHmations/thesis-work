#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import json

from project.detection.final.finalMain import Camera, Window

class PointSelector:
    def __init__(self, camera : Camera, outputDir="./"):
        self.outputDir = outputDir
        self.points = []

        self.window = Window(f"Preview{hash(camera)}", 896, 504)
        self.camera = camera

        cv2.setMouseCallback(self.window.name, self._mouse_cb)

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"Point added: {(x, y)}")
            else:
                print("Already selected two points. Press "r" to reset")

    # --- pontok kirajzolása ---
    def _draw_points(self, frame, points, color):
        for i, p in enumerate(points):
            cv2.circle(frame, p, 6, color, -1)
            cv2.putText(frame, f"{i + 1}", (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- pontok mentése JSON-ba ---
    def _save_points(self):
        os.makedirs(self.outputDir, exist_ok=True)

        fpath = os.path.join(self.outputDir, "points.json")

        data_front = {"GOAL1": self.points[0], "GOAL2": self.points[1]}

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data_front, f, indent=4, ensure_ascii=False)

        print(f"[INFO] Points saved:\n - {fpath}")

    # --- fő futtató függvény ---
    def run(self):
        print("Select two point by clicking left mouse button")
        print("Press 'r' to reset, 's' to save points, 'q' or ESC to exit.")

        while True:

            success, frame = self.camera.capture()

            if not success:
                print("Couldn't capture frame.")
                break

            self._draw_points(frame, self.points, (0, 255, 0))

            self.window.showFrame(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('r'):
                self.points.clear()
                print("[RESET] Points reset.")
            elif key == ord('s'):
                if len(self.points) == 2:
                    self._save_points()
                else:
                    print("[WARNING] You must select two points first.")

        del self.camera
        del self.window




class StereoPointSelector:
    def __init__(self, cameraFront: Camera, cameraSide: Camera, output_dir="./"):
        self.output_dir = output_dir
        self.pointsFront = []
        self.pointsSide = []

        self.frontWindow = Window("Front Camera", 896, 504)
        self.sideWindow = Window("Side Camera", 896, 504)

        self.camFront = cameraFront
        self.camSide = cameraSide

        if not self.camFront.camera.isOpened() or not self.camSide.camera.isOpened():
            raise RuntimeError("Nem sikerült mindkét kamerát megnyitni.")

        cv2.setMouseCallback(self.frontWindow.name, self._mouse_front)
        cv2.setMouseCallback(self.sideWindow.name, self._mouse_side)

    # --- egér callbackek ---
    def _mouse_front(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pointsFront) < 2:
                self.pointsFront.append((x, y))
                print(f"[FRONT] Pont hozzáadva: {(x, y)}")
            else:
                print("[FRONT] Már két pont van, nyomj 'r'-t a resethez.")

    def _mouse_side(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pointsSide) < 2:
                self.pointsSide.append((x, y))
                print(f"[SIDE] Pont hozzáadva: {(x, y)}")
            else:
                print("[SIDE] Már két pont van, nyomj 'r'-t a resethez.")

    # --- pontok kirajzolása ---
    def _draw_points(self, frame, points, color):
        for i, p in enumerate(points):
            cv2.circle(frame, p, 6, color, -1)
            cv2.putText(frame, f"{i + 1}", (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- pontok mentése JSON-ba ---
    def _save_points(self):
        os.makedirs(self.output_dir, exist_ok=True)

        front_path = os.path.join(self.output_dir, "pointsFront.json")
        side_path = os.path.join(self.output_dir, "pointsSide.json")

        data_front = {"GOAL1": self.pointsFront[0], "GOAL2": self.pointsFront[1]}
        data_side = {"GOAL1": self.pointsSide[0], "GOAL2": self.pointsSide[1]}

        with open(front_path, "w", encoding="utf-8") as f:
            json.dump(data_front, f, indent=4, ensure_ascii=False)
        with open(side_path, "w", encoding="utf-8") as f:
            json.dump(data_side, f, indent=4, ensure_ascii=False)

        print(f"[INFO] Pontok elmentve JSON formátumban:\n - {front_path}\n - {side_path}")

    # --- fő futtató függvény ---
    def run(self):
        print("Bal egérgombbal jelölj 2-2 pontot mindkét kameraképen.")
        print("Nyomj 's'-t a mentéshez, 'r'-t a törléshez, 'q' vagy ESC a kilépéshez.")

        while True:
            successFront, frameFront = self.camFront.capture()
            successSide, frameSide = self.camSide.capture()

            if not successFront or not successSide:
                print("Nem sikerült képet olvasni a kamerákból.")
                break

            self._draw_points(frameFront, self.pointsFront, (0, 255, 0))
            self._draw_points(frameSide, self.pointsSide, (0, 255, 0))

            self.frontWindow.showFrame(frameFront)
            self.sideWindow.showFrame(frameSide)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('r'):
                self.pointsFront.clear()
                self.pointsSide.clear()
                print("[RESET] Mindkét kamera pontjai törölve.")
            elif key == ord('s'):
                if len(self.pointsFront) == 2 and len(self.pointsSide) == 2:
                    self._save_points()
                else:
                    print("[FIGYELEM] Nem jelöltél ki 2-2 pontot mindkét kamerán!")

        del self.camFront
        del self.camSide
        cv2.destroyAllWindows()


# --- PROGRAM INDÍTÁSA ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo kamera pontkijelölő JSON mentéssel")
    parser.add_argument("--front", type=str, default="/dev/video3", help="Első kamera indexe (pl. 0)")
    parser.add_argument("--side", type=int, default=0, help="Oldalsó kamera indexe (pl. 1)")
    parser.add_argument("--out", type=str, default="./", help="Kimeneti mappa")
    parser.add_argument("--fps", type=float, default=60.0, help="Kamera FPS")
    parser.add_argument("--width", type=int, default=1280, help="Kamera szélesség")
    parser.add_argument("--height", type=int, default=720, help="Kamera magasság")
    args = parser.parse_args()

    camFront = Camera(args.front, args.width, args.height, args.fps)

    selector = PointSelector(camFront)

    selector.run()