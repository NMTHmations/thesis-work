#!/usr/bin/env python3
import cv2
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

    def _mouse_cb(self, event, x, y, p, t):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"Pont hozzáadva: {(x, y)}")
            else:
                print("Már két pont van, nyomj 'r'-t a resethez.")

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

        print(f"[INFO] Pontok elmentve JSON formátumban:\n - {fpath}")

    # --- fő futtató függvény ---
    def run(self):
        print("Bal egérgombbal jelölj 2-2 pontot mindkét kameraképen.")
        print("Nyomj 's'-t a mentéshez, 'r'-t a törléshez, 'q' vagy ESC a kilépéshez.")

        while True:

            success, frame = self.camera.capture()

            if not success:
                print("Nem sikerült képet olvasni a kamerából.")
                break

            self._draw_points(frame, self.points, (0, 255, 0))

            self.window.showFrame(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('r'):
                self.points.clear()
                print("[RESET] Pontok törölve.")
            elif key == ord('s'):
                if len(self.points) == 2:
                    self._save_points()
                else:
                    print("[FIGYELEM] Nem jelöltél ki 2 pontot mindkét kamerán!")

        del self.camera
        del self.window


# --- PROGRAM INDÍTÁSA ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo kamera pontkijelölő JSON mentéssel")
    parser.add_argument("--camera", type=str, default="/dev/video3", help="Első kamera indexe (pl. 0)")
    parser.add_argument("--out", type=str, default="./", help="Kimeneti mappa")
    parser.add_argument("--fps", type=float, default=60.0, help="Kamera FPS")
    parser.add_argument("--width", type=int, default=1280, help="Kamera szélesség")
    parser.add_argument("--height", type=int, default=720, help="Kamera magasság")
    args = parser.parse_args()

    camFront = Camera(args.camera, args.width, args.height, args.fps)

    selector = PointSelector(camera=camFront,outputDir=args.out)
    selector.run()
