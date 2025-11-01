#!/usr/bin/env python3
import cv2
import argparse
import os
import json

from project.detection.final.finalMain import Camera, Window



# --- PROGRAM INDÍTÁSA ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo kamera pontkijelölő JSON mentéssel")
    parser.add_argument("--camera", type=int, default=0, help="Első kamera indexe (pl. 0)")
    parser.add_argument("--out", type=str, default="./", help="Kimeneti mappa")
    parser.add_argument("--fps", type=float, default=60.0, help="Kamera FPS")
    parser.add_argument("--width", type=int, default=1280, help="Kamera szélesség")
    parser.add_argument("--height", type=int, default=720, help="Kamera magasság")
    args = parser.parse_args()

