import cv2
import os
import sys

class GetFrames:
    def __init__(self,folderPath,saveFolder):
        self.folderPath = folderPath
        self.saveFolder = saveFolder
    
    def getFrames(self):
        os.chdir(self.folderPath)
        scanned = os.scandir()
        for x in scanned:
            if x.is_file:
                if x.name.endswith(".mp4") or x.name.endswith(".mov"):
                    cap = cv2.VideoCapture(x.path)
                    if not cap.isOpened():
                        print("Error: Could not open video.")
                        exit()
                    frame_count = 1
                    while True: 
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if os.path.exists(self.saveFolder) == False:
                            os.mkdir(os.path.join(self.saveFolder))
                        newPath = os.path.join(self.saveFolder,x.name.strip(".mp4").strip(".mov"))
                        if os.path.exists(newPath) == False:
                            os.mkdir(newPath)
                        cv2.imwrite(os.path.join(self.saveFolder,x.name.strip(".mp4").strip(".mov"),str(frame_count) + ".png"),frame)
                        frame_count += 1
            elif x.is_dir:
                newFolder = os.path.join(self.saveFolder,x.name)
                if os.path.exists(newFolder) == False:
                    os.mkdir(newFolder)
                x = GetFrames(x.path,newFolder).getFrames()
            else:
                sys.stderr.write("Could not retrieve any data on this directory")