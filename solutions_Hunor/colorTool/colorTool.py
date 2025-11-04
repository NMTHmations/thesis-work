from tkinter import ttk
from tkinter import *
import cv2
import numpy as np
from tkinter import filedialog

def runOpenCV(min,max,cap):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        ret, frame = cap.read()
        if not ret:
            return
    frame = cv2.resize(frame, (640,384))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define green color range (tune these values)
    lower_yellow = np.array(min)
    upper_yellow = np.array(max)    
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_yellow , upper_yellow)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    _, thresh = cv2.threshold(mask, 85, 255, cv2.THRESH_BINARY)
    points, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("threshold",thresh)
    cv2.imshow("frame",frame)

def underWork(filename:str):
    cap = cv2.VideoCapture(filename)
    root = Tk(className="ColorTool")
    frm = ttk.Frame(root)
    minHtext = ttk.Label(root,text="Min Hue")
    minHtext.pack()
    minHscale = ttk.Scale(root,from_=0, to=255, orient=HORIZONTAL,length=500)
    minHscale.pack()
    minStext1 = ttk.Label(root,text="Min Saturation")
    minStext1.pack()
    minSscale1 = ttk.Scale(root,from_=0, to=255, orient=HORIZONTAL,length=500)
    minSscale1.pack()
    minVtext2 = ttk.Label(root,text="Min Vue")
    minVtext2.pack()
    minVscale2 = ttk.Scale(root,from_=0, to=255, orient=HORIZONTAL,length=500)
    minVscale2.pack()
    maxHtext = ttk.Label(root,text="Min Hue")
    maxHtext.pack()
    maxHscale = ttk.Scale(root,from_=0, to=255, orient=HORIZONTAL,length=500)
    maxHscale.pack()
    maxStext1 = ttk.Label(root,text="Max Saturation")
    maxStext1.pack()
    maxSscale1 = ttk.Scale(root,from_=0, to=255, orient=HORIZONTAL,length=500)
    maxSscale1.pack()
    maxVtext2 = ttk.Label(root,text="Max Vue")
    maxVtext2.pack()
    maxVscale2 = ttk.Scale(root,from_=0, to=255, orient=HORIZONTAL,length=500)
    maxVscale2.pack()
    Itext3 = ttk.Label(root,text="Iteration")
    Itext3.pack()
    Iscale3 = ttk.Scale(root,from_=0, to=16, orient=HORIZONTAL,length=500)
    Iscale3.pack()
    def getValue():
        minValue1 = int(minHscale.get())
        minValue2 = int(minSscale1.get())
        minValue3 = int(minVscale2.get())
        itValue = int(Iscale3.get())
        maxValue1 = int(maxHscale.get())
        maxValue2 = int(maxSscale1.get())
        maxValue3 = int(maxVscale2.get())
        runOpenCV([minValue1,minValue2,minValue3],[maxValue1,maxValue2,maxValue3],cap)
        print(f"min HSV: {minValue1},{minValue2},{minValue3} max HSV: {maxValue1},{maxValue2},{maxValue3} Iteration: {itValue}")
        root.after(20, getValue)

    button = ttk.Button(root,text="Get values!",command=getValue)
    button.pack()
    frm.pack()
    root.after(0, getValue)
    root.mainloop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    filepath = filedialog.askopenfilename(filetypes=(("MP4 Video files", "*.mp4"), (".MOV files", "*.mov")),title="Open File")
    underWork(filename=filepath)