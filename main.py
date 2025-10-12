from multicam import ParallelTools
from multicam import getFrames
from multicam import cameraTools
import sys
import argparse
import traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robotic Football - Goal Keeper Automat")
    subparsers = parser.add_subparsers(dest="command", required=True)
    camTools = subparsers.add_parser('CamTools', help='Shows camera settings under Raspberry Pi')
    frameGetter = subparsers.add_parser('getFrames', help='get Frames for Robflow annotation')
    frameGetter.add_argument('--src',type=str,required=True,help="The source file of the video (must be .mp4/.mov)")
    frameGetter.add_argument('--dst',type=str,required=True,help="The destination library of the frames")
    camDetection = subparsers.add_parser('startDetection', help='Starts detection')
    camDetection.add_argument('--lowerHSV',type=str,required=False,help="Sets lower HSV")
    camDetection.add_argument('--upperHSV',type=str,required=False,help="Sets upper HSV")
    camDetection.add_argument('--maxStep',type=int,required=False,help="Sets maximum steps of the motor (default 30)",default=30)
    camDetection.add_argument('--albument',required=False,help="Albument camera image",action='store_false')

    args = parser.parse_args()

    try:

        if args.command == "CamTools":
            cameraTools.CameraTools()
        elif args.command == "getFrames":
            getFrames.GetFrames(args.src,args.dst).getFrames()
        elif args.command == "startDetection":
            lowerHSV = None
            upperHSV = None
            if args.lowerHSV != None and args.upperHSV != None:
                if len(args.lowerHSV.split(",")) == 3 and len(args.upperHSV.split(",")) == 3:
                    lowerHSV =  [int(args.lowerHSV.split(',')[0]),int(args.lowerHSV.split(',')[1]),int(args.lowerHSV.split(',')[2])]
                    upperHSV =  [int(args.upperHSV.split(',')[0]),int(args.upperHSV.split(',')[1]),int(args.upperHSV.split(',')[2])]
                else:
                    raise Exception("Invalid HSV format")
            debug = lowerHSV != None and upperHSV != None
            dexterStrike = {
                "start": 50,
                "end": 540,
                "height": 480,
                "width": 640,
                "acceptStart": (150, 0),
                "acceptEnd": (150, 480),
                "lowerHSV": lowerHSV,
                "upperHSV": upperHSV,
                "debug": debug
            }

            frontStrike = {
                "start": 50,
                "end": 540,
                "height": 480,
                "width": 640,
                "acceptStart": (0, 480),
                "acceptEnd": (640, 480),
                "lowerHSV": lowerHSV,
                "upperHSV": upperHSV,
                "debug": debug
            }
            startingStep = round(args.maxStep / 2)
            executer = ParallelTools.ParallelTools("dev/video2","dev/video0",frontStrike,dexterStrike,startingStep,args.maxStep,args.albument)
            executer.CameraHandler()
    except Exception as e:
        traceback.print_exc()
        exit(1)       

