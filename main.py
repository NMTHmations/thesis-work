from multicam import ParallelTools
from multicam import getFrames
from multicam import cameraTools
import sys
import argparse
import traceback
import json

def processJson(path: str):
    frontStrike = {}
    dexterStrike = {}
    with open(path,"r") as file:
        content = file.read()
        json_content = dict(json.loads(content))
        compulsory_elems = ["height","width","acceptStartF","acceptEndF","acceptStartS","acceptEndS"]
        redirecting_elems = {
            "acceptStartF": "acceptStart",
            "acceptStartS": "acceptStart",
            "acceptEndF": "acceptEnd",
            "acceptEndS": "acceptEnd",
            "startS": "start",
            "startF": "start",
            "endS": "end",
            "endF": "end"
        }
        for celems in compulsory_elems:
            if celems in json_content:
                if celems == "acceptStartF" or celems == "acceptEndF":
                    frontStrike[redirecting_elems[celems]] = json_content[celems]
                elif celems == "acceptStartD" or celems == "acceptEndD":
                    dexterStrike[redirecting_elems[celems]] = json_content[celems]
                else:
                    frontStrike[celems] = json_content[celems]
                    dexterStrike[celems] = json_content[celems]
            else:
                raise Exception(f"The following element is missing from the {path} file: {celems}")
        optional_elems = ["lowerHSV","upperHSV","debug","startF","startS","endS","endF"]
        for oelems in optional_elems:
            if oelems in json_content:
                if oelems == "startS" or oelems == "endS":
                    dexterStrike[redirecting_elems[celems]] = json_content[celems]
                elif oelems == "startF" or oelems == "endF":
                    frontStrike[redirecting_elems[celems]] = json_content[celems]
                else:
                    frontStrike[celems] = json_content[celems]
                    dexterStrike[celems] = json_content[celems]
        if "upperHSV" in frontStrike and "lowerHSV" in frontStrike:
            frontStrike["debug"] = True
            dexterStrike["debug"] = True
        stepCount = json_content["maxSteps"] if "maxSteps" in json_content else 30
        albument = json_content["albument"] if "maxSteps" in json_content else False
        file.close()
        return frontStrike, dexterStrike, stepCount, albument
    return None, None, None


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
    camDetection.add_argument('--file',required=False,help="Get arguments from file (if you give other args, which is not in the file, then it will be ignored)")

    args = parser.parse_args()

    try:

        if args.command == "CamTools":
            cameraTools.CameraTools()
        elif args.command == "getFrames":
            getFrames.GetFrames(args.src,args.dst).getFrames()
        elif args.command == "startDetection":
            maxStep = args.maxStep
            albument = args.albument
            startingStep = round(args.maxStep / 2)
            if args.file != None:
                frontStrike, dexterStrike, maxStep, albument = processJson(args.file)
                startingStep = round(maxStep / 2)
            else:
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

            executer = ParallelTools.ParallelTools("/dev/video0","/dev/video1",frontStrike,dexterStrike,startingStep,maxStep,albument)
            executer.CameraHandler()
    except Exception as e:
        traceback.print_exc()
        exit(1)       

