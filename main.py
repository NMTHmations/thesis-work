from multicam import ParallelTools
from multicam import getFrames
from multicam import cameraTools
import argparse
import traceback
import json

def processJson(path: str):
    frontStrike = {}
    dexterStrike = {}
    with open(path,"r") as file:
        content = file.read()
        json_content = dict(json.loads(content))
        elems = ["acceptStartF","acceptEndF","acceptStartS","acceptEndS","lowerHSV","upperHSV","debug"]
        redirecting_elems = {
            "acceptStartF": "acceptStart",
            "acceptStartS": "acceptStart",
            "acceptEndF": "acceptEnd",
            "acceptEndS": "acceptEnd"
        }
        for index, celems in enumerate(elems):
            if celems in json_content:
                if celems == "acceptStartF" or celems == "acceptEndF":
                    frontStrike[redirecting_elems[celems]] = json_content[celems]
                elif celems == "acceptStartD" or celems == "acceptEndD":
                    dexterStrike[redirecting_elems[celems]] = json_content[celems]
                else:
                    frontStrike[celems] = json_content[celems]
                    dexterStrike[celems] = json_content[celems]
            else:
                if index < 4: 
                    raise Exception(f"The following element is missing from the {path} file: {celems}")
        if "upperHSV" in frontStrike and "lowerHSV" in frontStrike:
            frontStrike["debug"] = True
            dexterStrike["debug"] = True
        startStep = json_content["startStep"] if "startStep" in json_content else 0
        endStep = json_content["endStep"] if "endStep" in json_content else 600
        albument = json_content["albument"] if "albument" in json_content else False
        frontStrike["isFront"] = True
        file.close()
        return frontStrike, dexterStrike, startStep, endStep, albument
    return None, None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robotic Football - Goal Keeper Automat")
    subparsers = parser.add_subparsers(dest="command", required=True)
    frameGetter = subparsers.add_parser('getFrames', help='get Frames for Robflow annotation')
    frameGetter.add_argument('--src',type=str,required=True,help="The source file of the video (must be .mp4/.mov)")
    frameGetter.add_argument('--dst',type=str,required=True,help="The destination library of the frames")
    camTools = subparsers.add_parser('CamTools', help='Shows camera settings under Raspberry Pi')
    camDetection = subparsers.add_parser('startDetection', help='Starts detection')
    camDetection.add_argument('--lowerHSV',type=str,required=False,help="Sets lower HSV")
    camDetection.add_argument('--upperHSV',type=str,required=False,help="Sets upper HSV")
    camDetection.add_argument('--albument',required=False,help="Albument camera image",action='store_false')
    camDetection.add_argument('--file',required=False,help="Get arguments from file (if you give other args, which is not in the file, then it will be ignored)")

    args = parser.parse_args()

    try:

        if args.command == "CamTools":
            cameraTools.CameraTools()
        elif args.command == "getFrames":
            getFrames.GetFrames(args.src,args.dst).getFrames()
        elif args.command == "startDetection":
            albument = args.albument
            startStep = 0
            endStep = 600
            if args.file != None:
                frontStrike, dexterStrike, startStep, stopStep, albument = processJson(args.file)
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
                    "acceptStart": (150, 0),
                    "acceptEnd": (150, 480),
                    "lowerHSV": lowerHSV,
                    "upperHSV": upperHSV,
                    "debug": debug
                }

                frontStrike = {
                    "acceptStart": (0, 480),
                    "acceptEnd": (640, 480),
                    "lowerHSV": lowerHSV,
                    "upperHSV": upperHSV,
                    "debug": debug,
                    "isFront": True
                }

            executer = ParallelTools.ParallelTools("/dev/video0","/dev/video2",frontStrike,dexterStrike,startStep,endStep,albument)
            executer.CameraHandler()
    except Exception as e:
        traceback.print_exc()
        exit(1)       

