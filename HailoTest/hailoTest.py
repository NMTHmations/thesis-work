import traceback
import hailo_platform
import hailo_platform.pyhailort
import hailo_platform.pyhailort._pyhailort
from hailo_platform.pyhailort.pyhailort import InferModel
import numpy as np
import cv2
import os
import supervision as sv

def extract_detections(output:list, h: int, w: int, threshold: float = 0.05):
    xyxy = []
    confidence = []
    class_id = []
    num_detections = 0

    for i, detections in enumerate(output):
        if len(detections) == 0:
            continue
        for detection in detections:
            if len(detection) == 0:
                continue
            fdetection = detection[0]
            bbox, score = fdetection[:4], fdetection[4]

            print(score)

            if score < np.float32(threshold):
                continue

            bbox[0], bbox[1], bbox[2], bbox[3] = (
                bbox[1] * w,
                bbox[0] * h,
                bbox[3] * w,
                bbox[2] * h,
            ) 

            xyxy.append(bbox)
            confidence.append(score)
            class_id.append(i)
            num_detections += 1
    
    print(xyxy)
    return {
        "xyxy": np.array(xyxy),
        "confidence": np.array(confidence, dtype=np.float32),
        "class_id": np.array(class_id),
        "num_detections": num_detections
    }

def process_detections(frame: np.ndarray, detections: dict, class_names: list, tracker: sv.ByteTrack, 
                       box_annotator: sv.BoxAnnotator, label_annotator: sv.LabelAnnotator):
    if len(detections['xyxy']) == 0:
        return frame
    sv_detections = sv.Detections(xyxy=detections["xyxy"],
                                  confidence=detections["confidence"],
                                  class_id=detections["class_id"])
    
    print(sv_detections)
    
    sv_detections = tracker.update_with_detections(sv_detections)

    labels = [f"{class_names[cls]} {conf:.2f}" for cls, conf in zip(sv_detections.class_id, sv_detections.tracker_id)]

    print(labels)

    annotated_frame = box_annotator.annotate(scene=frame.copy(),detections=sv_detections)

    annotated_frame = label_annotator.annotate(scene=annotated_frame,detections=sv_detections,labels=labels)
    return annotated_frame

# Load model
hef_path = "hailort/ball-detection--640x480_quant_hailort_hailo8_1.hef"
platform = hailo_platform.HEF(hef_source=hef_path)
annotator = sv.BoxAnnotator()
label = sv.LabelAnnotator()
tracker = sv.ByteTrack(track_activation_threshold=0.25,minimum_matching_threshold=1)

with hailo_platform.VDevice() as target:
    try:
        config = hailo_platform.ConfigureParams.create_from_hef(platform, interface=hailo_platform.HailoStreamInterface)
    except:
        print("Create from_hef not found")
        config = None
    if config is not None:
        groups = target.configure(platform, config)
    else:
        groups = target.configure(platform)
    network_group = groups[0]

    try:
        ctx = network_group.activate()
    except:
        ctx = None
    
    if ctx is not None:
        with ctx:
            input_info = platform.get_input_vstream_infos()[0]
            output_info = platform.get_output_vstream_infos()[0]

            ivsp = hailo_platform.InputVStreamParams.make_from_network_group(network_group, quantized=False)
            ovsp = hailo_platform.OutputVStreamParams.make_from_network_group(network_group, quantized=False)

            os.environ["QT_QPA_PLATFORM"] = "xcb"

            cap = cv2.VideoCapture("multicam/real4.mp4")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS,120)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
            class_names = ['Ball','Football']
            if not cap.isOpened():
                print("Error happened!")
                exit(2)
            while True:
                ret, frame = cap.read()
                #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                #if not ret:
                #    print("Error with showing the image")
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                    continue
                frame = cv2.resize(frame, (640,480))
                input_data = frame.astype(np.uint8)
                input_data = np.expand_dims(input_data, 0)
                input_name = platform.get_input_vstream_infos()[0].name
                inputs = {input_name: input_data}
                try:
                    with hailo_platform.InferVStreams(network_group,ivsp,ovsp, False) as pipeline:
                        output = pipeline.infer(input_data)
                        print(output)
                        detections = extract_detections(output['best_v8/yolov8_nms_postprocess'],480,640)
                        frame = process_detections(frame,detections,class_names,tracker,annotator,label)
                        cv2.imshow("Test Image", frame)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print ("InferVStreams of pipeline infer not working")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break