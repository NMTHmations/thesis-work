import hailo_platform
import hailo_platform.pyhailort
import hailo_platform.pyhailort._pyhailort
from hailo_platform.pyhailort.pyhailort import InferModel
import numpy as np
import cv2
import os


# Load model
hef_path = "hailort/ball-detection--640x480_quant_hailort_hailo8_1.hef"
platform = hailo_platform.HEF(hef_source=hef_path)

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
            #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
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
                    with hailo_platform.InferVStreams(network_group,ivsp,ovsp) as pipeline:
                        output = pipeline.infer(input_data)
                except:
                    print ("InferVStreams of pipeline infer not working")
                cv2.imshow("Test Image", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break