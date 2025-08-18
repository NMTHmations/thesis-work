import cv2
import supervision as sv
from inference.models.utils import get_model

source = "../sources/vid/ballpassing2.mp4"
video_info = sv.VideoInfo.from_video_path(source)

def fill_undetected(detections : list):
    for i in range(len(detections)):
        if detections[i].xyxy.size == 0:
            if detections[i-1].size != 0 and detections[i+1].size != 0:
                #prevLength, prevHeight = calculate_len_height(detections[i-1].xyxy)

                detections[i].xyxy = detections[i-1].xyxy



def main():
    model = get_model("experiment-sxxxi/1", api_key="PlEVRUdW9e6KwDkUHIX6")
    #model = get_model("detect-and-classify-object-detection-h3ecz/1", api_key="PlEVRUdW9e6KwDkUHIX6")

    boxAnnotator = sv.BoxAnnotator()
    labelAnnotator = sv.LabelAnnotator()
    traceAnnotator = sv.TraceAnnotator()
    tracker = sv.ByteTrack()

    frames = sv.get_video_frames_generator(source)
    #plt.plot(next(frames))


    allDetecions = list()

    for frame in frames:
        results = model.infer(frame)[0]
        detection = sv.Detections.from_inference(results)
        detection = detection[detection.confidence > 0.7]
        #detection = tracker.update_with_detections(detection)

        allDetecions.append(detection)

        labels = [f"#{class_id}" for class_id in detection.class_id]

        annotated = boxAnnotator.annotate(frame,detection)
        annotated = labelAnnotator.annotate(annotated,detection,labels)


        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 1280, 720)
        cv2.imshow("frame", annotated)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #fill_undetected(allDetecions)

    cv2.destroyAllWindows()




if __name__ == '__main__':
    print(video_info)
    main()