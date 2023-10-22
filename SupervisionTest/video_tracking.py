import argparse

from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

import cv2
import numpy as np
import json

LINE_START = sv.Point(320, 0)
LINE_END = sv.Point(320, 480)


model = YOLO('yolov8n.pt')

def main_func(input_data,frame):

    global model

    if 'ignore' in input_data:
        return "ignore"
    else:

        input_data = json.loads(input_data)

        results= model.track(frame, imgsz=640, classes=0,tracker="bytetrack.yaml")[0]
        detections = sv.Detections.from_yolov8(results)

        if results.boxes.id is not None:
            detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)
            detections.conf = results.boxes.conf.cpu().numpy().astype(float)
            detections.xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
            
            # print(detections.tracker_id)
            # print(detections.conf)
            # print(detections.xyxy)
        else:
            detections.tracker_id = np.array([])
            detections.conf = np.array([])
            detections.xyxy = np.array([])
 
        new_box=detections.xyxy
        new_box = new_box.astype("int").tolist()
        new_ids = detections.tracker_id.astype("int").tolist()
        new_confs = detections.conf.astype("float").tolist()    



        print("count",count)

        output = {
                  'frame_index':input_data['frame_index'],
                  'time_stamp':input_data['time_stamp'],
                  'detections':new_box,
                  'demo_run':input_data['demo_run'],
                  'ids':new_ids,
                  'confs':new_confs
                  }
        
        print(output)
        return output

def process_frame(input_data, frame, line_counter_1, line_counter_2, line_counter_3, line_annotator, box_annotator):

    model = YOLO("yolov8n.pt")
    results = model.track(frame, imgsz=640, classes=0, tracker="bytetrack.yaml")[0]
    detections = sv.Detections.from_yolov8(results)

    if results.boxes.id is not None:
        detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)
        detections.conf = results.boxes.conf.cpu().numpy().astype(float)
        detections.xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
    else:
        detections.tracker_id = np.array([])
        detections.conf = np.array([])
        detections.xyxy = np.array([])

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=[str(id) for id in detections.tracker_id])
    line_counter_1.trigger(detections=detections)
    line_counter_2.trigger(detections=detections)
    line_counter_3.trigger(detections=detections)

    line_annotator.annotate(frame=frame, line_counter=line_counter_1)
    line_annotator.annotate(frame=frame, line_counter=line_counter_2)
    line_annotator.annotate(frame=frame, line_counter=line_counter_3)

    new_ids = detections.tracker_id.astype("int").tolist()
    new_box = detections.xyxy.astype("int").tolist()
    new_confs = detections.conf.astype("float").tolist()

    output = {
        'frame_index': input_data['frame_index'],
        'time_stamp': input_data['time_stamp'],
        'detections': new_box,
        'demo_run': input_data['demo_run'],
        'ids': new_ids,
        'confs': new_confs,
        'line_1_out_count': line_counter_1.out_count,
        'line_2_out_count': line_counter_2.out_count,
        'line_3_out_count': line_counter_3.out_count
    }

    print(output)
    return output




def display_frame(output, frame):
    cv2.imshow("Output", frame)
    cv2.waitKey(1)


# Provide the path to your video file
video_path = 'D:\ATP\Emaar_VIP\Data\emaar__41_50.mp4'
main_func(video_path)