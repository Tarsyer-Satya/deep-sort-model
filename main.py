import os
import random

import cv2
from ultralytics import YOLO

from tracker import Tracker
from collections import defaultdict



def find_centroid(x1,y1,x2,y2):
    return [(x1 + x2)//2, (y1 + y2)//2]


start_point = (150,420)
end_point = (1100,420)


video_path = os.path.join('.', 'data', 'people.mp4')
video_out_path = os.path.join('.', 'cars.mp4')

video_path = 'data/cars.mp4'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8n.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(20)]
count = 0
detection_threshold = 0.3


track_dict = defaultdict(list)
cars_entered = 0
cars_left = 0

while ret:

    results = model(frame)
    count += 1

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            centroid = find_centroid(x1, y1, x2, y2)

            if(len(track_dict[track_id]) != 0): 
                if(track_dict[track_id][1] < start_point[1] and start_point[1] <= centroid[1]):
                    cars_entered += 1
                if(track_dict[track_id][1] > start_point[1] and start_point[1] >= centroid[1]):
                    cars_left += 1

            track_dict[track_id] = centroid
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

        cv2.putText(frame, f"vehicles_entered: {cars_entered}", (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 3)
        cv2.putText(frame, f"vehicles_left: {cars_left}", (50,100), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 3)

            
        
    





            

        






    cap_out.write(frame)
    frame = cv2.line(frame, start_point, end_point, (0,155,0), 3)
    cv2.imshow('image',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    ret, frame = cap.read()





cap.release()
cap_out.release()
cv2.destroyAllWindows()