import cv2
import numpy as np
import time


config_file = "yolov3.cfg"  
weights_file = "yolov3.weights"  
classes_file = "coco.names" 

# Load YOLO model
try:
    net = cv2.dnn.readNet(weights_file, config_file)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


with open(classes_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]


people_count_threshold = 4  
time_threshold = 120 
alert = False
start_time = None

def detect_people(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0: 
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return len(indexes)

def process_video(video_path):
    global alert, start_time

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        num_people = detect_people(frame)

        if num_people > people_count_threshold:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time > time_threshold:
                alert = True
                cv2.putText(frame, "ALERT! Too many people!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            start_time = None
            alert = False

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("crowd_sample.mp4") 
