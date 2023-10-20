from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# cap = cv2.VideoCapture(0) # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("Yolo/Videos/cars.mp4") # For Video


model = YOLO("Yolo/Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", 
"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
"broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
"pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", 
"mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
"toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
"teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("Yolo/mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [633, 497, 1003, 487]
totalCounts = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2 - x1, y2 - y1
            

            conf = math.ceil(box.conf[0]*100) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike" and conf > 0.5:   
                # cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=5)         
                # cvzone.putTextRect(img, f"{currentClass} {conf}", ( max(0, x1), max(35, y1) ), scale=0.8, thickness=2, offset=3)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 255))
        # cvzone.putTextRect(img, f"{id}", ( max(0, x1), max(35, y1) ), scale=2, thickness=3, offset=10)

        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[3]+20:
            if id not in totalCounts:
                totalCounts.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f"Count: {len(totalCounts)}", (500, 500), scale=2, thickness=3, offset=10)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
