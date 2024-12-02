from ultralytics import YOLO
import cv2

model = YOLO("uno100_2.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()  
    if not ret:
        break
    results = model.predict(source=frame)
    for result in results:
        for box in result.boxes:
            x1, y1,x2,y2 = map(int, box.xyxy[0]) 
            conf = box.conf[0] 
            cls = box.cls[0]    
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Code to train the model
# model = YOLO('yolov8m.pt')

# model.train(
#     data=f'{dataset.location}/data.yaml',
#     epochs=100,
#     batch=14,
#     imgsz=640,
#     lr0=0.001,
#     lrf=0.2,
#     flipud=1,
#     fliplr=1,
#     perspective=0.0005,
#     degrees=36
# )