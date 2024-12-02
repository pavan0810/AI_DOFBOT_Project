from ultralytics import YOLO
import cv2
model = YOLO('best.pt')

image_path = 'carrot.jpg'
results = model.predict(source=image_path, save=True, save_txt=True, conf=0.5)
annotated_image_path = results[0].path  
annotated_image = cv2.imread(annotated_image_path)
annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)