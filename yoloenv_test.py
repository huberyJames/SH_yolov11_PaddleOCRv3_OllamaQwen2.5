from ultralytics import YOLO
import os

BASE_MODEL_DIR = r"C:\Users\Administrator\Desktop\sh\models"
YOLO_MODEL_DIR = os.path.join(BASE_MODEL_DIR, "yolo")

IMG_PATH = r"C:\Users\Administrator\Desktop\sh\pic\car.jpg"  

model = YOLO(os.path.join(YOLO_MODEL_DIR, "yolo11n.pt"))

model.predict(source=IMG_PATH, save=True)