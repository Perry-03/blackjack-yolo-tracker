from ultralytics import YOLO
model = YOLO("models/weights/best.pt")
model.export(format="openvino")