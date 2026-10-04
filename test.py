from ultralytics import YOLO

model = YOLO("models/weights/best.pt")

metrics = model.val(data="final_dataset/data.yaml", split="test")