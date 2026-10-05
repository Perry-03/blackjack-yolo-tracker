from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("models/weights/best.pt")

    metrics = model.val(data="dataset/data.yaml", split="test")
