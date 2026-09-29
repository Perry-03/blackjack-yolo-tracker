from ultralytics import YOLO
import torch


def main():
    print("GPU available:", torch.cuda.is_available())

    model = YOLO("runs/detect/runs/blackjack_v2-3/weights/best.pt")

    model.train(
        data="final_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        patience=15,
        project="blackjack_final",
        name="final_model"
    )


if __name__ == "__main__":
    main()