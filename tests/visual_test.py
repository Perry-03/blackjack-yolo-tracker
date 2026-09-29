from ultralytics import YOLO

model = YOLO("../models/weights/best.pt")

print("=== Eseguendo il test sulle immagini ===")

results = model.predict(
    source="final_dataset/test/images",
    save=True,
    conf=0.5,
    show_conf=True
)

print("\n=== TEST COMPLETATO ===")