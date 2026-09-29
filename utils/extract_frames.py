import cv2
import os

RAW_DIR = "raw_videos"
OUT_DIR = "dataset/images"
os.makedirs(OUT_DIR, exist_ok=True)

FRAME_SKIP = 20

for video_file in os.listdir(RAW_DIR):
    if not video_file.endswith(".mp4"):
        continue

    card_name = video_file.replace(".mp4", "")

    card_folder = os.path.join(OUT_DIR, card_name)
    os.makedirs(card_folder, exist_ok=True)

    cap = cv2.VideoCapture(os.path.join(RAW_DIR, video_file))

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % FRAME_SKIP == 0:
            filename = f"{card_name}_{saved:04d}.jpg"
            cv2.imwrite(os.path.join(card_folder, filename), frame)
            saved += 1
        count += 1

    cap.release()
    print(f"{card_name}: estratti {saved} frame")