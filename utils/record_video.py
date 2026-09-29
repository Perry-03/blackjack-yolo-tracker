import cv2
import os
import time

OUTPUT_DIR = "raw_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

card_name = input("Card name (e.g. 8S): ").strip()
output_path = os.path.join(OUTPUT_DIR, f"{card_name}.mp4")

cap = cv2.VideoCapture(0)

# Abbasso la risoluzione nativa della webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 20

# Imposto il file video di output esattamente a 640x640
target_width = 640
target_height = 640
out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

print("Press 'r' for start/end the video recording, 'q' for exit")
recording = False
start_time = None
elapsed_accumulated = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Forzo il ridimensionamento
    frame = cv2.resize(frame, (target_width, target_height))
    
    display = frame.copy()
    if recording:
        out.write(frame)
        cv2.putText(display, "REC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        current_elapsed = elapsed_accumulated + (time.time() - start_time)
        timer_text = f"{current_elapsed:.1f}s"
        cv2.putText(display, timer_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Recording card", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        if not recording:
            start_time = time.time()
        else:
            elapsed_accumulated += time.time() - start_time
        recording = not recording
    elif key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved in: {output_path}")
print(f"Total recording duration: {elapsed_accumulated:.1f}s")