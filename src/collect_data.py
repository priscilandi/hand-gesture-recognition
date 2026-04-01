import cv2
import os
import time

# Change this to the gesture to collect
GESTURE_NAME = "open_palm"

SAVE_DIR = f"data/raw_images/{GESTURE_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

print("Camera opened.")
print("Press 's' to save an image.")
print("Press 'q' to quit.")

img_count = len(os.listdir(SAVE_DIR))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame")
        break

    frame = cv2.flip(frame, 1)

    display_text = f"Gesture: {GESTURE_NAME} | Saved: {img_count}"
    cv2.putText(
        frame,
        display_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = os.path.join(SAVE_DIR, f"{GESTURE_NAME}_{int(time.time() * 1000)}.jpg")
        cv2.imwrite(filename, frame)
        img_count += 1
        print(f"Saved: {filename}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()