import os
import cv2
import csv
import mediapipe as mp

# -------- Settings --------
DATA_DIR = "data/raw_images"
OUTPUT_CSV = "data/landmarks.csv"
GESTURES = ["thumbs_up", "peace", "open_palm", "fist"]

# -------- MediaPipe setup --------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,   # for image-by-image processing
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -------- Build CSV header --------
header = []
for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])
header.append("label")

saved_rows = 0
skipped_images = 0

with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for gesture in GESTURES:
        gesture_folder = os.path.join(DATA_DIR, gesture)

        if not os.path.exists(gesture_folder):
            print(f"Folder not found: {gesture_folder}")
            continue

        image_files = [
            file for file in os.listdir(gesture_folder)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"\nProcessing gesture: {gesture}")
        print(f"Found {len(image_files)} images")

        for image_file in image_files:
            image_path = os.path.join(gesture_folder, image_file)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read: {image_path}")
                skipped_images += 1
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            if not results.multi_hand_landmarks:
                print(f"No hand detected: {image_path}")
                skipped_images += 1
                continue

            hand_landmarks = results.multi_hand_landmarks[0]

            row = []
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])

            row.append(gesture)
            writer.writerow(row)
            saved_rows += 1

print("\nDone!")
print(f"Saved rows: {saved_rows}")
print(f"Skipped images: {skipped_images}")
print(f"CSV saved to: {OUTPUT_CSV}")