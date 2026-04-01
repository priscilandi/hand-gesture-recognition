import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import Counter, deque

# -------- Load trained model --------
MODEL_PATH = "models/gesture_model.pkl"
model = joblib.load(MODEL_PATH)

# -------- Prediction smoothing setup --------
prediction_history = deque(maxlen=10)

# -------- MediaPipe setup --------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_smoothed_prediction(history):
    if not history:
        return None
    return Counter(history).most_common(1)[0][0]

# -------- Webcam --------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    prediction_text = "No hand detected"
    confidence_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # -------- Extract features --------
            row = []
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])

            # Convert to numpy array
            input_data = np.array(row).reshape(1, -1)

            # -------- Predict --------
            prediction = model.predict(input_data)[0]
            prediction_history.append(prediction)
            smoothed_prediction = get_smoothed_prediction(prediction_history)

            probabilities = model.predict_proba(input_data)[0]
            confidence = np.max(probabilities)

            prediction_text = f"Gesture: {smoothed_prediction}"
            confidence_text = f"Confidence: {confidence:.2f}"

    cv2.putText(
        frame,
        prediction_text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        confidence_text,
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    if not results.multi_hand_landmarks:
        prediction_history.clear()
    
    cv2.imshow("Live Gesture Recognition", frame)
    
    key = cv2.waitKey(10)
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()