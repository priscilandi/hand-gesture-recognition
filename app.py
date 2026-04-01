import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np
from PIL import Image
from collections import Counter, deque

st.set_page_config(page_title="Gesture Recognition App", page_icon="🖐️")
st.title("🖐️ Hand Gesture Recognition")
st.write("Take a photo or upload an image to predict the hand gesture.")

MODEL_PATH = "models/gesture_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    return mp_hands, hands

model = load_model()
mp_hands, hands = load_hands()

input_method = st.radio("Choose input method:", ["Camera", "Upload Image"])

image = None

if input_method == "Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image is not None:
    image = image.convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Input image", use_container_width=True)

    results = hands.process(image_np)

    if not results.multi_hand_landmarks:
        st.error("No hand detected in this image.")
    else:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks on a copy
        annotated = image_np.copy()
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            annotated,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Extract features
        row = []
        for landmark in hand_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])

        input_data = np.array(row).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(np.max(probabilities))

        st.subheader("Prediction")
        st.success(f"Gesture: {prediction}")
        st.info(f"Confidence: {confidence:.2f}")

        st.subheader("Annotated image")
        st.image(annotated, caption="Detected hand landmarks", use_container_width=True)