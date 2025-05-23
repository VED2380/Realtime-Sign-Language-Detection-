#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import time

# Initialize TTS Engine
tts_engine = pyttsx3.init()

# Load the trained model
MODEL_PATH = r"D:\code\proj\trained_model\sign_language_cnn_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ASL labels (J and Z excluded)
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def preprocess_hand_image(hand_roi):
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    inverted = cv2.bitwise_not(resized)
    normalized = inverted.astype("float32") / 255.0
    return normalized.reshape(1, 28, 28, 1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam feed for sign language recognition. Press 'q' to quit.")
    current_letter = ""
    last_spoken_letter = ""
    letter_buffer = []
    buffer_size = 10
    word_buffer = []
    last_gesture_time = time.time()
    word_formation_delay = 2

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        detected_letter_this_frame = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                padding = 40
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                if x_max > x_min and y_max > y_min:
                    hand_roi = frame[y_min:y_max, x_min:x_max]
                    if hand_roi.size == 0:
                        continue

                    processed = preprocess_hand_image(hand_roi)
                    prediction = model.predict(processed, verbose=0)
                    print(f"Prediction: {prediction}")  # Debug output
                    predicted_index = np.argmax(prediction)
                    confidence = np.max(prediction)

                    if confidence > 0.6:
                        if predicted_index in label_map:
                            detected_letter_this_frame = label_map[predicted_index]
                        else:
                            detected_letter_this_frame = ""
                    else:
                        detected_letter_this_frame = ""

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"{detected_letter_this_frame} ({confidence:.2f})", 
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                (0, 255, 0), 2, cv2.LINE_AA)
            last_gesture_time = time.time()
        else:
            if time.time() - last_gesture_time > word_formation_delay and word_buffer and word_buffer[-1] != " ":
                word_buffer.append(" ")
                last_spoken_letter = ""
            detected_letter_this_frame = ""

        letter_buffer.append(detected_letter_this_frame)
        if len(letter_buffer) > buffer_size:
            letter_buffer.pop(0)

        if letter_buffer:
            filtered = [l for l in letter_buffer if l]
            if filtered:
                stable_letter = max(set(filtered), key=filtered.count)
                if stable_letter != current_letter:
                    current_letter = stable_letter
                    if current_letter != last_spoken_letter:
                        word_buffer.append(current_letter)
                        print(f"Recognized: {current_letter}")
                        tts_engine.say(current_letter)
                        tts_engine.runAndWait()
                        last_spoken_letter = current_letter
            elif not any(letter_buffer):
                current_letter = ""

        cv2.putText(frame, f"Text: {''.join(word_buffer)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Current: {current_letter}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Webcam feed stopped.")

if __name__ == '__main__':
    main()
