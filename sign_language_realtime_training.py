#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Create directories for saving data and models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "collected_data")
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Define the labels for ASL letters (excluding J and Z which require motion)
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
NUM_CLASSES = len(LABELS)

# Create a mapping from label to index
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}

def preprocess_hand_image(hand_roi):
    """Preprocesses the hand region of interest for model training/prediction."""
    # Convert to grayscale
    gray_hand = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28 (same as MNIST)
    resized_hand = cv2.resize(gray_hand, (28, 28))
    # Normalize pixel values
    normalized_hand = resized_hand.astype("float32") / 255.0
    # Reshape for model input
    processed_image = normalized_hand.reshape(28, 28, 1)
    return processed_image

def collect_data():
    """Collect and label hand gesture data from webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("\nData Collection Mode")
    print("--------------------")
    print("Press a letter key (A-Y, excluding J and Z) to start collecting samples for that letter.")
    print("Press SPACE to capture a sample when the letter is selected.")
    print("Press ESC to exit data collection mode.")
    
    current_label = None
    samples_collected = {label: 0 for label in LABELS}
    total_samples = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract bounding box for the hand
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding to the bounding box
                padding = 30
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Display instructions and current label
        cv2.putText(frame, f"Current Label: {current_label if current_label else 'None'}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Samples: {samples_collected.get(current_label, 0) if current_label else 0}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Press letter key (A-Y) to select label", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "Press SPACE to capture sample", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "Press ESC to exit", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(5) & 0xFF
        
        # Check for ESC key
        if key == 27:  # ESC key
            break
        
        # Check for letter keys (A-Y, excluding J and Z)
        if 97 <= key <= 121:  # ASCII for 'a' to 'y'
            letter = chr(key).upper()
            if letter != 'J' and letter != 'Z' and letter in LABELS:
                current_label = letter
                print(f"Selected label: {current_label}")
        
        # Check for SPACE key to capture sample
        if key == 32 and current_label and results.multi_hand_landmarks:  # SPACE key
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding
                padding = 30
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                if x_max > x_min and y_max > y_min:
                    hand_roi = frame[y_min:y_max, x_min:x_max]
                    if hand_roi.size == 0:
                        continue
                    
                    # Preprocess the hand image
                    processed_hand = preprocess_hand_image(hand_roi)
                    
                    # Save the processed image
                    label_dir = os.path.join(DATA_DIR, current_label)
                    os.makedirs(label_dir, exist_ok=True)
                    
                    sample_count = samples_collected.get(current_label, 0)
                    sample_path = os.path.join(label_dir, f"{current_label}_{sample_count}.npy")
                    np.save(sample_path, processed_hand)
                    
                    samples_collected[current_label] = sample_count + 1
                    total_samples += 1
                    
                    print(f"Saved sample {sample_count + 1} for label {current_label}. Total samples: {total_samples}")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print("\nData Collection Summary:")
    print("------------------------")
    for label, count in samples_collected.items():
        if count > 0:
            print(f"Label {label}: {count} samples")
    print(f"Total samples collected: {total_samples}")
    
    return samples_collected

def load_collected_data():
    """Load the collected data for training."""
    X = []
    y = []
    
    for label in LABELS:
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.exists(label_dir):
            continue
        
        for filename in os.listdir(label_dir):
            if filename.endswith('.npy'):
                sample_path = os.path.join(label_dir, filename)
                sample = np.load(sample_path)
                X.append(sample)
                y.append(LABEL_TO_INDEX[label])
    
    if not X:
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def train_model():
    """Train a CNN model on the collected data."""
    # Load the collected data
    X, y = load_collected_data()
    
    if X is None or len(X) < NUM_CLASSES:
        print("Error: Insufficient data for training. Please collect more samples.")
        return None
    
    print(f"\nTraining model on {len(X)} samples...")
    
    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Define the model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )
    
    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"\nValidation accuracy: {val_acc*100:.2f}%")
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, "sign_language_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(MODEL_DIR, "training_history.png")
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    
    return model

def real_time_recognition(use_tts=False):
    """Perform real-time sign language recognition using the trained model."""
    # Load the trained model
    model_path = os.path.join(MODEL_DIR, "sign_language_model.keras")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize TTS if requested
    if use_tts:
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            print("Text-to-speech initialized.")
        except Exception as e:
            print(f"Error initializing text-to-speech: {e}")
            use_tts = False
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("\nReal-time Recognition Mode")
    print("-------------------------")
    print("Press 'q' to quit.")
    
    current_letter = ""
    letter_buffer = []
    buffer_size = 10  # Number of frames to average prediction over
    word_buffer = []
    last_spoken_letter = ""
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        detected_letter_this_frame = ""
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract bounding box for the hand
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding
                padding = 30
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                if x_max > x_min and y_max > y_min:
                    hand_roi = frame[y_min:y_max, x_min:x_max]
                    if hand_roi.size == 0:
                        continue
                    
                    # Preprocess the hand image
                    processed_hand = preprocess_hand_image(hand_roi)
                    
                    # Make prediction
                    prediction = model.predict(np.expand_dims(processed_hand, axis=0), verbose=0)
                    predicted_class_index = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    if confidence > 0.7:  # Confidence threshold
                        predicted_letter = LABELS[predicted_class_index]
                        detected_letter_this_frame = predicted_letter
                    else:
                        detected_letter_this_frame = ""
                    
                    # Display the prediction
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"Prediction: {detected_letter_this_frame} ({confidence:.2f})", 
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update letter buffer and current recognized letter
        letter_buffer.append(detected_letter_this_frame)
        if len(letter_buffer) > buffer_size:
            letter_buffer.pop(0)
        
        if letter_buffer:
            # Find the most frequent non-empty letter in the buffer
            filtered_buffer = [l for l in letter_buffer if l]
            if filtered_buffer:
                stable_letter = max(set(filtered_buffer), key=filtered_buffer.count)
                if stable_letter != current_letter:
                    current_letter = stable_letter
                    if current_letter and current_letter != last_spoken_letter:
                        word_buffer.append(current_letter)
                        print(f"Recognized: {current_letter}")
                        
                        # Speak the letter if TTS is enabled
                        if use_tts:
                            tts_engine.say(current_letter)
                            tts_engine.runAndWait()
                        
                        last_spoken_letter = current_letter
            elif not any(letter_buffer):  # If buffer is all empty strings
                current_letter = ""
        
        # Display the current recognized word/sequence
        word_text = "".join(word_buffer)
        cv2.putText(frame, f"Text: {word_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Current: {current_letter}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add a clear text button
        cv2.rectangle(frame, (10, 100), (150, 140), (200, 200, 200), -1)
        cv2.putText(frame, "Clear Text", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.imshow('Sign Language Recognition', frame)
        
        key = cv2.waitKey(5) & 0xFF
        
        # Check for 'q' key to quit
        if key == ord('q'):
            break
        
        # Check for mouse click to clear text
        # (This doesn't work directly in OpenCV, would need a callback function)
        # For simplicity, we'll use the 'c' key to clear text
        if key == ord('c'):
            word_buffer = []
            print("Text cleared")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Recognition stopped.")

def main():
    """Main function to run the application."""
    print("\nReal-Time Sign Language Recognition")
    print("==================================")
    print("1. Collect training data")
    print("2. Train model")
    print("3. Run real-time recognition")
    print("4. Run real-time recognition with text-to-speech")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            collect_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            real_time_recognition(use_tts=False)
        elif choice == '4':
            real_time_recognition(use_tts=True)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
