#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt

# Define paths
DATA_DIR = r"D:\code\proj\preprocessed_data"
MODEL_SAVE_DIR = r"D:\code\proj\trained_model"
MODEL_FILE = os.path.join(MODEL_SAVE_DIR, "sign_language_cnn_model.keras")
HISTORY_FILE = os.path.join(MODEL_SAVE_DIR, "training_history.npy")
PLOT_FILE = os.path.join(MODEL_SAVE_DIR, "training_history.png")

# Create model save directory if it doesn't exist
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

print(f"Model will be saved in: {MODEL_SAVE_DIR}")

# Load preprocessed data
print("Loading preprocessed data...")
try:
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print("Data loaded successfully.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
except FileNotFoundError as e:
    print(f"Error loading data files: {e}. Please ensure preprocessing was successful and files exist.")
    exit()

# Define the model architecture
num_classes = 25  # 0-24 labels

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
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    MODEL_FILE, 
    save_best_only=True, 
    monitor='val_accuracy',
    save_weights_only=True
)
EPOCHS = 50
BATCH_SIZE = 128

# Train the model
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint])

# Save the full model after training to avoid options error
model.save(MODEL_FILE)
print(f"Full model saved to {MODEL_FILE}")

# Save training history
np.save(HISTORY_FILE, history.history)
print(f"Training history saved to {HISTORY_FILE}")

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
plt.savefig(PLOT_FILE)
print(f"Training history plot saved to {PLOT_FILE}")

# Evaluate the model on the test set
print("Evaluating model on test data...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc*100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

print("Script finished.")
