# Real-Time Sign Language Detection

A computer vision application that recognizes American Sign Language (ASL) gestures in real-time using a webcam, converting hand gestures to text with optional text-to-speech output.


## Features

- **Real-time hand tracking** using MediaPipe
- **Custom data collection** tool to create your own training dataset
- **CNN-based gesture recognition** trained on your own hand gestures
- **Real-time text conversion** from recognized gestures
- **Text-to-speech output** for audible feedback (optional)
- **Cross-platform compatibility** (Windows, macOS, Linux)

## Requirements

- Python 3.8+
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/realtime-sign-language-recognition.git
   cd realtime-sign-language-recognition
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python sign_language_realtime_training.py
```

The application provides four main functionalities:

### 1. Data Collection

Collect and label your own hand gesture data:
- Select option `1` from the main menu
- Press a letter key (A-Y, excluding J and Z) to select which sign you want to capture
- Position your hand to form the selected sign
- Press SPACE to capture samples
- Collect at least 20-30 samples per sign for good results
- Press ESC to exit data collection mode

### 2. Model Training

Train the CNN model on your collected data:
- Select option `2` from the main menu
- The script will automatically load your collected samples
- Training progress and results will be displayed
- The trained model will be saved to the `trained_model` directory

### 3. Real-Time Recognition

Test the trained model with real-time recognition:
- Select option `3` for basic recognition or option `4` for recognition with text-to-speech
- Form hand gestures in front of your webcam
- The recognized letters will be displayed on screen
- Press 'c' to clear the current text
- Press 'q' to quit recognition mode

## Project Structure

```
sign-language-recognition/
├── sign_language_realtime_training.py  # Main application script
├── collected_data/                     # Directory for collected training samples
│   ├── A/                              # Samples for letter A
│   ├── B/                              # Samples for letter B
│   └── ...                             # Other letter directories
├── trained_model/                      # Directory for trained model files
│   ├── sign_language_model.keras       # Trained model
│   └── training_history.png            # Training performance plot
└── README.md                           # This file
```

## How It Works

1. **Hand Detection**: MediaPipe's hand tracking solution detects and tracks hand landmarks in the webcam feed.
2. **Image Processing**: The hand region is extracted, converted to grayscale, resized to 28x28 pixels, and normalized.
3. **Model Architecture**: A Convolutional Neural Network (CNN) processes the hand images to recognize gestures.
4. **Prediction Smoothing**: A buffer-based approach reduces flickering in predictions.
5. **Text Formation**: Recognized letters are combined to form words and sentences.

## Model Architecture

The CNN model consists of:
- 3 convolutional layers with max pooling
- Flatten layer
- Dense layer with dropout for regularization
- Output layer with softmax activation

## Limitations

- Recognizes static hand gestures only (not dynamic gestures like J and Z in ASL)
- Performance depends on lighting conditions and camera quality
- Requires sufficient training data for each gesture
- May struggle with similar-looking gestures

## Future Improvements

- Support for dynamic gestures (motion-based signs)
- Word prediction and auto-completion
- Sentence formation with grammar correction
- Support for two-handed signs
- Integration with other accessibility tools

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the hand tracking solution
- [TensorFlow](https://www.tensorflow.org/) for the machine learning framework
- [Sign Language MNIST Dataset](https://www.kaggle.com/datamunge/sign-language-mnist) for inspiration
