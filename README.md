# Face Recognition using MediaPipe Face Mesh

A robust facial recognition system that uses geometric facial measurements from MediaPipe's Face Mesh for accurate identity verification.

## Overview

This project implements a facial recognition system that captures and analyzes facial landmarks to create unique biometric signatures. Unlike traditional image-based facial recognition, this system uses precise geometric measurements between facial points, making it more resilient to changes in lighting, makeup, or minor appearance changes.

## Features

- **Real-time face detection and tracking**
- **Automated camera zoom adjustment** for optimal face positioning
- **468 facial landmark detection** using MediaPipe Face Mesh
- **Geometric feature extraction** for robust identification
- **Multiple recognition models**: Decision Tree and ANN support
- **Visual confidence indicators** showing match quality
- **Stable recognition tracking** with streak monitoring
- **Easy data collection** for enrolling new users

## Prerequisites

- Python 3.7+
- Webcam
- Git

## Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/face-recognition-face-mesh.git
cd face-recognition-face-mesh
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Packages

```bash
# Install all dependencies
pip install -r requirements.txt
```

If no requirements.txt is available, install the following packages:

```bash
pip install opencv-python mediapipe numpy cvzone pandas scikit-learn joblib tensorflow
```

### Step 4: Configure Pylance (Optional)

Create a file named `pyrightconfig.json` to eliminate Pylance warnings:

```json
{
    "include": [
        "*.py"
    ],
    "reportMissingImports": false,
    "reportAttributeAccessIssue": false
}
```

## Usage

### Data Collection Mode

To add new users to the system:

1. Edit `mesh1.py` to set the user's name:
   - Find line 569: `csv_writer.writerow([t2,sum1,sum2,...,'Ethan'])`
   - Change 'Ethan' to the name of the person you want to add (e.g., 'Alice')

2. Run the data collection script:
   ```bash
   python mesh1.py
   ```

3. Position yourself approximately 40cm from the camera
   - The system will automatically adjust the zoom
   - When your face angle is within the correct range (~97 degrees), data collection begins
   - It will collect 500 samples automatically
   - Stay relatively still during the process

### Training the Model

After collecting data for multiple individuals:

1. Train the decision tree model:
   ```bash
   python train_model.py
   ```

2. This creates two files:
   - `decision_tree_model.pkl`: The trained classifier
   - `label_encoder.pkl`: The label encoder for class names

### Recognition Mode

To identify people:

1. Run the prediction script:
   ```bash
   python predict_the_model.py
   ```

2. Position your face in the camera view
   - Keep approximately 40cm distance from the camera
   - The system will display:
     - Identified person's name
     - Confidence percentage
     - Similarity scores to all known individuals
     - Color-coded match quality indicators
     - Recognition streak (stable identification)

## Project Structure

- `mesh1.py` - Face detection and data collection script
- `predict_the_model.py` - Face recognition script
- `train_model.py` - Model training script
- `data3.csv` - Dataset containing facial measurements
- `decision_tree_model.pkl` - Trained classifier model
- `label_encoder.pkl` - Encoder for person names
- `debug_model.py` - (Optional) Script for troubleshooting model issues

## How It Works

1. **Face Detection**: MediaPipe detects 468 landmarks on the face
2. **Feature Extraction**: Calculates distances between key facial landmarks:
   - Top to bottom facial measurements
   - Left to right facial measurements 
   - Outer face contour distances
   - Nose feature measurements
3. **Feature Vector Creation**: Combines measurements into a 21-dimensional feature vector
4. **Classification**: Compares against known profiles using machine learning
5. **Confidence Calculation**: Determines similarity to known individuals

## Troubleshooting

### Common Issues:

1. **"No face detected"**
   - Ensure adequate lighting
   - Position your face to be fully visible
   - Check webcam permissions

2. **"One of the required model components is None"**
   - Ensure model files exist in the correct directory
   - Try retraining the model: `python train_model.py`

3. **Low Recognition Accuracy**
   - Collect more data samples
   - Try to maintain consistent lighting during both collection and recognition
   - Make sure to have enough examples from different angles

4. **Package Import Errors**
   - Ensure the virtual environment is activated
   - Verify all dependencies are installed: `pip list`
   - Try reinstalling the problematic package

## Development and Extension

To extend this project:

1. **Improve Feature Extraction**:
   - Modify the facial measurements in `face_dis()` function
   - Experiment with different facial landmark combinations

2. **Add User Interface**:
   - Create a simple GUI for easier user enrollment and recognition

3. **Improve Recognition Model**:
   - Implement a neural network model for potentially higher accuracy
   - Try different classifiers from scikit-learn

## License

This project is for educational purposes. Please respect privacy and obtain proper consent before collecting facial data.

---

## Acknowledgements

- MediaPipe team for the Face Mesh solution
- OpenCV community for computer vision tools
- scikit-learn for machine learning capabilities
