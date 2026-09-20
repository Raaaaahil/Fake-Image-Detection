# TruthLens AI

## AI-Generated Image Detection System

TruthLens AI is an AI-powered image detection system designed to
estimate whether an uploaded image is **AI-generated** or
**real/natural**.

The system combines a **ResNet-18 deep learning model**, **FastAPI
backend**, and **React + TypeScript frontend** to provide an interactive
image analysis experience.

------------------------------------------------------------------------

## Features

-   AI-generated vs real image classification
-   ResNet-18 based deep learning model
-   Robustly trained detection model
-   Image upload through web interface
-   Drag-and-drop image upload
-   Image preview before analysis
-   AI and real probability scores
-   Confidence-level indication
-   FastAPI REST API
-   React + TypeScript frontend
-   Tailwind CSS based responsive UI
-   Support for JPG, JPEG, PNG and WEBP images
-   Maximum upload size of 10 MB
-   CUDA/GPU support when available

------------------------------------------------------------------------

## System Architecture

``` text
                    ┌─────────────────────┐
                    │   React Frontend    │
                    │ React + TypeScript  │
                    │    Tailwind CSS     │
                    └──────────┬──────────┘
                               │
                               │ HTTP POST
                               │ /predict
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Backend   │
                    │     Python API      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Image Preprocess  │
                    │ Resize + Normalize  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ResNet-18 Model   │
                    │   Binary Classifier │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
              AI Generated         Real / Natural
```

------------------------------------------------------------------------

## Tech Stack

### Machine Learning

-   Python
-   PyTorch
-   Torchvision
-   ResNet-18
-   CUDA / NVIDIA GPU support
-   ImageNet normalization

### Backend

-   Python
-   FastAPI
-   Uvicorn
-   Pillow
-   Python Multipart

### Frontend

-   React
-   TypeScript
-   Vite
-   Tailwind CSS
-   Lucide React

### Development Tools

-   Visual Studio Code
-   Git
-   GitHub
-   PowerShell

------------------------------------------------------------------------

## Model

TruthLens AI uses a **ResNet-18 convolutional neural network** adapted
for binary image classification.

The model predicts two classes:

  Class    Meaning
  -------- ----------------------
  `AI`     AI-generated image
  `REAL`   Real / natural image

The application uses the robustly trained model:

``` text
models/fake_image_resnet18_robust.pth
```

------------------------------------------------------------------------

## Dataset

The primary training dataset used for the project was the **GenImage ADM
dataset**.

The dataset contained approximately:

-   **319,447 usable training images**
-   **12,000 validation images**
-   **2 classes**
-   AI-generated images
-   Real/natural images

During dataset preparation, six corrupted PNG files were removed from
the training set.

### Dataset Classes

``` text
AI     → AI-generated images
REAL   → Natural/real images
```

------------------------------------------------------------------------

## Model Training

### Original Model

The initial ResNet-18 model was trained using the GenImage ADM dataset.

Training configuration included:

-   Image size: 224 × 224
-   Batch size: 8
-   Optimizer: AdamW
-   Learning rate: 1e-4
-   Epochs: 10
-   ImageNet normalization
-   Random horizontal flipping
-   Automatic Mixed Precision (AMP)

### Original Validation Results

  Metric        Result
  ----------- --------
  Accuracy      99.91%
  Precision     99.93%
  Recall        99.88%
  F1 Score      99.91%
  ROC-AUC       1.0000

### Confusion Matrix

``` text
                 Predicted
                AI       REAL

Actual AI      5993       7
Actual REAL       4    5996
```

------------------------------------------------------------------------

## Robust Model

The project also includes a robustly trained ResNet-18 model designed to
improve performance on images outside the original validation
distribution.

Additional training augmentations included:

-   Random resized cropping
-   Horizontal flipping
-   Color jitter
-   Random grayscale
-   Gaussian blur
-   Random erasing
-   ImageNet normalization

The robust model achieved:

``` text
GenImage validation accuracy: 96.24%
```

The robust model is the model used by the web application.

------------------------------------------------------------------------

## External Testing

To evaluate generalization beyond the original validation set, the
models were tested using a separate collection of 24 external images:

-   11 AI-generated images
-   13 real images

### Original Model

  Category        Correct   Accuracy
  ------------- --------- ----------
  AI images        0 / 11      0.00%
  Real images     13 / 13    100.00%
  Overall         13 / 24     54.17%

### Robust Model

  Category        Correct   Accuracy
  ------------- --------- ----------
  AI images        8 / 11     72.73%
  Real images     10 / 13     76.92%
  Overall         18 / 24     75.00%

The external test demonstrates that robustness and generalization remain
important challenges for AI-generated image detection.

The external test set is relatively small and should not be interpreted
as representative of all image sources or AI-generation methods.

------------------------------------------------------------------------

## Confidence Interpretation

TruthLens AI reports the model's probability distribution between the
two classes.

The interface also provides a confidence interpretation:

  Confidence   Interpretation
  ------------ ---------------------
  ≥ 90%        High confidence
  70--89.99%   Moderate confidence
  \< 70%       Low confidence

Confidence represents how strongly the model favors its predicted class.
It does **not** represent a guarantee that the classification is
correct.

------------------------------------------------------------------------

## Project Structure

``` text
Fake-Image-Detection/
│
├── backend/
│   ├── app.py
│   ├── predictor.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.tsx
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.ts
│
├── models/
│   ├── fake_image_resnet18.pth
│   └── fake_image_resnet18_robust.pth
│
├── evaluate.py
├── evaluate_external.py
├── predict.py
├── train.py
├── train_robust.py
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

# Installation

## 1. Clone the Repository

``` bash
git clone https://github.com/Raaaaahil/Fake-Image-Detection.git
cd Fake-Image-Detection
```

------------------------------------------------------------------------

# Backend Setup

## 2. Create a Python Virtual Environment

``` powershell
python -m venv .venv
```

Activate it on Windows:

``` powershell
.\.venv\Scripts\Activate.ps1
```

------------------------------------------------------------------------

## 3. Install Backend Dependencies

``` powershell
cd backend
pip install -r requirements.txt
```

The backend dependencies include:

``` text
fastapi
uvicorn
python-multipart
pillow
```

------------------------------------------------------------------------

## 4. Start the Backend

From the `backend` directory:

``` powershell
python -m uvicorn app:app --reload
```

The API will run at:

``` text
http://127.0.0.1:8000
```

FastAPI documentation is available at:

``` text
http://127.0.0.1:8000/docs
```

------------------------------------------------------------------------

# Frontend Setup

Open another terminal.

Navigate to the frontend:

``` powershell
cd frontend
```

Install dependencies:

``` powershell
npm install
```

Start the development server:

``` powershell
npm run dev
```

The frontend will normally be available at:

``` text
http://localhost:5173
```

If that port is already in use, Vite may automatically select another
available port.

------------------------------------------------------------------------

# API

## Health Check

### Endpoint

``` text
GET /health
```

Example response:

``` json
{
  "status": "healthy"
}
```

------------------------------------------------------------------------

## Image Prediction

### Endpoint

``` text
POST /predict
```

### Request

Send an image using multipart form data:

``` text
file=<image>
```

### Supported Formats

``` text
JPEG
JPG
PNG
WEBP
```

### Maximum Size

``` text
10 MB
```

### Example Response

``` json
{
  "filename": "example.jpg",
  "prediction": "AI Generated",
  "class": "AI",
  "confidence": 95.44,
  "ai_probability": 95.44,
  "real_probability": 4.56
}
```

------------------------------------------------------------------------

# Using the Application

1.  Start the FastAPI backend.
2.  Start the React frontend.
3.  Open the frontend in a browser.
4.  Upload an image.
5.  Preview the selected image.
6.  Click **Analyze image**.
7.  The image is sent to the FastAPI backend.
8.  The ResNet-18 model analyzes the image.
9.  The prediction and probabilities are returned to the frontend.
10. TruthLens AI displays the result.

------------------------------------------------------------------------

# Important Note About Accuracy

High validation accuracy on a specific dataset does not guarantee
equivalent performance on unseen real-world images.

AI-generated images can vary significantly depending on:

-   Image-generation model
-   Generation technique
-   Image resolution
-   Compression
-   Resizing
-   Post-processing
-   Editing
-   Image source
-   Dataset distribution

Therefore, TruthLens AI should be considered a **machine-learning based
estimation system**, not an absolute authenticity verification system.

------------------------------------------------------------------------

# Limitations

-   The current model is trained primarily on the GenImage ADM dataset.
-   External testing was performed on a relatively small set of 24
    images.
-   Some unseen AI-generated images may be classified as real.
-   Some real images may be classified as AI-generated.
-   Image editing and compression can affect predictions.
-   Performance may vary across different AI-generation models.
-   The system does not provide forensic proof of image authenticity.

------------------------------------------------------------------------

# Future Improvements

Potential improvements include:

-   Training on larger and more diverse datasets
-   Adding images from more AI-generation models
-   Transformer-based image detectors
-   Ensemble model architecture
-   Frequency-domain analysis
-   JPEG artifact analysis
-   Metadata analysis
-   Image-forensics features
-   Larger real-world evaluation datasets
-   Explainable AI visualizations
-   Production deployment
-   Cloud-based inference API

------------------------------------------------------------------------

# License

This project is intended primarily for **academic and educational
purposes**.

See the repository license file for the applicable usage terms.

------------------------------------------------------------------------

# Author

**Intakhab Nabi**

B.Tech Computer Science & Engineering

GitHub:\
https://github.com/Raaaaahil

------------------------------------------------------------------------

## Disclaimer

TruthLens AI provides a machine-learning based estimate of whether an
image appears AI-generated or real/natural.

The output should not be treated as definitive proof of image
authenticity. Always consider the source, context, and other available
evidence when evaluating an image.
