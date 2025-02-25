# Potato Diseases Predictor - Version 1

## Overview
The **Potato Diseases Predictor** is a deep learning-based model designed to classify and detect diseases in potato plants using image processing techniques. This project utilizes a Convolutional Neural Network (CNN) to accurately identify common potato diseases from leaf images.

## Features
- Classifies potato leaves into different disease categories.
- Uses a trained TensorFlow/Keras deep learning model.
- Provides confidence scores for predictions.
- Easy-to-use interface for uploading and analyzing images.

## Dataset
The model is trained on a dataset of potato leaf images, categorized into:
- **Healthy**
- **Early Blight**
- **Late Blight**

## Technologies Used
- **Python**
- **TensorFlow / Keras**
- **OpenCV** (for image preprocessing)
- **Matplotlib** (for visualization)
- **FastAPI** (for deploying the model as an API)
- **React** (for frontend, if applicable)

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Potato_Diseases_Predictor.git
   cd Potato_Diseases_Predictor
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the model training script (if training from scratch):
   ```sh
   python train.py
   ```

4. Start the FastAPI backend:
   ```sh
   uvicorn app:app --reload
   ```

## Usage
- Upload an image of a potato leaf through the web interface or API.
- The model will process the image and return the predicted disease category with confidence.
- View results in the frontend or through API responses.

## API Endpoints
- **POST `/predict`**: Upload an image and get a prediction.
- **GET `/health`**: Check if the API is running.

## Model Saving & Loading
The trained model is saved in `.keras` format:
```python
model.save("models/potato_disease_model.keras")
```
To load the model:
```python
from tensorflow.keras.models import load_model
model = load_model("models/potato_disease_model.keras")
```

## Future Improvements
- Improve model accuracy with a larger dataset.
- Develop a mobile-friendly version.
- Integrate real-time prediction using a camera.[](https://github.com/Rajat-Kumar-Pandey/FullStack-Modals-Project.git)

## Contributors
- **Your Name** ([GitHub Profile](https://github.com/Rajat-Kumar-Pandey))

## License
This project is open-source and available under the MIT License.
