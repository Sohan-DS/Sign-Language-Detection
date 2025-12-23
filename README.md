# Sign Language Detection using Deep Learning

This project implements an American Sign Language (ASL) alphabet detection system using Convolutional Neural Networks (CNNs). The model is trained on image data and supports real-time sign language prediction using a webcam.

## Project Overview

- ASL alphabet classification (A–Z)
- CNN model built using TensorFlow and Keras
- Real-time webcam-based prediction using OpenCV
- Implemented and tested using VS Code and a virtual environment

## Dataset

This project uses the ASL Alphabet Dataset from Kaggle.

Dataset link:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

After downloading and extracting the dataset, place the folders in the project root directory as:

asl_alphabet_train/
asl_alphabet_test/

Note: The dataset is not included in this repository due to size and Kaggle licensing restrictions.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

## Setup Instructions

1. Clone the repository  
git clone https://github.com/your-username/Sign-Language-Detection.git  
cd Sign-Language-Detection

2. Create and activate a virtual environment  
python -m venv venv  

Windows:  
venv\Scripts\activate  

macOS/Linux:  
source venv/bin/activate  

3. Install dependencies  
pip install -r requirements.txt

4. Download the dataset from Kaggle and extract it into the project folder.

## Model Training

Run the notebook "Sign Language Identification.ipynb" to load the dataset, train the CNN model, and save the trained model.

## Real-Time Webcam Prediction

After training, the model can be used for live sign language detection using the system webcam. The webcam opens, detects the hand inside a fixed region, and displays the predicted ASL alphabet in real time.

Press 'q' to exit the webcam window.

## Notes

- Ensure good lighting conditions for better predictions
- Keep the hand inside the bounding box
- Model accuracy depends on training epochs and dataset quality

## Future Improvements

- Improve accuracy with data augmentation
- Extend detection to words and sentences
- Deploy as a web or mobile application

## Author

Developed as a learning project in Machine Learning and Computer Vision.
