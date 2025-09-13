# Deepfake Detection System

This project aims to build a deep learning model to detect AI-generated fake videos and images (deepfakes). With the rise of misinformation, such a system is crucial for maintaining digital integrity.

## Running the Application

This project includes a web application that allows you to upload an image and get a prediction from the trained model.

1.  **Install Dependencies**: Make sure you have Python installed. Then, install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**: Start the Streamlit application from your terminal:
    ```bash
    streamlit run app.py
    ```
    Your web browser will open with the application running.

## Project Plan

1.  **Project Setup**: Initialize the repository and project structure.
2.  **Dataset Research & Preparation**: Find and preprocess a suitable dataset for training. We started with the [Deepfake Detection Dataset from Hugging Face](https://huggingface.co/datasets/saakshigupta/deepfake-detection-dataset-v3).
3.  **Model Development**:
    *   Implement a Convolutional Neural Network (CNN) to analyze images and video frames.
    *   Train the model on the chosen dataset.
    *   Evaluate and iterate on the model to improve its accuracy.
4.  **Real-time Detection (Future Goal)**: Explore methods to implement real-time detection capabilities.
5.  **Browser Extension (Future Goal)**: Package the model into a browser extension for use on social media platforms.

## Getting Started

To see how the project was developed, you can follow the Jupyter Notebooks in order:
1. `1_Data_Exploration.ipynb`
2. `2_Model_Training.ipynb`
3. `3_Model_Evaluation.ipynb`