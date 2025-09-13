import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# --- Model Definition ---
# This class must be identical to the one used for training
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# --- Image Transformations ---
# Use the same transformations as in training/evaluation
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Load Model ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    model = SimpleCNN()
    # Load the saved model state. Ensure 'deepfake_detector_model.pth' is in the same directory.
    try:
        model.load_state_dict(torch.load('deepfake_detector_model.pth', map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- Streamlit App Interface ---
st.title("Deepfake Detection System")
st.write("Upload an image to determine if it is real or a deepfake.")

if model is None:
    st.error("Model file not found. Please make sure 'deepfake_detector_model.pth' is in the project directory.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")

        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image and make a prediction
        image_tensor = data_transforms(image).unsqueeze(0) # Add batch dimension
        
        with torch.no_grad():
            output = model(image_tensor)
            prediction_score = output.item()
            prediction = 1 if prediction_score > 0.5 else 0

        if prediction == 1:
            st.success(f"**Prediction: Real** (Confidence: {prediction_score:.2%})")
        else:
            st.error(f"**Prediction: Fake** (Confidence: {1 - prediction_score:.2%})")