import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import base64


st.markdown("""
    <style>
    .title {
        font-size:36px !important;
        color: #3F72AF;
        text-align: center;
    }
    .subtext {
        font-size:18px !important;
        color: #112D4E;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-box {
        background-color: #F9F7F7;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #DBE2EF;
        margin-top: 20px;
        text-align: center;
    }
    .pred-pneumonia {
        color: red;
        font-weight: bold;
        font-size: 22px;
    }
    .pred-normal {
        color: green;
        font-weight: bold;
        font-size: 22px;
    }
    .upload-button {
        background-color: #3F72AF;
        color: white;
        padding: 0.5em 1em;
        border-radius: 8px;
        text-align: center;
        display: inline-block;
        margin: 20px auto;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🩺 Pneumonia Detection from Chest X-Ray</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Upload a chest X-ray image. The model will predict whether it is <strong>NORMAL</strong> or <strong>PNEUMONIA</strong>.</div>', unsafe_allow_html=True)


# Load model
@st.cache_resource

def load_model():
    # Define the same model architecture used during training
    conv_layers = nn.Sequential(
        nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2)
    )
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 150, 150)
        dummy_output = conv_layers(dummy_input)
        flatten_dim = dummy_output.view(1, -1).shape[1]
    
    model = nn.Sequential(
        conv_layers,
        nn.Flatten(),
        nn.Linear(flatten_dim, 256), nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load("model1.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Image preprocessing
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray",  use_container_width=True)

    # Preprocess and predict
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = "PNEUMONIA" if output.item() > 0.5 else "NORMAL"
        confidence = output.item()

    # Display result
    st.markdown(f"### 🧠 Prediction: **{prediction}**")
    st.markdown(f"Confidence: **{confidence:.2f}**")
