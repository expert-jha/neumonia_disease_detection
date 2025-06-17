import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim

import torchvision.models as models
import warnings
warnings.warn("ignore")

# Apply custom CSS styling
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        color: #3F72AF;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtext {
        font-size: 18px;
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

# Title and description
st.markdown('<div class="title">🩺 Pneumonia Detection from Chest X-Ray</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Upload a chest X-ray image. The model will predict whether it is <strong>NORMAL</strong> or <strong>PNEUMONIA</strong>.</div>', unsafe_allow_html=True)

# Load the model
#@st.cache(allow_output_mutation=True)
def load_model():
    model = models.resnet18(pretrained=False)
    model.load_state_dict(torch.load('resnet18-f37072fd.pth'))
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.fc.parameters(),lr=0.001)

    model.load_state_dict(torch.load("resnet18_pneumonia_trained.pth", map_location=torch.device("cpu")))
    
    model.eval()
    return model

model = load_model()

# Image preprocessing
def preprocess(image):
    transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ]
    )
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess and predict
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = "PNEUMONIA" if output.item() > 0.5 else "NORMAL"
        confidence = output.item()

    # Display result
    st.markdown(f"### 🧠 Prediction: **{prediction}**")
    st.markdown(f"Confidence: **{confidence:.2f}**")
