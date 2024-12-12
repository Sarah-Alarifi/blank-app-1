import streamlit as st
from PIL import Image
import torch
import numpy as np

# Title of the App
st.title("YOLO Object Detection")
st.write("Upload an image, and let the YOLO model detect objects in it.")

# Cache the model loading for efficiency
@st.cache_resource
def load_model():
    """
    Load the YOLO model. Ensure the 'model.pt' file is in the same directory.
    """
    model = torch.hub.load(
        'ultralytics/yolov5', 
        'custom', 
        path='kidney_yolo.pt',  # Update the path if your model is stored elsewhere
        force_reload=True,
        device='cpu'  # Ensure compatibility with CPU-only environments
    )
    return model

# Load the model
model = load_model()

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting objects...")

    # Convert image to numpy array
    image_np = np.array(image)

    # Perform inference
    results = model(image_np)

    # Render and display results
    st.image(results.render()[0], caption="Detection Results", use_column_width=True)

    # Optional: Display raw detection data (confidence, bounding boxes, etc.)
    st.write("Detection Results:")
    st.write(results.pandas().xyxy[0])  # Display detections in a pandas DataFrame format
