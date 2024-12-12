import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO  # Use the new ultralytics library

# Title of the App
st.title("YOLO Object Detection")
st.write("Upload an image, and let the YOLO model detect objects in it.")

# Cache the model loading for efficiency
@st.cache_resource
def load_model():
    """
    Load the YOLO model. Ensure the 'kidney_yolo.pt' file is in the same directory.
    """
    model = YOLO("kidney_yolo.pt")  # Update the path if your model is stored elsewhere
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
    results = model.predict(image_np)

    # Render and display results
    st.image(results[0].plot(), caption="Detection Results", use_column_width=True)

    # Optional: Display raw detection data (confidence, bounding boxes, etc.)
    st.write("Detection Results:")
    st.write(results[0].boxes.data.cpu().numpy())  # Display raw detection data
