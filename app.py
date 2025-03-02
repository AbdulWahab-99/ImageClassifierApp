import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("models/imageclassifier.h5")

# Define class labels
class_labels = ["Happy", "Sad"]  # Modify if necessary

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to match model input shape
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title(":green[Happy] or :red[Sad] Detector 😊☹️")
st.write("Upload an image, and the model will predict if the person is happy or sad.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    # Make a prediction
    # Make a prediction
    prediction = model.predict(processed_image)

    # Extract the single prediction value (since it's a binary classification)
    prediction_value = prediction[0][0]

    # Determine the class based on the threshold (0.5)
    if prediction_value < 0.5:
        predicted_class = "Happy"
    else:
        predicted_class = "Sad"

    # Display only the prediction result
    st.subheader(f"Prediction: {predicted_class} 🎯")



