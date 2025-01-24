import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
MODEL_PATH = 'fruit_disease_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the uploaded image
def preprocess_image(image, img_size=(100, 100)):
    image = cv2.resize(image, img_size)  # Resize to the size used during training
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("Fruit Disease Detection")
st.write("Upload an image of a fruit to determine if it is healthy or diseased.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    label = "Diseased" if prediction[0][0] > 0.5 else "Healthy"

    # Display the result
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{prediction[0][0] * 100:.2f}%**" if label == "Diseased" else f"Confidence: **{(1 - prediction[0][0]) * 100:.2f}%**")
