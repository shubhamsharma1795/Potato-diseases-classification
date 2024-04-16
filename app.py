import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import requests

# Function to preprocess the image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_disease(model, image_file):
    processed_image = preprocess_image(image_file)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title("Potato Diseases Classification")
    st.write("Upload an image of a potato leaf to classify its disease.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Load the model
    model_path = "potatoes.h5"
    if os.path.exists(model_path):
        st.write("Loading model from local file...")
        model = load_model(model_path)
    else:
        st.write("Model file not found locally. Attempting to download from GitHub...")
        github_model_url = "https://raw.githubusercontent.com/shubhamsharma1795/repository/main/potatoes.h5"
        try:
            response = requests.get(github_model_url)
            with open(model_path, "wb") as f:
                f.write(response.content)
            model = load_model(model_path)
            st.write("Model downloaded and loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        if st.button('Classify'):
            st.write("Classifying...")
            prediction = predict_disease(model, uploaded_file)
            st.write("Prediction:", prediction)
            disease_class = np.argmax(prediction)
            if disease_class == 0:
                st.write("Prediction: Early Blight")
            elif disease_class == 1:
                st.write("Prediction: Late Blight")
            else:
                st.write("Prediction: Healthy Potato Leaf")

if __name__ == '__main__':
    main()
