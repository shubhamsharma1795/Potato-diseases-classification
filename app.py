import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('potatoes.h5')

# Function to preprocess the image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_disease(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title("Potato Diseases Classification")
    st.write("Upload an image of a potato leaf to classify its disease.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        prediction = predict_disease(uploaded_file)
        
        classes = ['Early Blight', 'Late Blight', 'Healthy']
        predicted_class = classes[np.argmax(prediction)]
        st.write(f"Prediction: {predicted_class}")

if __name__ == '__main__':
    main()
