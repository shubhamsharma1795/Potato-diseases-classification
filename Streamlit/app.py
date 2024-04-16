import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Function to preprocess the image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_disease(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title("Potato Diseases Classification")
    st.write("Upload an image of a potato leaf to classify its disease.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Load the model without specifying custom_objects
    try:
        model_path = os.path.abspath('potatoes.h5')
        model = load_model(model_path)
    except Exception as e:
        st.error("Error loading model. Please check the model file path and try again.")
        st.stop()
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        if st.button('Classify'):
            st.write("Classifying...")
            prediction = predict_disease(model, uploaded_file)
            st.write("Prediction:", prediction)  # Add this line to print the prediction
            disease_class = np.argmax(prediction)
            if disease_class == 0:
                st.write("Prediction: Early Blight")
            elif disease_class == 1:
                st.write("Prediction: Late Blight")
            else:
                st.write("Prediction: Healthy Potato Leaf")

if __name__ == '__main__':
    main()