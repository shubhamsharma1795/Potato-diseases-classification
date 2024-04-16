import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
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
def predict_disease(model, image_file):
    processed_image = preprocess_image(image_file)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title("Potato Diseases Classification")
    st.write("Upload an image of a potato leaf to classify its disease.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Define the model file path
    model_path = "Streamlit/potatoes.h5"
    
    # Verify file existence and load the model
    if os.path.exists(model_path):
        try:
            # Load the model
            model = load_model(model_path, compile=False, custom_objects={})
            # Wrap the model with a Sequential model with desired batch size
            model = tf.keras.Sequential([model])
            model._layers[0]._batch_input_shape = (None, 224, 224, 3)  # Set batch size and input shape
            st.write("Model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    else:
        st.error("Model file not found. Please make sure the model file exists at the specified path.")
        st.stop()
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        if st.button('Classify'):
            st.write("Classifying...")
            try:
                prediction = predict_disease(model, uploaded_file)
                st.write("Prediction:", prediction)
                disease_class = np.argmax(prediction)
                if disease_class == 0:
                    st.write("Prediction: Early Blight")
                elif disease_class == 1:
                    st.write("Prediction: Late Blight")
                else:
                    st.write("Prediction: Healthy Potato Leaf")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
