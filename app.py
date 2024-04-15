import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import InputLayer

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    try:
        # Define a custom object dictionary to handle custom layers
        custom_objects = {'InputLayer': InputLayer}
        model = load_model('potatoes.h5', custom_objects=custom_objects)
        return model
    except Exception as e:
        return None, str(e)

model, load_error = load_trained_model()

if model is None:
    st.error(f"Failed to load the model. Error: {load_error}")
    st.stop()

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
        try:
            prediction = predict_disease(uploaded_file)
            classes = ['Early Blight', 'Late Blight', 'Healthy']
            predicted_class = classes[np.argmax(prediction)]
            st.write(f"Prediction: {predicted_class}")
        except Exception as e:
            st.error("Failed to classify the image. Please try with a different image.")
            st.error(str(e))

if __name__ == '__main__':
    main()
