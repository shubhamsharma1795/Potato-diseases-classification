import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    try:
        model = load_model('potatoes.h5')
        return model
    except Exception as e:
        st.error("Failed to load the model. Please make sure the model file 'potatoes.h5' exists and is valid.")
        st.stop()

model = load_trained_model()

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
