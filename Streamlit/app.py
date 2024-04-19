import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
try:
    model = load_model('potatoes.h5')
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Function to preprocess the image
def preprocess_image(image_file):
    try:
        img = image.load_img(image_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error in image preprocessing: {e}")
        return None

# Function to make predictions
def predict_disease(image_file):
    try:
        processed_image = preprocess_image(image_file)
        if processed_image is not None:
            prediction = model.predict(processed_image)
            return prediction
        else:
            return None
    except Exception as e:
        st.error(f"Error in making prediction: {e}")
        return None

# Streamlit app
def main():
    st.title("Potato Diseases Classification")
    st.write("Upload an image of a potato leaf to classify its disease.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        if st.button('Classify'):
            st.write("Classifying...")
            prediction = predict_disease(uploaded_file)
            if prediction is not None:
                disease_class = np.argmax(prediction)
                if disease_class == 0:
                    st.write("Prediction: Early Blight")
                elif disease_class == 1:
                    st.write("Prediction: Late Blight")
                else:
                    st.write("Prediction: Healthy Potato Leaf")
            else:
                st.write("Failed to make a prediction. Check earlier errors.")
    else:
        st.write("Please upload an image of a potato leaf.")

if __name__ == '__main__':
    main()
