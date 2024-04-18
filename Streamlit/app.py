import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Get the absolute path to the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'potatoes.h5')

# Load the pre-trained model
model = load_model(model_path)

def predict_disease(img):
    img = img.resize((255, 255))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0  # Normalization
    prediction = model.predict(img)
    return prediction

def main():
    st.title("Potato Disease Classification")

    uploaded_file = st.file_uploader("Choose a potato image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Potato Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        prediction = predict_disease(img)
        labels = ['Early Blight', 'Late Blight', 'Healthy']
        result = labels[np.argmax(prediction)]
        st.success(f'The potato is classified as: {result}')

if __name__ == '__main__':
    main()
