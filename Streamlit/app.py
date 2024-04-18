import streamlit as st
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Function to run prediction
def run_prediction(img_path, model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)

    return np.argmax(prediction)

def main():
    st.title("Potato Disease Classification")

    # Get the directory of the current script file
    THIS_FOLDER = Path(__file__).resolve().parent

    # Define the file path for the model
    model_path = THIS_FOLDER / "potatoes.h5"

    # Load the model
    if not os.path.exists(model_path):
        st.error("Model file not found.")
        return

    try:
        model = load_model(model_path)
    except Exception as e:
        st.error("Error loading the model.")
        st.error(str(e))
        return

    # Load and preprocess an example image for prediction
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = np.array(uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Run prediction
        prediction = run_prediction(image, model)
        classes = ['early_blight', 'late_blight', 'healthy']
        st.write("Prediction:", classes[prediction])

    # Display predictions for a batch of test images
    display_batch = st.button("Display Predictions for Batch")
    if display_batch:
        test_images_folder = THIS_FOLDER / "C:/Users/SHUBHAM SHARMA/Deep_Learning_project/Plant"
        classes = ['early_blight', 'late_blight', 'healthy']
        for class_name in classes:
            class_path = test_images_folder / class_name
            images = []
            for filename in os.listdir(class_path):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    img_path = class_path / filename
                    images.append(img_path)

            if images:
                st.write(f"Displaying predictions for {class_name} images:")
                fig = plt.figure(figsize=(15, 15))
                for i in range(min(len(images), 9)):
                    ax = fig.add_subplot(3, 3, i + 1)
                    img = plt.imread(images[i])
                    ax.imshow(img)
                    predicted_class = run_prediction(images[i], model)
                    ax.set_title(f"Predicted: {classes[predicted_class]}")
                    ax.axis("off")
                st.pyplot(fig)
            else:
                st.write(f"No {class_name} images found.")

if __name__ == "__main__":
    main()
