import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Function to run prediction
def run_prediction(image, model):
    # Preprocess the image
    img = tf.keras.preprocessing.image.img_to_array(image)
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)

    return np.argmax(prediction)

def main():
    st.title("Potato Disease Classification")

    # Load the model
    model_path = 'potatoes.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found.")
        return

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error("Error loading the model.")
        st.error(str(e))
        return

    # Load and preprocess an example image for prediction
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = tf.keras.preprocessing.image.load_img(uploaded_image, target_size=(224, 224))
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Run prediction
        prediction = run_prediction(image, model)
        classes = ['early_blight', 'late_blight', 'healthy']
        st.write("Prediction:", classes[prediction])

    # Display predictions for a batch of test images
    display_batch = st.button("Display Predictions for Batch")
    if display_batch:
        test_images_folder = 'C:/Users/SHUBHAM SHARMA/Deep_Learning_project/Plant'
        classes = ['early_blight', 'late_blight', 'healthy']
        for class_name in classes:
            class_path = os.path.join(test_images_folder, class_name)
            images = []
            for filename in os.listdir(class_path):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    img = tf.keras.preprocessing.image.load_img(os.path.join(class_path, filename), target_size=(224, 224))
                    images.append(img)

            if images:
                st.write(f"Displaying predictions for {class_name} images:")
                fig = plt.figure(figsize=(15, 15))
                for i in range(min(len(images), 9)):
                    ax = fig.add_subplot(3, 3, i + 1)
                    ax.imshow(images[i])
                    predicted_class = run_prediction(images[i], model)
                    ax.set_title(f"Predicted: {classes[predicted_class]}")
                    ax.axis("off")
                st.pyplot(fig)
            else:
                st.write(f"No {class_name} images found.")

if __name__ == "__main__":
    main()
