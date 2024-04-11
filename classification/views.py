from django.shortcuts import render
from django.http import HttpResponse
import os
import cv2
import numpy as np
import tensorflow as tf

# Define the path to the saved Keras model
model_filename = 'brain_tumor_model.h5'
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, model_filename)

# Define the directory to save uploaded images
image_dir = os.path.join(model_dir, 'uploaded_images')
os.makedirs(image_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Function to load the Keras model
def load_model(filepath):
    model = tf.keras.models.load_model(filepath)
    return model

# Load the Keras model
try:
    model = load_model(model_path)
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Please check the model path.")

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (150, 150))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    img_normalized = img_rgb / 255.0  # Normalize pixel values
    img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    return img_input


# Function to predict brain tumor using the loaded model
def predict_tumor(image_path):
    img_input = preprocess_image(image_path)
    prediction = model.predict(img_input)
    tumor_classes = ['giloma_tumor', 'no_tumor', 'pituitary_tumor', 'meningioma_tumor']
    predicted_class = np.argmax(prediction)
    return tumor_classes[predicted_class]

# Django view function for index page
def index(request):
    return render(request, 'index.html')

# Django view function for predicting brain tumor
def predict_tumor_view(request):
    if request.method == 'POST':
        # Get the uploaded image from the request
        try:
            image_file = request.FILES['image']
        except KeyError:
            return HttpResponse("No image file uploaded.")

        # Save the uploaded image to the directory
        image_path = os.path.join(image_dir, image_file.name)
        with open(image_path, 'wb') as f:
            f.write(image_file.read())
        
        try:
            # Predict the brain tumor using the uploaded image
            predicted_tumor = predict_tumor(image_path)
        except Exception as e:
            return HttpResponse(f"Error: {str(e)}")

        # Return the predicted tumor type as a response
        return render(request, 'result.html', {'predicted_tumor': predicted_tumor})
    else:
        # If the request method is not POST, return a default response
        return HttpResponse("Please upload an image.")
