import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model.keras")

# Class names
class_names = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]


def predict_flower(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    return class_names[predicted_index], confidence