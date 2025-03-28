from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# Load the trained model
model = load_model("unet_model.h5")


# Preprocessing function
def preprocess_image(image: Image.Image, target_size=(128, 128)):
    image = image.resize(target_size)  # Resize to model's expected input size
    image = np.array(image) / 255.0  # Normalize to [0,1]

    if len(image.shape) == 2:  # If grayscale, convert to 3-channel
        image = np.stack([image, image, image], axis=-1)

    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image.astype(np.float32)


# Endpoint to handle image upload and segmentation
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess image
    processed_image = preprocess_image(image)

    # Run inference
    prediction = model.predict(processed_image)

    # Convert to binary mask (threshold at 0.5)
    binary_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)

    return {"mask": binary_mask.tolist()}  # Convert to list for JSON response

# Run the server using: uvicorn backend:app --reload
