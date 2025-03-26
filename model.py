import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Load model
model = load_model("unet_model.h5")
matched_data = pd.read_csv(r'C:\Users\kier0\Downloads\mri_data\kaggle_3m\patient_data.csv')
# Example of loading a batch of images and masks from the CSV
image_data = []
mask_data = []
def load_image_and_mask(image_path, mask_path, target_size=(128, 128)):
    # Load the image and mask
    image = Image.open(image_path).resize(target_size)
    mask = Image.open(mask_path).resize(target_size)

    # Convert image to numpy array and normalize it
    image = img_to_array(image).astype(np.float32) / 255.0
    mask = np.array(mask).astype(np.uint8)
    mask = (mask > 127).astype(np.uint8)

    # Ensure correct shape (batch, height, width, channels)
    if image.shape[-1] != 3:  # If the image is grayscale
        img = np.stack([image] * 3, axis=-1)  # Convert to 3-channel RGB

    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image, mask

for index, row in matched_data.iterrows():
    image_path = row['image_path']
    mask_path = row['mask_path']

    # Load the image and mask for the given pair
    image, mask = load_image_and_mask(image_path, mask_path)

    # Append to the list
    image_data.append(image)
    mask_data.append(mask)

# Convert lists to numpy arrays for training
image_data = np.array(image_data)
mask_data = np.array(mask_data)
unique_values = np.unique(mask_data)
print(f"Unique values in the mask: {unique_values}")

# Print the shape of the data
print(f"Image data shape: {image_data.shape}")
print(f"Mask data shape: {mask_data.shape}")

# You can now split the data into training and validation sets (e.g., 80-20 split) for training
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(image_data, mask_data, test_size=0.2, random_state=42)


# Save the processed data to numpy files (optional)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
# Load validation images and masks




for i in range(len(X_train)):
    image_array = (X_train[i] * 255).astype(np.uint8)  # Get an image from dataset
      # Preprocess image

    # Predict the mask
    predicted_mask = model.predict(image_array)

    # Post-process the output
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Threshold for binary mask
    predicted_mask = np.squeeze(predicted_mask)  # Remove batch dimension

    # Load original image for visualization
    original_image = Image.fromarray(np.squeeze(image_array).astype(np.uint8)).resize((128, 128))

    # Plot the images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f"Original Image {i+1}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap="gray")
    plt.title(f"Predicted Mask {i+1}")
    plt.axis("off")

    plt.show()


