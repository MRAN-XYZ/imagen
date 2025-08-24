import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Path to your dataset folder
dataset_path = '/content/imagen/Logos'  # Replace with the actual path if different, e.g., 'path/to/logos'

# Load all images from the folder
# Changed target_size to (128, 128) to ensure dimensions are compatible with the pooling/upsampling layers
# This avoids mismatch issues due to odd dimensions after multiple pooling operations
image_files = glob.glob(os.path.join(dataset_path, '*.[jp][pn]g'))  # Matches .jpg, .jpeg, .png
images = []
for file in image_files:
    img = load_img(file, target_size=(128, 128))  # Resize to 128x128
    img_array = img_to_array(img)
    images.append(img_array)

# Convert list to numpy array
X = np.array(images)

# Normalize the images to [0,1]
X = X / 255.0

print(f"Loaded {len(X)} images.")

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For autoencoder, we use flow with X as both input and target (reconstruction)
train_generator = datagen.flow(X, X, batch_size=32)

# Define a simple convolutional autoencoder model
# Updated input_shape to (128, 128, 3)
model = Sequential([
    # Encoder
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    
    # Decoder
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(3, (3, 3), activation='sigmoid', padding='same')  # Output reconstructed image
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
# Steps per epoch = number of images // batch_size
steps_per_epoch = len(X) // 32
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10  # Adjust as needed; more epochs for better results
)

# Optional: Visualize original vs reconstructed
# Take a sample image
sample_img = X[0:1]  # Batch of 1
reconstructed = model.predict(sample_img)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(sample_img[0])
axes[0].set_title('Original')
axes[0].axis('off')
axes[1].imshow(reconstructed[0])
axes[1].set_title('Reconstructed')
axes[1].axis('off')
plt.show()

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('autoencoder_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Autoencoder model converted to TFLite and saved as 'autoencoder_model.tflite'")

# Note: This autoencoder can be used for image reconstruction or as a base for generation.
# For more advanced generation, consider training a GAN, but that requires more data and compute.
# The dimension mismatch was due to the original 150x150 size not dividing evenly through the pooling layers;
# changing to 128x128 resolves this as it's divisible by 2^3=8.