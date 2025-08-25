import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

dataset_path = '/content/imagen/Logos'
image_files = glob.glob(os.path.join(dataset_path, '*.[jp][pn]g'))
images = []
for file in image_files:
    img = load_img(file, target_size=(128, 128))
    img_array = img_to_array(img)
    images.append(img_array)

X = np.array(images)
if len(X) == 0:
    raise ValueError("No images found. Check your dataset_path and image extensions.")

X = X / 255.0
print(f"Loaded {len(X)} images.")

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = datagen.flow(X, X, batch_size=32)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(3, (3, 3), activation='sigmoid', padding='same')
])
model.compile(optimizer='adam', loss='mean_squared_error')

steps_per_epoch = max(1, len(X) // 32)
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10
)

sample_img = X[0:1]
reconstructed = model.predict(sample_img)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(sample_img[0])
axes[0].set_title('Original')
axes[0].axis('off')
axes[1].imshow(reconstructed[0])
axes[1].set_title('Reconstructed')
axes[1].axis('off')
plt.show()

# Save the model in TensorFlow SavedModel format
model.save('autoencoder_saved_model.keras')
print("Model saved in TensorFlow SavedModel format at 'autoencoder_saved_model.keras'")