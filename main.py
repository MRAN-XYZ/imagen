import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Assuming your custom dataset is organized in a directory structure like:
# dataset/
#   class1/
#     img1.jpg
#     img2.jpg
#     ...
#   class2/
#     img3.jpg
#     ...
# Replace 'path/to/your/dataset' with the actual path to your dataset folder.
# If your dataset isn't classified (no subdirectories), use flow_from_dataframe or load images manually.

# Create an ImageDataGenerator instance with optional data augmentation parameters.
# You can customize these based on your needs (e.g., add rotation, zoom, etc.).
datagen = ImageDataGenerator(
    rescale=1.0/255,          # Normalize pixel values to [0,1]
    rotation_range=20,        # Random rotation in degrees
    width_shift_range=0.2,    # Horizontal shift
    height_shift_range=0.2,   # Vertical shift
    shear_range=0.2,          # Shear transformation
    zoom_range=0.2,           # Random zoom
    horizontal_flip=True,     # Random horizontal flip
    fill_mode='nearest',      # Fill mode for new pixels
    validation_split=0.2      # Optional: split for training/validation
)

# Create a generator for training data.
# Adjust target_size to match your image dimensions, batch_size as needed.
# class_mode can be 'categorical', 'binary', 'sparse', or None depending on your labels.
train_generator = datagen.flow_from_directory(
    './Logos',   # Replace with your dataset path
    target_size=(150, 150),   # Resize images to this size
    batch_size=32,
    class_mode='categorical', # Assuming multi-class classification; change if needed
    subset='training'         # Use 'training' subset if validation_split is used
)

# Optional: Validation generator if using validation_split
validation_generator = datagen.flow_from_directory(
    'path/to/your/dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Determine the number of classes from the generator
num_classes = len(train_generator.class_indices)

# Define a simple CNN model for image classification
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer for multi-class
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model using the generators
# Adjust epochs and steps as needed based on your dataset size
history = model.fit(
    train_generator,
    epochs=10,  # Increase for better training; monitor for overfitting
    validation_data=validation_generator
)

# Optional: Evaluate the model
# test_loss, test_acc = model.evaluate(validation_generator)
# print(f'Test accuracy: {test_acc}')

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite and saved as 'model.tflite'")

# Optional: To use the TFLite model for inference, you can load it like this:
# interpreter = tf.lite.Interpreter(model_path='model.tflite')
# interpreter.allocate_tensors()
# ... (further inference code)
