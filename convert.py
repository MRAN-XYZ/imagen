import tensorflow as tf

# Replace 'model.keras' with your actual file path
model = keras.models.load_model('logimagen.keras')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)