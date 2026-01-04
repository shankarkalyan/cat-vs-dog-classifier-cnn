import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np

# --- Step 1: robust Download & Extraction (Local) ---

# We download the file directly to the current directory (/content)
# giving us full control over where it lands.
url = 'https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip'
filename = 'cats_and_dogs_filtered.zip'

# Download the file if it doesn't exist
if not os.path.exists(filename):
    print("Downloading dataset...")
    tf.keras.utils.get_file(filename, origin=url, cache_dir='.', cache_subdir='.')

# Force extraction (using Python's zipfile for transparency)
print("Extracting files...")
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(".") # Extracts to /content/cats_and_dogs_filtered

# Define the local directories
base_dir = 'cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# DEBUG: Print what we found to ensure it worked
if os.path.exists(train_dir):
    print(f"\nSUCCESS! Found training directory: {train_dir}")
    print(f"Contents: {os.listdir(train_dir)}")
else:
    print(f"\nERROR: Still cannot find {train_dir}")
    # Stop here if files are missing
    raise FileNotFoundError("Extraction failed")

# --- Step 2: Data Preprocessing ---

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)

validation_image_generator = ImageDataGenerator(rescale=1./255)

print("\nLoading Generators:")
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

# --- Step 3: Model Building ---

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# --- Step 4: Training ---

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\nStarting Training...")
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // batch_size
)

# --- Step 5: Visualization ---

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
