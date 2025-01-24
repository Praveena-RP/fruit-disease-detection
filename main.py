import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load images and labels from the given folder
def load_images_from_folder(folder, label, img_size=(100, 100)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)  # Resize to a fixed size
                images.append(img)
                labels.append(label)  # Label the images (0 for healthy, 1 for diseased)
    return np.array(images), np.array(labels)

# Define paths to the folders
healthy_train_folder = 'C:/Users/PRAVEENA/Downloads/fruit disease/fruit disease/train'
diseased_train_folder = 'C:/Users/PRAVEENA/Downloads/fruit disease/fruit disease/train1'
healthy_valid_folder = 'C:/Users/PRAVEENA/Downloads/fruit disease/fruit disease/valid'
diseased_valid_folder = 'C:/Users/PRAVEENA/Downloads/fruit disease/fruit disease/valid1'

# Load images from both healthy and diseased folders
train_healthy_images, train_healthy_labels = load_images_from_folder(healthy_train_folder, label=0)
train_diseased_images, train_diseased_labels = load_images_from_folder(diseased_train_folder, label=1)

valid_healthy_images, valid_healthy_labels = load_images_from_folder(healthy_valid_folder, label=0)
valid_diseased_images, valid_diseased_labels = load_images_from_folder(diseased_valid_folder, label=1)

# Combine healthy and diseased images for training and validation
train_images = np.concatenate((train_healthy_images, train_diseased_images), axis=0)
train_labels = np.concatenate((train_healthy_labels, train_diseased_labels), axis=0)

valid_images = np.concatenate((valid_healthy_images, valid_diseased_images), axis=0)
valid_labels = np.concatenate((valid_healthy_labels, valid_diseased_labels), axis=0)

# Normalize the images (scale pixel values between 0 and 1)
train_images = train_images / 255.0
valid_images = valid_images / 255.0

# Define the CNN model
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    return model

# Prepare the input shape based on image size
input_shape = train_images.shape[1:]  # Shape of (height, width, channels)

# Create the model
model = create_cnn_model(input_shape)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary Crossentropy for binary classification
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels,
                    epochs=10, 
                    batch_size=32,
                    validation_data=(valid_images, valid_labels))

# Evaluate the model on the validation set
valid_loss, valid_accuracy = model.evaluate(valid_images, valid_labels)
print(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")

# Save the model if training is successful
model.save('fruit_disease_model.h5')  # Save the trained model
