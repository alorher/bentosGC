## Carga de datos ##
import pandas as pd
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Load the data from the CSV
csv_path = 'C:/Users/alorh/OneDrive/Escritorio/TFM/Datos/habitat_summary_wide_itemized.csv'
data = pd.read_csv(csv_path)

# Directory containing the images
image_folder = 'C:/Users/alorh/OneDrive/Escritorio/TFM/Datos/output/DenoisingMask/Gran Canaria'

# Lists to store images and labels (porcentajes)
images = []
labels = []

# Iterate over each row in the CSV
for index, row in data.iterrows():
    image_id = row['id']
    image_path = os.path.join(image_folder, f"{image_id}.png")  # Adjust the extension as needed

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is not None:
        # Resize the image to 128x128 pixels
        image = cv2.resize(image, (128, 128))
        # Normalize pixel values to range [0, 1]
        image = image / 255.0
        # Append the processed image to the images list
        images.append(image)

        # Aquí se podrían sacar los valores de las etiquetas (porcentajes de cada tipo de algas) de las filas
        labels.append(row[2:].values)  # Si empiezan por la 3a columna

# Convert the lists of images and labels to NumPy arrays
X = np.array(images)
y = np.array(labels)

# Verify the shapes of the resulting arrays
print(f"X shape: {X.shape}")  # Should be (number of images, 128, 128, 3)
print(f"y shape: {y.shape}")  # Should be (number of images, número de tipos de algas)

## Partición train-test ##
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the shapes of the datasets
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")

## Entrenamiento y evaluación ##
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y_train.shape[1], activation='linear')  # Ajustar según el número de tipos de algas.
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model
loss, mae = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}, Validation MAE: {mae}")
