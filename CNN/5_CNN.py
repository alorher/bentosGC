## Carga de datos ##
import pandas as pd
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Iniciar el tiempo de ejecución
start_time = time.time()

# Medir el uso inicial de recursos
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024  # Convertir a MB
initial_cpu = process.cpu_percent(interval=None)

# Load the data from the CSV
csv_path = 'C:/Users/alorh/OneDrive/Escritorio/TFM/Datos/output/csv/Gran Canaria/habitat_summary_wide_itemized_etl.csv'
data = pd.read_csv(csv_path)

# Directory containing the images
image_folder = 'C:/Users/alorh/OneDrive/Escritorio/TFM/Datos/output/PNG/ImagesSinProcesar/'

# Lists to store images and labels (porcentajes)
images = []
labels = []

# Iterate over each row in the CSV
for index, row in data.iterrows():
    image_id = row['id']
    image_path = os.path.join(image_folder, f"{image_id}.png")  # Adjust the extension as needed
    print(f"Processing image with ID: {image_id}")
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
    else:
        print(f"Warning: Could not read image {image_path}")

# Convert the lists of images and labels to NumPy arrays if they are not empty
if images and labels:
    X = np.array(images)
    y = np.array(labels)

    # Verify the shapes of the resulting arrays
    print(f"X shape: {X.shape}")  # Should be (number of images, 128, 128, 3)
    print(f"y shape: {y.shape}")  # Should be (number of images, número de tipos de algas)

    ## Partición train-val-test ##
    # Split the data into training and temporary sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    # Split the temporary set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Verify the shapes of the resulting arrays
    print(f"X_train shape: {X_train.shape}")  # Should be (60% of number of images, 128, 128, 3)
    print(f"y_train shape: {y_train.shape}")  # Should be (60% of number of images, número de tipos de algas)
    print(f"X_val shape: {X_val.shape}")  # Should be (20% of number of images, 128, 128, 3)
    print(f"y_val shape: {y_val.shape}")  # Should be (20% of number of images, número de tipos de algas)
    print(f"X_test shape: {X_test.shape}")  # Should be (20% of number of images, 128, 128, 3)
    print(f"y_test shape: {y_test.shape}")  # Should be (20% of number of images, número de tipos de algas)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    ## Entrenamiento y evaluación ##

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(y_train.shape[1], activation='linear')
    ])

    # Resumen del modelo
    model.summary()

    # Compila el modelo
    model.compile(optimizer='RMSProp', loss='mae', metrics=['accuracy'])

    model_checkpoint_path = 'C:/Users/alorh/OneDrive/Escritorio/TFM/PruebasPY/YOLOv8_pruebas/modelos/rnn.keras'

    # Callback ModelCheckpoint para guardar el mejor modelo
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )

    # Callback EarlyStopping para detener el entrenamiento si la pérdida de validación no mejora
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )

    model_history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=20,
        shuffle=False,  # Desactiva el barajado de los datos para mantener la secuencialidad
        callbacks=[model_checkpoint_callback, early_stopping_callback]
    )

    # Evaluar el modelo en el conjunto de prueba
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Calcular el tiempo de ejecución
    end_time = time.time()
    execution_time = end_time - start_time

    # Convertir el tiempo de ejecución a minutos
    execution_time_minutes = execution_time / 60

    # Medir el uso final de recursos
    final_memory = process.memory_info().rss / 1024 / 1024  # Convertir a MB
    final_cpu = process.cpu_percent(interval=None)

    # Calcular el uso de recursos
    memory_used = final_memory - initial_memory
    cpu_used = final_cpu - initial_cpu

    # Obtener la época en la que se detuvo el entrenamiento
    stopped_epoch = len(model_history.history['loss'])

    # Obtener la precisión y la pérdida en esa época
    accuracy = model_history.history['accuracy'][stopped_epoch - 1]
    loss = model_history.history['loss'][stopped_epoch - 1]

    # Mostrar el mensaje con la información recopilada
    print(f"-----------------------------------------------")
    print(f"RESULTADO DEL ENTRENAMIENTO")
    print(f"El entrenamiento se detuvo en la época {stopped_epoch} con una precisión de {accuracy:.4f} y una pérdida de {loss:.4f}.")

    # Mostrar el mensaje de finalización y el uso de recursos
    print(f"-----------------------------------------------")
    print(f"ESTADISTICAS DEL PROCESO DE CREACIÓN DEL MODELO")
    print(f"Proceso finalizado. Número de imágenes procesadas: {len(X)}")
    print(f"Tiempo de ejecución: {execution_time} segundos.")
    print(f"Tiempo de ejecución: {execution_time_minutes:.2f} minutos.")
    print(f"Memoria utilizada: {memory_used:.2f} MB.")
    print(f"Uso de CPU: {cpu_used:.2f} %.")

    # Graficar accuracy y loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(model_history.history['accuracy'], label='Training Accuracy')
    plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(model_history.history['loss'], label='Training Loss')
    plt.plot(model_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Generar un diagrama de cajas de los errores absolutos en las predicciones
    y_pred = model.predict(X_test)
    errors = np.abs(y_pred - y_test)

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=errors)
    plt.title('Box Plot of Absolute Errors in Predictions')
    plt.xlabel('Algae Types')
    plt.ylabel('Absolute Error')
    plt.xticks(rotation=45)
    plt.show()

else:
    print("Error: No images were loaded. Check the file paths and ensure the images are available.")

