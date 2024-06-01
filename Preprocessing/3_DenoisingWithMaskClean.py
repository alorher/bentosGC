import os
import cv2
import time
import psutil
import matplotlib.pyplot as plt


from glob import glob
from skimage.morphology import disk, closing
from skimage.filters import rank
from skimage.util import img_as_ubyte

# Iniciar el tiempo de ejecución
start_time = time.time()

# Medir el uso inicial de recursos
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024  # Convertir a MB
initial_cpu = process.cpu_percent(interval=None)


# Ruta a la carpeta de imágenes
image_dir_png = 'C:/Users/alorh/OneDrive/Escritorio/TFM/Datos/Detection/images/train'

# Crear la carpeta de salida si no existe
output_dir_processed= 'C:/Users/alorh/OneDrive/Escritorio/TFM/Datos/Detection/images/train/denoising'
if not os.path.exists(output_dir_processed):
    os.makedirs(output_dir_processed)

# Obtener la lista de rutas de todas las imágenes en la carpeta
image_paths = glob(os.path.join(image_dir_png, '*.png'))

# Definir diferentes tamaños de disco
disk_sizes = [30]

# Contador de imágenes procesadas
processed_count = 0

# Procesar cada imagen
for image_path in image_paths:
    # Cargar la imagen original (en color)
    image_original = cv2.imread(image_path)

    # Cargar la imagen en escala de grises
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if image_gray is None:
        print(f"Error al cargar la imagen: {image_path}")
        continue

    # 1. CLAHE
    # Convertir la imagen original en gris a formato de 8 bits
    image_gray_ubyte = img_as_ubyte(image_gray)

    # Crear el objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Aplicar CLAHE a la imagen
    clahe_image = clahe.apply(image_gray_ubyte)

    # Convertir la imagen procesada con CLAHE a formato de 8 bits
    clahe_image_ubyte = img_as_ubyte(clahe_image)

    # 2. Aplicar CMF y Closing para cada tamaño de disco
    for size in disk_sizes:
        selem = disk(size)

        # Aplicar el filtro mediano circular (CMF)
        cmf_image = rank.median(clahe_image_ubyte, selem)

        # Aplicar el proceso de closing
        cmf_closing_image = closing(cmf_image, selem)

        # 3. Aplicar umbral (thresholding) para crear la máscara
        ret, mask = cv2.threshold(cmf_closing_image, 127, 255, cv2.THRESH_BINARY)

        # 4. Limpiar la máscara utilizando operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

        # Generar la ruta de salida manteniendo la estructura de carpetas
        relative_path = os.path.relpath(os.path.dirname(image_path), image_dir_png)
        output_subdir = os.path.join(output_dir_processed, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        # Guardar la imagen con CMF y closing aplicado
        image_name = os.path.basename(image_path)
        cmf_closing_image_path = os.path.join(output_subdir,
                                              f'CMF_Closing_{size}_{os.path.splitext(image_name)[0]}.png')
        cv2.imwrite(cmf_closing_image_path, cmf_closing_image)

        # Guardar la máscara limpia
        mask_cleaned_path = os.path.join(output_subdir, f'Mask_Cleaned_{size}_{os.path.splitext(image_name)[0]}.png')
        cv2.imwrite(mask_cleaned_path, mask_cleaned)

    # Incrementar el contador de imágenes procesadas
    processed_count += 1

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


# Mostrar el mensaje de finalización
print(f"ESTADISTICAS DEL PROCESO DE DENOISING")
print(f"Proceso finalizado. Número de imágenes procesadas: 787")
print(f"Tiempo de ejecución: {execution_time} segundos.")
print(f"Tiempo de ejecución: {execution_time_minutes:.2f} minutos.")
print(f"Memoria utilizada: {memory_used:.2f} MB.")
print(f"Uso de CPU: {cpu_used:.2f} %.")

