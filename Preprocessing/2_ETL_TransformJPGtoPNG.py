import os
import time
import psutil
from PIL import Image

# Iniciar el tiempo de ejecución
start_time = time.time()

# Medir el uso inicial de recursos
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024  # Convertir a MB
initial_cpu = process.cpu_percent(interval=None)

# Ruta raíz de las imágenes JPG
image_dir_jpg = 'C:/Users/alorh/OneDrive/Escritorio/TFM/Datos/output/JPG/ETL/GC'

# Ruta raíz para guardar las imágenes PNG
output_dir_png = 'C:/Users/alorh/OneDrive/Escritorio/TFM/Datos/output/PNG/GC/pics3'

# Crea la carpeta de salida si no existe
if not os.path.exists(output_dir_png):
    os.makedirs(output_dir_png)

# Inicializa un contador para el número de imágenes convertidas
num_images_converted = 0

# Recorre todas las subcarpetas y archivos
for root, dirs, files in os.walk(image_dir_jpg):
    for file in files:
        if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.JPG'):
            # Ruta completa del archivo de entrada
            input_file_path = os.path.join(root, file)

            # Crear la ruta de salida correspondiente manteniendo la estructura de carpetas
            relative_path = os.path.relpath(root, image_dir_jpg)
            output_subdir = os.path.join(output_dir_png, relative_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            # Ruta completa del archivo de salida
            output_file_path = os.path.join(output_subdir, os.path.splitext(file)[0] + '.png')

            # Cargar la imagen
            image = Image.open(input_file_path)

            # Guardar la imagen en formato .png
            image.save(output_file_path, 'PNG')
            num_images_converted += 1

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

# Imprimir el número de imágenes convertidas y el uso de recursos
print(f"ESTADISTICAS DEL PROCESO DE CONVERSION DE IMÁGENES JPG A PNG")
print(f"Conversión completa. Se han convertido {num_images_converted} imágenes.")
print(f"Tiempo de ejecución: {execution_time} segundos.")
print(f"Tiempo de ejecución: {execution_time_minutes:.2f} minutos.")
print(f"Memoria utilizada: {memory_used:.2f} MB.")
print(f"Uso de CPU: {cpu_used:.2f} %.")
