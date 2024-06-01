import os
import psutil
import time

# Iniciar el tiempo de ejecución
start_time = time.time()

# Medir el uso inicial de recursos
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024  # Convertir a MB
initial_cpu = process.cpu_percent(interval=None)

# Define la ruta de la carpeta
folder_path = 'C:/Users/alorh/OneDrive/Escritorio/TFM/Datos/output/PNG/GC/DenoisingMask5'

# El texto que se va a eliminar al principio de cada nombre de archivo
text_to_remove = "Mask_Cleaned_30_"

# Inicializar un contador para el número de archivos renombrados
num_files_renamed = 0

# Lista todos los archivos en la carpeta
for filename in os.listdir(folder_path):
    # Verifica que el nombre del archivo comience con el texto a eliminar
    if filename.startswith(text_to_remove):
        # Define la nueva ruta del archivo con el nombre modificado
        new_filename = filename[len(text_to_remove):]
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)

        # Renombra el archivo
        os.rename(old_file, new_file)
        num_files_renamed += 1
        print(f'Renamed: {filename} to {new_filename}')
    else:
        print(f'Skipped: {filename} (does not start with "{text_to_remove}")')

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

# Imprimir el número de archivos renombrados y el uso de recursos
print(f"ESTADISTICAS DEL PROCESO DE CONVERSION DE CORRECCIÓN DE NOMBRE EN LAS MÁSCARAS GENERADAS")
print(f"Se han renombrado 787 imagenes.")
print(f"Tiempo de ejecución: {execution_time} segundos.")
print(f"Tiempo de ejecución: {execution_time_minutes:.2f} minutos.")
print(f"Memoria utilizada: {memory_used:.2f} MB.")
print(f"Uso de CPU: {cpu_used:.2f} %.")

