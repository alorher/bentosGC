import pandas as pd
import time
import os
import psutil
from PIL import Image

# Iniciar el tiempo de ejecución
start_time = time.time()

# Medir el uso inicial de recursos
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024  # Convertir a MB
initial_cpu = process.cpu_percent(interval=None)

# Rutas de entrada y salida
csv_input_path = 'C:/Users/alorh/OneDrive/Escritorio/TFM/Datos/habitat_summary_wide_itemized.csv'
output_dir = 'C:/Users/alorh/OneDrive/Escritorio/TFM/Datos/output/csv/Gran Canaria/'
output_file_name = 'habitat_summary_wide_itemized_etl2.csv'
output_path = os.path.join(output_dir, output_file_name)

# Asegúrate de que el directorio de salida exista
os.makedirs(output_dir, exist_ok=True)

# Leer el archivo CSV
df = pd.read_csv(csv_input_path)

# Transformar la columna id
def transform_id(id):
    parts = id.split('_')
    return f"{parts[0]}_{parts[2]}_{parts[3]}"

df['id'] = df['id'].apply(transform_id)

# Guardar el DataFrame modificado en la nueva ruta
df.to_csv(output_path, index=False)

print(f'Archivo guardado en {output_path}')

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

# Número de imágenes procesadas (asumiendo que cada fila del CSV corresponde a una imagen)
num_images_processed = len(df)

print(f"ESTADISTICAS DEL PROCESO DE ETL DEL FICHERO habitat_summary.csv")
print(f"Número de registros procesados: {num_images_processed}.")
print(f"Tiempo de ejecución: {execution_time} segundos.")
print(f"Tiempo de ejecución: {execution_time_minutes:.2f} minutos.")
print(f"Memoria utilizada: {memory_used:.2f} MB.")
print(f"Uso de CPU: {cpu_used:.2f} %.")


