from Funciones.data_loader import *
import pandas as pd

def main():
    """
    Función principal que usa el módulo de procesamiento
    """
    # 1. Obtener datos procesados
    print("Cargando y procesando datos de Cleveland...")
    df = get_cleveland_data()
    
    # 2. Ver información del dataset
    info = get_data_info(df)
    print(f"\nDataset procesado:")
    print(f"- Filas: {info['filas']}")
    print(f"- Columnas: {info['columnas']}")
    print(f"- Columnas: {info['columnas_lista']}")
    
    # 3. Guardar datos procesados (opcional)
    output_path = save_processed_data(df, "mi_dataset_cleveland.csv")
    print(f"\nDatos guardados en: {output_path}")
    
    # 4. Aquí puedes continuar con tu análisis...
    # Ejemplo: análisis exploratorio, entrenamiento de modelos, etc.
    print("\nDataset listo para análisis!")
    
    return df

if __name__ == "__main__":
    data = main()
    data['num'].value_counts()