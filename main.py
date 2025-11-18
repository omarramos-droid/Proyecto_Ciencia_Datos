from Funciones.data_loader import load_heart_disease_data, get_model_data
import pandas as pd

def main():
    """
    Función principal del proyecto de ciencia de datos
    """
    print("=== PROYECTO CIENCIA DE DATOS - ENFERMEDAD CARDÍACA ===\n")
    
    # 1. Cargar datos
    print("1. Cargando datos...")
    df = load_heart_disease_data()
    
    if df.empty:
        print("Error: No se pudieron cargar los datos")
        return
    
    print(f"Datos cargados exitosamente: {len(df)} registros\n")
    
    # 2. Preparar datos para el modelo
    print("2. Preparando datos para el modelo...")
    df_model = get_model_data(df, include_extra=True)
    
    # 3. Mostrar información general
    print("\n3. Información del dataset:")
    print(f"   - Total de registros: {len(df_model)}")
    print(f"   - Total de características: {len(df_model.columns)}")
    print(f"   - Columnas: {df_model.columns.tolist()}")
    
    # 4. Información sobre la variable objetivo 'num'
    if 'num' in df_model.columns:
        print(f"\n4. Distribución de la variable objetivo 'num':")
        print(df_model['num'].value_counts().sort_index())
    
    # 5. Información sobre datasets si está disponible
    if 'dataset' in df_model.columns:
        print(f"\n5. Distribución por dataset origen:")
        print(df_model['dataset'].value_counts())
    
    print("\n=== PROCESO COMPLETADO ===")

if __name__ == "__main__":
    main()