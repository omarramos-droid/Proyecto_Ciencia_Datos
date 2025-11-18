import pandas as pd
import os
from pathlib import Path

def load_heart_disease_data(data_dir=None):
    """
    Carga y combina todos los datasets de heart disease
    
    Parameters:
    data_dir (str): Directorio donde están los archivos .data
    
    Returns:
    pd.DataFrame: DataFrame combinado con todos los datos
    """
    
    # Si no se especifica directorio, usar el predeterminado
    if data_dir is None:
        data_dir = Path("C:/Users/dell/Desktop/Proyecto_Ciencia_Datos/Data/heart+disease/")
    else:
        data_dir = Path(data_dir)
    
    # Nombres de las columnas (definidos por el repositorio UCI)
    column_names = [
        "id", "ccf", "age", "sex", "painloc", "painexer", "relrest", "pncaden",
        "cp", "trestbps", "htn", "chol", "smoke", "cigs", "years", "fbs", "dm", "famhist",
        "restecg", "ekgmo", "ekgday", "ekgyr", "dig", "prop", "nitr", "pro", "diuretic",
        "proto", "thaldur", "thaltime", "met", "thalach", "thalrest", "tpeakbps",
        "tpeakbpd", "dummy", "trestbpd", "exang", "xhypo", "oldpeak", "slope",
        "rldv5", "rldv5e", "ca", "restckm", "exerckm", "restef", "restwm", "exeref",
        "exerwm", "thal", "thalsev", "thalpul", "earlobe", "cmo", "cday", "cyr", "num",
        "lmt", "ladprox", "laddist", "diag", "cxmain", "ramus", "om1", "om2", "rcaprox",
        "rcadist", "lvx1", "lvx2", "lvx3", "lvx4", "lvf", "cathef", "junk", "name"
    ]
    
    # Archivos a cargar
    data_files = {
        'cleveland': 'cleveland.data',
        'hungarian': 'hungarian.data', 
        'long-beach': 'long-beach-va.data',
        'switzerland': 'switzerland.data'
    }
    
    dataframes = []
    
    for dataset_name, filename in data_files.items():
        file_path = data_dir / filename
        
        if not file_path.exists():
            print(f"Advertencia: No se encontró {file_path}")
            continue
            
        print(f"Cargando {dataset_name}...")
        
        try:
            # Leer todos los tokens del archivo
            tokens = []
            with open(file_path, 'r') as f:
                for line in f:
                    # Limpiar y dividir la línea
                    cleaned_line = line.strip()
                    if cleaned_line:
                        tokens.extend(cleaned_line.split())
            
            # Agrupar tokens en registros (cada registro tiene 76 columnas)
            records = []
            current_record = []
            
            for token in tokens:
                current_record.append(token)
                if len(current_record) == 76:
                    records.append(current_record)
                    current_record = []
            
            # Crear DataFrame
            if records:
                df_temp = pd.DataFrame(records, columns=column_names)
                df_temp['dataset'] = dataset_name  # Agregar columna para identificar el origen
                dataframes.append(df_temp)
                print(f"  {len(df_temp)} registros cargados")
            else:
                print(f"  No se pudieron procesar registros para {dataset_name}")
                
        except Exception as e:
            print(f"Error cargando {dataset_name}: {e}")
    
    # Combinar todos los DataFrames
    if dataframes:
        df_combined = pd.concat(dataframes, ignore_index=True)
        print(f"\nTotal de registros combinados: {len(df_combined)}")
        
        # Procesamiento de datos
        df_processed = preprocess_data(df_combined)
        return df_processed
    else:
        print("Error: No se pudieron cargar datos de ningún archivo")
        return pd.DataFrame()

def preprocess_data(df):
    """
    Preprocesa los datos: convierte a numérico y maneja valores missing
    
    Parameters:
    df (pd.DataFrame): DataFrame raw combinado
    
    Returns:
    pd.DataFrame: DataFrame procesado
    """
    print("\nPreprocesando datos...")
    
    # Hacer una copia para no modificar el original
    df_processed = df.copy()
    
    # Convertir columnas a numérico (excepto 'name' y 'dataset')
    non_numeric_cols = ['name', 'dataset']
    numeric_cols = [col for col in df_processed.columns if col not in non_numeric_cols]
    
    for col in numeric_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        # Reemplazar -9 (valor que indica missing en estos datasets) por NaN
        df_processed[col] = df_processed[col].replace(-9, pd.NA)
    
    print("Conversión a numérico completada")
    return df_processed

def get_model_data(df, include_extra=False):
    """
    Extrae las columnas relevantes para el modelo
    
    Parameters:
    df (pd.DataFrame): DataFrame procesado
    include_extra (bool): Si incluir columnas adicionales
    
    Returns:
    pd.DataFrame: DataFrame con columnas para el modelo
    """
    
    # Columnas base (las más comunes en la literatura)
    cols_base = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    
    # Columnas adicionales (opcionales)
    cols_extra = ["smoke", "cigs", "years", "famhist", "dataset"]
    
    # Verificar qué columnas existen
    if include_extra:
        cols_final = [c for c in (cols_base + cols_extra) if c in df.columns]
    else:
        cols_final = [c for c in cols_base if c in df.columns]
    
    # Eliminar columnas que estén completamente vacías
    cols_final = [col for col in cols_final if not df[col].isna().all()]
    
    # Crear DataFrame final
    df_model = df[cols_final].copy()
    
    print(f"Columnas finales del modelo: {df_model.columns.tolist()}")
    print(f"\nNAs por columna:")
    print(df_model.isna().sum())
    
    return df_model

# Función principal para testing
if __name__ == "__main__":
    # Cargar datos
    df = load_heart_disease_data()
    
    if not df.empty:
        print(f"\nDataset cargado: {len(df)} registros")
        print(f"Columnas: {df.columns.tolist()}")
        
        # Obtener datos para modelo
        df_model = get_model_data(df, include_extra=True)
        print(f"\nPrimeras filas del dataset del modelo:")
        print(df_model.head())
        
        # Información sobre los datasets
        print(f"\nDistribución por dataset:")
        if 'dataset' in df.columns:
            print(df['dataset'].value_counts())
    else:
        print("No se pudieron cargar los datos")