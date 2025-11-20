import pandas as pd
import numpy as np
from pathlib import Path

def load_cleveland_data():
    """
    Carga los datos crudos del dataset de Cleveland
    """
    # Configurar rutas de archivos
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "Data" / "heart+disease"
    
    # Nombres de todas las columnas según la documentación
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
    

    
    file_path = data_dir / 'cleveland.data'
    
    try:
        # Leer y tokenizar el archivo
        with open(file_path, 'r') as f:
            tokens = []
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    tokens.extend(clean_line.split())
        
        # Agrupar tokens en registros de 76 columnas
        records = []
        current_record = []
        
        for token in tokens:
            current_record.append(token)
            if len(current_record) == 76:
                records.append(current_record)
                current_record = []
        
        if records:
            df = pd.DataFrame(records, columns=column_names)
            df['dataset'] = 'cleveland'
            print(f"{len(df)} registros cargados exitosamente")
            return df
        else:
            print("✗ No se encontraron registros válidos")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"✗ Error en lectura: {e}")
        return pd.DataFrame()

def filter_data_quality(df):
    """
    Filtra los datos aplicando criterios de intervalos
    """

    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # Convertir columnas a numéricas (maneja '?' como NaN)
    numeric_columns = [col for col in df_clean.columns if col not in ['dataset', 'name']]
    
    for col in numeric_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Reemplazar valores missing (-9) con NaN
    df_clean = df_clean.replace(-9, np.nan)
    
    # Criterios de calidad para variables importantes
    quality_criteria = {
        'age': {'min': 10, 'max': 90},
        'sex': {'values': [0, 1]},
        'cp': {'values': [1, 2, 3, 4]},
        'trestbps': {'min': 80, 'max': 200},
        'chol': {'min': 30, 'max': 600},
        'fbs': {'values': [0, 1]},
        'restecg': {'values': [0, 1, 2]},
        'thalach': {'min': 60, 'max': 220},
        'exang': {'values': [0, 1]},
        'oldpeak': {'min': 0, 'max': 6.2},
        'slope': {'values': [1, 2, 3]},
        'ca': {'values': [0, 1, 2, 3]},
        'thal': {'values': [3, 6, 7]},
        'num': {'values': [0, 1, 2, 3, 4]}
    }
    
    # Aplicar filtros de calidad
    valid_mask = pd.Series([True] * len(df_clean))
    
    for col, criteria in quality_criteria.items():
        if col in df_clean.columns:
            col_mask = pd.Series([True] * len(df_clean))
            
            if 'values' in criteria:
                col_mask = df_clean[col].isin(criteria['values']) | df_clean[col].isna()
            elif 'min' in criteria and 'max' in criteria:
                col_mask = ((df_clean[col] >= criteria['min']) & 
                           (df_clean[col] <= criteria['max'])) | df_clean[col].isna()
            
            invalid_count = (~col_mask).sum()
            # if invalid_count > 0:
            #     print(f"   {col}: {invalid_count} valores fuera de rango eliminados")
            
            valid_mask = valid_mask & col_mask
    
    # Aplicar filtro final
    df_filtered = df_clean[valid_mask].copy()
    

    return df_filtered

def select_relevant_columns(df):
    """
    Selecciona las columnas más relevantes basado en datos faltantes e importancia clínica
    """
  
    
    # Columnas esenciales - las 14 variables principales
    essential_columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    
    # Columnas a eliminar definitivamente
    columns_to_drop = [
        'thalsev', 'thalpul', 'earlobe', 'restckm', 'exerckm',
        'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk',
        'id', 'ccf', 'name', 'ekgmo', 'ekgday', 'ekgyr', 
        'cmo', 'cday', 'cyr', 'dummy', 'smoke', 'cigs'  # smoke y cigs eliminadas
    ]

    # Columnas adicionales para evaluar
    columns_to_evaluate = ['htn', 'years', 'dm', 'famhist', 'thalrest']
    
    # Evaluar columnas adicionales basado en datos faltantes
    columns_to_keep = []
    
    for col in columns_to_evaluate:
        if col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct <= 10:
                columns_to_keep.append(col)
    
    # Combinar todas las columnas a mantener
    final_columns = essential_columns + columns_to_keep + ['dataset']
    final_columns = list(set(final_columns) & set(df.columns))
    
    # Crear DataFrame final
    df_final = df[final_columns].copy()
    
    return df_final

def impute_missing_values(df):
    """
    Imputa los valores faltantes usando estrategias específicas
    """

    
    df_imputed = df.copy()
    
    # print("Valores faltantes antes de la imputación:")
    missing_before = df_imputed.isna().sum()
    # print(missing_before[missing_before > 0])
    
    # Estrategia de imputación
    # Variables categóricas: imputar por moda
    if 'ca' in df_imputed.columns:
        ca_mode = df_imputed['ca'].mode()[0]
        df_imputed['ca'].fillna(ca_mode, inplace=True)
        # print(f"   ca: 2 valores imputados con moda ({ca_mode})")
    
    if 'thal' in df_imputed.columns:
        thal_mode = df_imputed['thal'].mode()[0]
        df_imputed['thal'].fillna(thal_mode, inplace=True)
        # print(f"   thal: 2 valores imputados con moda ({thal_mode})")
    
    # Imputar con 0
    if 'years' in df_imputed.columns:
        df_imputed['years'].fillna(0, inplace=True)
        # print(f"   years: 5 valores imputados con 0 años)")
    
    # Verificación
    missing_after = df_imputed.isna().sum().sum()
    # print(f" Valores faltantes restantes: {missing_after}")
    
    return df_imputed


def save_processed_data(df, filename="cleveland_clean.csv"):
    """
    Guarda el DataFrame procesado
    """
    output_path = Path(__file__).parent / filename
    df.to_csv(output_path, index=False)
    return output_path

def main_data():
    """
    Función principal para cargar los datos y filtrar por criterior , seleccion de columnas
    Y la imputación de valores mediantes criterios individuales
    """
    # 1. Cargar datos crudos
    df_raw = load_cleveland_data()
 
    
    # 2. Filtrar por calidad clínica
    df_filtered = filter_data_quality(df_raw)
    
    # 3. Seleccionar columnas relevantes
    df_selected = select_relevant_columns(df_filtered)
    
    # 4. Imputar valores faltantes
    df_imputed = impute_missing_values(df_selected)
    

    return df_imputed

