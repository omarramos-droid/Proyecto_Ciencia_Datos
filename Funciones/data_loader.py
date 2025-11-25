
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
def load_cleveland_data():
    """
    Carga los datos crudos del dataset de Cleveland
    """
    
    # Configurar rutas de archivos
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "Data" / "heart+disease"
    
    # Nombres de todas las columnas
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
    
    # Leer el archivo
    with open(file_path, 'r') as archivo:
        lineas = archivo.readlines()
    
    # Procesar todas las líneas
    todos = []
    for linea in lineas:
        linea_limpia = linea.strip()
        if linea_limpia:
            tokens_linea = linea_limpia.split()
            todos.extend(tokens_linea)
    
    # Crear registros de 76 columnas
    registros = []
    registro_actual = []
    
    for token in todos:
        registro_actual.append(token)
        if len(registro_actual) == 76:
            registros.append(registro_actual)
            registro_actual = []
    
    # Crear DataFrame
    df = pd.DataFrame(registros, columns=column_names)
    df['dataset'] = 'cleveland'
    
    print(f"{len(df)} registros cargados ")
    return df
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
    ca_mode = df_imputed['ca'].mode()
    df_imputed['ca'] = df_imputed['ca'].fillna(ca_mode.iloc[0])
    # print(f"   ca: 2 valores imputados con moda ({ca_mode.iloc[0]})")
    
    thal_mode = df_imputed['thal'].mode()
    df_imputed['thal'] = df_imputed['thal'].fillna(thal_mode.iloc[0])
    # print(f"   thal: 2 valores imputados con moda ({thal_mode.iloc[0]})")
    
    # Imputar con 0
    df_imputed['years'] = df_imputed['years'].fillna(0)
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

def visualizar_outliers(df):
    """
    Visualización de outliers usando boxplots y scatter plots
    """
    # Boxplots para variables numéricas
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    
    for i, col in enumerate(numeric_cols[:6]):
        if col in df.columns:
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'Boxplot de {col}')
    
    plt.tight_layout()
    plt.show()
    
    # Scatter plots para relaciones clave
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Colesterol vs Edad
    axes[0].scatter(df['age'], df['chol'], alpha=0.6)
    axes[0].set_xlabel('Edad')
    axes[0].set_ylabel('Colesterol')
    axes[0].set_title('Colesterol vs Edad')
    
    # Presión arterial vs Frecuencia cardíaca máxima
    axes[1].scatter(df['trestbps'], df['thalach'], alpha=0.6)
    axes[1].set_xlabel('Presión Arterial en Reposo')
    axes[1].set_ylabel('Frecuencia Cardíaca Máxima')
    axes[1].set_title('Presión vs Frecuencia Cardíaca')
    
    plt.tight_layout()
    plt.show()

data=main_data()

visualizar_outliers(data)
# Función principal 
if __name__ == "__main__":
    df_cleveland = main_data()
    # print( df_cleveland.columns)