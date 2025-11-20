import pandas as pd
import numpy as np
from pathlib import Path

def load_and_clean_data():
    """
    Carga y limpia los datos replicando exactamente el procesamiento
    del archivo processed.cleveland.data
    """
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "Data" / "heart+disease"
    
    # Columnas completas
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
    
    # Solo Cleveland dataset por ahora
    datasets = {
        'cleveland': 'cleveland.data'
    }
    
    all_data = []
    
    print("=" * 70)
    print("ETAPA 1: CARGA INICIAL DE DATOS")
    print("=" * 70)
    
    for name, file in datasets.items():
        file_path = data_dir / file
        print(f"\nLeyendo archivo: {file}")
        
        try:
            with open(file_path, 'r') as f:
                tokens = []
                for line in f:
                    clean_line = line.strip()
                    if clean_line:
                        tokens.extend(clean_line.split())
            
            # Agrupar en registros de 76 columnas
            records = []
            current_record = []
            
            for token in tokens:
                current_record.append(token)
                if len(current_record) == 76:
                    records.append(current_record)
                    current_record = []
            
            if records:
                df = pd.DataFrame(records, columns=column_names)
                df['dataset'] = name
                all_data.append(df)
                print(f"  ✓ {len(df)} registros cargados exitosamente")
            else:
                print(f"  ✗ No se encontraron registros válidos")
                
        except Exception as e:
            print(f"  ✗ Error en lectura: {e}")
    
    if not all_data:
        print("ERROR: No se pudieron cargar datos de ningún archivo")
        return pd.DataFrame()
    
    # Combinar datasets
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\nRESUMEN CARGA INICIAL:")
    print(f"Total de registros: {len(df_combined)}")
    print(f"Total de columnas: {len(df_combined.columns)}")
    
    return df_combined

def clean_to_processed_format(df):
    """
    Limpia los datos manteniendo TODAS las columnas y solo eliminando filas
    que no cumplan con los criterios de calidad
    """
    print("\n" + "=" * 70)
    print("ETAPA 2: LIMPIEZA MANTENIENDO TODAS LAS COLUMNAS")
    print("=" * 70)
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # 1. CONVERTIR COLUMNAS A NUMÉRICAS (manejar '?' como NaN)
    print("\n1. Convirtiendo columnas a numéricas...")
    numeric_columns = [col for col in df_clean.columns if col not in ['dataset', 'name']]
    
    for col in numeric_columns:
        # Convertir, manejando '?' y otros valores no numéricos
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 2. REEMPLAZAR VALORES MISSING (-9) CON NaN
    print("\n2. Reemplazando valores missing (-9) con NaN...")
    df_clean = df_clean.replace(-9, np.nan)
    
    # 3. DEFINIR CRITERIOS DE CALIDAD PARA LAS VARIABLES CLAVE
    print("\n3. Aplicando criterios de calidad para variables clave:")
    
    quality_criteria = {
        'age': {'min': 29, 'max': 77, 'description': 'edad razonable'},
        'sex': {'values': [0, 1], 'description': '0=female, 1=male'},
        'cp': {'values': [1, 2, 3, 4], 'description': 'tipos de dolor torácico'},
        'trestbps': {'min': 80, 'max': 200, 'description': 'presión arterial razonable'},
        'chol': {'min': 100, 'max': 600, 'description': 'colesterol razonable'},
        'fbs': {'values': [0, 1], 'description': 'azúcar en sangre en ayunas'},
        'restecg': {'values': [0, 1, 2], 'description': 'resultados ECG en reposo'},
        'thalach': {'min': 60, 'max': 220, 'description': 'frecuencia cardíaca máxima'},
        'exang': {'values': [0, 1], 'description': 'angina inducida por ejercicio'},
        'oldpeak': {'min': 0, 'max': 6.2, 'description': 'depresión ST'},
        'slope': {'values': [1, 2, 3], 'description': 'pendiente segmento ST'},
        'ca': {'values': [0, 1, 2, 3], 'description': 'vasos coloreados'},
        'thal': {'values': [3, 6, 7], 'description': '3=normal, 6=fixed, 7=reversible'},
        'num': {'values': [0, 1, 2, 3, 4], 'description': 'diagnóstico enfermedad'}
    }
    
    # 4. APLICAR FILTROS SOLO PARA LAS VARIABLES CLAVE
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
            if invalid_count > 0:
                print(f"   {col}: {invalid_count} valores fuera de rango eliminados")
            
            valid_mask = valid_mask & col_mask
    
    # 5. APLICAR FILTRO FINAL (solo eliminar filas, mantener todas las columnas)
    rows_before = len(df_clean)
    df_final = df_clean[valid_mask].copy()
    rows_after = len(df_final)
    
    print(f"\n4. RESUMEN FILTRADO:")
    print(f"   Filas antes del filtrado: {rows_before}")
    print(f"   Filas después del filtrado: {rows_after}")
    print(f"   Filas eliminadas: {rows_before - rows_after}")
    print(f"   Tasa de retención: {(rows_after/rows_before)*100:.1f}%")
    
    return df_final

def analyze_and_select_columns(df):
    """
    Analiza qué columnas mantener basándose en datos faltantes y relevancia
    """
    print("\n" + "=" * 70)
    print("ETAPA 3: ANÁLISIS Y SELECCIÓN DE COLUMNAS")
    print("=" * 70)
    
    # COLUMNAS QUE DEFINITIVAMENTE VAMOS A MANTENER (las 14 esenciales)
    essential_columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    
    # COLUMNAS QUE DEFINITIVAMENTE VAMOS A ELIMINAR (según documentación)
    columns_to_drop_definite = [
        # Marcadas explícitamente como "not used" o "irrelevant"
        'thalsev', 'thalpul', 'earlobe',  # 52-54: not used
        'restckm', 'exerckm',             # 45-46: irrelevant
        'lvx1', 'lvx2', 'lvx3', 'lvx4',   # 69-72: not used
        'lvf', 'cathef', 'junk',          # 73-75: not used
        
        # Identificadores y datos sensibles
        'id', 'ccf', 'name',
        
        # Columnas de fecha específicas que no aportan al modelo
        'ekgmo', 'ekgday', 'ekgyr',       # 20-22: fecha ECG
        'cmo', 'cday', 'cyr',             # 55-57: fecha cateterismo cardíaco
        
        # Dummy variables
        'dummy'                           # 36: dummy
    ]
    
    # COLUMNAS PARA EVALUAR 
    columns_to_evaluate = [
    
        # Factores de riesgo adicionales
        'htn', 'years', 'dm', 'famhist',
        

        # Medidas cardíacas adicionales
        'thalrest', 
  
    ]
    
    print("1. COLUMNAS ESENCIALES (14 variables principales):")
    for col in essential_columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            total = len(df)
            pct_missing = (missing / total) * 100
            print(f"   ✓ {col}: {missing}/{total} faltantes ({pct_missing:.1f}%)")
    
    print(f"\n2. COLUMNAS A ELIMINAR DEFINITIVAMENTE ({len(columns_to_drop_definite)} columnas):")
    for col in columns_to_drop_definite:
        if col in df.columns:
            print(f"   ✗ {col}")
    
    print(f"\n3. EVALUANDO {len(columns_to_evaluate)} COLUMNAS POTENCIALMENTE ÚTILES:")
    
    columns_to_keep_from_evaluation = []
    
    for col in columns_to_evaluate:
        if col in df.columns:
            missing = df[col].isna().sum()
            total = len(df)
            pct_missing = (missing / total) * 100
            unique_values = df[col].nunique()
            
            print(f"   {col}:")
            print(f"      - Faltantes: {missing}/{total} ({pct_missing:.1f}%)")
            print(f"      - Valores únicos: {unique_values}")
            
            # Decisión basada en porcentaje de faltantes
            if pct_missing <= 10:  # Mantener si tiene menos del 10% de faltantes
                columns_to_keep_from_evaluation.append(col)
                print(f"      - DECISIÓN: ✓ MANTENER (solo {pct_missing:.1f}% faltantes)")
            else:
                print(f"      - DECISIÓN: ✗ ELIMINAR (demasiados faltantes: {pct_missing:.1f}%)")
    
    # 4. CREAR LISTA FINAL DE COLUMNAS A MANTENER
    final_columns_to_keep = essential_columns + columns_to_keep_from_evaluation + ['dataset']
    
    # Eliminar duplicados y asegurar que existen en el DataFrame
    final_columns_to_keep = list(set(final_columns_to_keep) & set(df.columns))
    
    print(f"\n4. RESUMEN FINAL DE SELECCIÓN:")
    print(f"   Columnas totales originales: {len(df.columns)}")
    print(f"   Columnas a mantener: {len(final_columns_to_keep)}")
    print(f"   Columnas a eliminar: {len(df.columns) - len(final_columns_to_keep)}")
    
    # Crear DataFrame final
    df_final = df[final_columns_to_keep].copy()
    
    print(f"\n5. COLUMNAS MANTENIDAS ({len(df_final.columns)}):")
    for i, col in enumerate(sorted(df_final.columns), 1):
        if col != 'dataset':
            missing = df_final[col].isna().sum()
            total = len(df_final)
            pct_missing = (missing / total) * 100
            print(f"   {i:2d}. {col}: {missing}/{total} faltantes ({pct_missing:.1f}%)")
    
    return df_final

def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("INICIANDO PROCESAMIENTO DE DATOS DE CLEVELAND")
    print("OBJETIVO: Seleccionar columnas basándose en datos faltantes y relevancia")
    
    # 1. Cargar datos originales
    df_raw = load_and_clean_data()
    if df_raw.empty:
        return
    
    # 2. Limpiar datos (manteniendo todas las columnas)
    df_cleaned = clean_to_processed_format(df_raw)
    
    # 3. Analizar y seleccionar columnas
    df_final = analyze_and_select_columns(df_cleaned)
    
    # 4. Guardar resultado
    output_path = Path(__file__).parent / "cleveland_selected_columns.csv"
    df_final.to_csv(output_path, index=False)
    print(f"\n✓ Dataset con columnas seleccionadas guardado en: {output_path}")
    print(f"✓ Columnas finales: {len(df_final.columns)}")
    print(f"✓ Filas finales: {len(df_final)}")
    
    return df_final

if __name__ == "__main__":
    df_final = main()