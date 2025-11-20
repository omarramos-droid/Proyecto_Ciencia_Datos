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
    # (Estas son las variables que se usan en processed.cleveland.data)
    print("\n3. Aplicando criterios de calidad para variables clave:")
    
    quality_criteria = {
        'age': {'min': 15, 'max': 85, 'description': 'edad razonable'},
        'sex': {'values': [0, 1], 'description': '0=female, 1=male'},
        'cp': {'values': [1, 2, 3, 4], 'description': 'tipos de dolor torácico'},
        'trestbps': {'min': 50, 'max': 200, 'description': 'presión arterial razonable'},
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
    # Mantenemos todas las filas que cumplan con los criterios de las variables clave
    valid_mask = pd.Series([True] * len(df_clean))
    
    for col, criteria in quality_criteria.items():
        if col in df_clean.columns:
            col_mask = pd.Series([True] * len(df_clean))
            
            if 'values' in criteria:
                # Para variables categóricas
                col_mask = df_clean[col].isin(criteria['values']) | df_clean[col].isna()
            elif 'min' in criteria and 'max' in criteria:
                # Para variables continuas
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
    
    print(f"\n4. RESUMEN FINAL:")
    print(f"   Filas antes del filtrado: {rows_before}")
    print(f"   Filas después del filtrado: {rows_after}")
    print(f"   Filas eliminadas: {rows_before - rows_after}")
    print(f"   Tasa de retención: {(rows_after/rows_before)*100:.1f}%")
    print(f"   Columnas conservadas: {len(df_final.columns)}")
    
    # 6. VERIFICAR RESULTADO
    print(f"\n5. VERIFICANDO RESULTADO:")
    print(f"   Variables en dataset final: {len(df_final.columns)}")
    print(f"   Registros en dataset final: {len(df_final)}")
    
    # Mostrar estadísticas básicas de las variables clave
    key_columns = list(quality_criteria.keys())
    available_key_columns = [col for col in key_columns if col in df_final.columns]
    
    print(f"\n6. ESTADÍSTICAS DE VARIABLES CLAVE:")
    if available_key_columns:
        print(df_final[available_key_columns].describe())
    
    return df_final


def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("INICIANDO PROCESAMIENTO DE DATOS DE CLEVELAND")
    print("OBJETIVO: Limpiar datos manteniendo TODAS las columnas")
    
    # 1. Cargar datos originales
    df_raw = load_and_clean_data()
    if df_raw.empty:
        return
    
    # 2. Limpiar datos (manteniendo todas las columnas)
    df_cleaned = clean_to_processed_format(df_raw)
    
    # 3. Guardar resultado
    output_path = Path(__file__).parent / "cleveland_complete_cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n✓ Dataset completo limpio guardado en: {output_path}")
    print(f"✓ Columnas totales: {len(df_cleaned.columns)}")
    print(f"✓ Filas totales: {len(df_cleaned)}")
    
    # Mostrar lista de todas las columnas conservadas
    print(f"\nCOLUMNAS CONSERVADAS ({len(df_cleaned.columns)}):")
    for i, col in enumerate(df_cleaned.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return df_cleaned

if __name__ == "__main__":
    df_final = main()

