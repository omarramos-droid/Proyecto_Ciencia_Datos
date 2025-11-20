import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def optimizar_umbral_inteligente(y_test, y_pred_proba):
    """
    Optimiza umbral considerando empates y preferiendo umbrales más bajos
    (más sensibilidad) en caso de igualdad
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    
    # Encontrar TODOS los umbrales con F1 máximo
    max_f1 = np.max(f1_scores)
    optimal_indices = np.where(f1_scores == max_f1)[0]
    
    print(f"🔍 Se encontraron {len(optimal_indices)} umbrales con F1 máximo = {max_f1:.4f}")
    
    # Entre los óptimos, preferir el umbral MÁS BAJO (más sensible)
    best_idx = optimal_indices[0]  # El primer umbral (más bajo)
    optimal_threshold = thresholds[best_idx]
    
    # Mostrar todos los umbrales óptimos
    for idx in optimal_indices:
        umbral = thresholds[idx]
        y_temp = y_pred_proba >= umbral
        cm = confusion_matrix(y_test, y_temp)
        tn, fp, fn, tp = cm.ravel()
        
        sensibilidad = tp / (tp + fn)
        especificidad = tn / (tn + fp)
        
        print(f"   • Umbral {umbral:.3f}: Sensibilidad={sensibilidad:.3f}, Especificidad={especificidad:.3f}")
    
    return optimal_threshold, max_f1

def analizar_umbral_optimo(y_test, y_pred_proba, umbrales_a_comparar=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Analiza diferentes umbrales y sus trade-offs"""
    
    resultados = []
    
    print(f"{'Umbral':>6} | {'Accuracy':>8} | {'Sensibilidad':>12} | {'Especificidad':>13} | {'Precisión':>9} | {'F1-Score':>8}")
    print("-" * 85)
    
    for umbral in umbrales_a_comparar:
        y_pred_temp = y_pred_proba >= umbral
        cm = confusion_matrix(y_test, y_pred_temp)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = accuracy_score(y_test, y_pred_temp)
        sensibilidad = tp / (tp + fn)
        especificidad = tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred_temp)
        
        resultados.append({
            'umbral': umbral,
            'accuracy': accuracy,
            'sensibilidad': sensibilidad,
            'especificidad': especificidad,
            'precision': precision,
            'f1': f1,
            'fn': fn,  # Falsos negativos
            'fp': fp   # Falsos positivos
        })
        
        print(f"{umbral:6.2f} | {accuracy:8.4f} | {sensibilidad:12.4f} | {especificidad:13.4f} | {precision:9.4f} | {f1:8.4f}")
    
    return resultados

def plot_metricas_umbral(y_test, y_pred_proba, optimal_threshold):
    """Grafica métricas vs umbral de decisión"""
    umbrales = np.linspace(0, 1, 200)
    sensibilidad_list = []
    especificidad_list = []
    f1_list = []
    precision_list = []

    for t in umbrales:
        y_temp = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_temp).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        esp = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_val = f1_score(y_test, y_temp)

        sensibilidad_list.append(sens)
        especificidad_list.append(esp)
        f1_list.append(f1_val)
        precision_list.append(prec)

    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(umbrales, sensibilidad_list, label='Sensibilidad', linewidth=2)
    plt.plot(umbrales, especificidad_list, label='Especificidad', linewidth=2)
    plt.plot(umbrales, f1_list, label='F1-Score', linewidth=2)
    plt.plot(umbrales, precision_list, label='Precisión', linewidth=2)
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Umbral óptimo F1 = {optimal_threshold:.3f}')
    plt.title("Métricas según el Umbral de Clasificación - Random Forest")
    plt.xlabel("Umbral")
    plt.ylabel("Valor de la Métrica")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    # Enfoque en región de interés
    plt.plot(umbrales, sensibilidad_list, label='Sensibilidad', linewidth=2)
    plt.plot(umbrales, especificidad_list, label='Especificidad', linewidth=2)
    plt.plot(umbrales, f1_list, label='F1-Score', linewidth=2)
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Umbral óptimo F1 = {optimal_threshold:.3f}')
    plt.xlim(0.2, 0.8)  # Zoom en región relevante
    plt.title("Zoom - Región de Umbrales Relevantes")
    plt.xlabel("Umbral")
    plt.ylabel("Valor de la Métrica")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_importancia_variables_rf(model, feature_names, top_n=15):
    """Grafica la importancia de variables para Random Forest"""
    importancia = model.feature_importances_
    indices = np.argsort(importancia)[::-1]
    
    plt.figure(figsize=(12, 8))
    
    # Gráfico de barras horizontal
    plt.subplot(1, 2, 1)
    features_sorted = [feature_names[i] for i in indices[:top_n]]
    importancia_sorted = importancia[indices[:top_n]]
    
    plt.barh(range(len(features_sorted)), importancia_sorted, align='center', color='steelblue')
    plt.yticks(range(len(features_sorted)), features_sorted)
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Variables Más Importantes\nRandom Forest')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    # Gráfico de importancia acumulada
    plt.subplot(1, 2, 2)
    importancia_acumulada = np.cumsum(importancia_sorted)
    plt.plot(range(1, len(importancia_acumulada) + 1), importancia_acumulada, 'o-', linewidth=2)
    plt.xlabel('Número de Variables')
    plt.ylabel('Importancia Acumulada')
    plt.title('Importancia Acumulada de Variables')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return importancia

def interpretar_importancia_rf(importancia_df, top_n=10):
    """Interpreta la importancia de variables para Random Forest"""
    print(f"\n{'='*60}")
    print("INTERPRETACIÓN DE VARIABLES IMPORTANTES - RANDOM FOREST")
    print(f"{'='*60}")
    
    top_vars = importancia_df.head(top_n)
    
    for i, (_, row) in enumerate(top_vars.iterrows()):
        var = row['Variable']
        importancia = row['Importancia']
        
        print(f"\n#{i+1:2d} 🔍 {var}:")
        print(f"   • Importancia: {importancia:.4f} ({importancia*100:.2f}%)")
        
        # Interpretaciones específicas basadas en conocimiento médico
        interpretaciones = {
            'ca': 'Número de vasos coronarios principales coloreados por fluoroscopia',
            'thal': 'Resultado de la prueba de thalassemia',
            'cp': 'Tipo de dolor pectoral',
            'oldpeak': 'Depresión del ST inducida por ejercicio',
            'exang': 'Angina inducida por ejercicio',
            'thalach': 'Frecuencia cardíaca máxima alcanzada',
            'age': 'Edad del paciente',
            'trestbps': 'Presión arterial en reposo',
            'chol': 'Colesterol sérico',
            'sex': 'Género del paciente',
            'slope': 'Pendiente del segmento ST de ejercicio máximo',
        }
        
        if var in interpretaciones:
            print(f"   • Significado médico: {interpretaciones[var]}")
    
    # Análisis de importancia acumulada
    importancia_acumulada = top_vars['Importancia'].cumsum()
    print(f"\n📊 ANÁLISIS DE IMPORTANCIA ACUMULADA:")
    for i in range(min(5, len(importancia_acumulada))):
        print(f"   • Top {i+1} variables: {importancia_acumulada.iloc[i]*100:.1f}% de importancia total")
    
    return top_vars

def RandomForestClasificacion(df, target_col='num', binary_threshold=0, n_estimators=100, random_state=42):
    """
    Función para ejecutar Random Forest en clasificación binaria
    
    Args:
        df (pd.DataFrame): Dataset 
        target_col (str): Columna objetivo
        binary_threshold (int): Umbral para binarizar
        n_estimators (int): Número de árboles en el forest
        random_state (int): Semilla para reproducibilidad
    
    Returns:
        dict: Todos los resultados del modelo
    """
   
    # 1. Preparar datos para clasificación binaria
    df_clean = df.copy()
    df_clean['target_binary'] = df_clean[target_col].apply(
        lambda x: 0 if x <= binary_threshold else 1
    )
    
    # Excluir columnas no predictoras
    exclude_cols = [target_col, 'dataset', 'target_binary']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean['target_binary']
    
    print(f"🎯 DISTRIBUCIÓN DE CLASES:")
    print(f"   • Sin enfermedad (0): {y.value_counts()[0]} casos")
    print(f"   • Con enfermedad (1): {y.value_counts()[1]} casos")
    print(f"   • Proporción: {y.value_counts()[1]/len(y)*100:.1f}% con enfermedad")
    
    # 2. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # 3. Entrenar modelo Random Forest
    print(f"\n🌲 ENTRENANDO RANDOM FOREST:")
    print(f"   • Número de árboles: {n_estimators}")
    print(f"   • Semilla: {random_state}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight='balanced',
        max_depth=10,  # Evitar sobreajuste
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # 4. Predicciones y métricas base
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = y_pred_proba >= 0.5
    
    accuracy_default = accuracy_score(y_test, y_pred_default)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # 5. OPTIMIZAR UMBRAL
    print(f"\n🔍 OPTIMIZACIÓN INTELIGENTE DE UMBRAL:")
    optimal_threshold, max_f1 = optimizar_umbral_inteligente(y_test, y_pred_proba)
    
    # 6. Análisis comparativo de umbrales
    print(f"\n📊 COMPARACIÓN DETALLADA DE UMBRALES:")
    
    umbrales_especiales = [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    # Asegurarnos de incluir el umbral óptimo si no está en la lista
    if optimal_threshold not in umbrales_especiales:
        umbrales_especiales.append(optimal_threshold)
        umbrales_especiales.sort()
    
    resultados_umbrales = analizar_umbral_optimo(y_test, y_pred_proba, umbrales_especiales)
    
    # 7. RECOMENDACIÓN INTELIGENTE
    print(f"\n💡 RECOMENDACIÓN:")
    
    # Buscar el umbral más bajo que maximice F1
    umbrales_optimos = [r for r in resultados_umbrales if abs(r['f1'] - max_f1) < 0.001]
    if umbrales_optimos:
        mejor_umbral = min(umbrales_optimos, key=lambda x: x['umbral'])
        umbral_final = mejor_umbral['umbral']
        print(f"   • F1 máximo encontrado: {max_f1:.4f}")
        print(f"   • Umbral recomendado: {umbral_final:.3f} (más bajo con F1 máximo)")
        print(f"   • Razón: Mayor sensibilidad ({mejor_umbral['sensibilidad']:.3f}) sin perder F1-Score")
    else:
        umbral_final = optimal_threshold
        print(f"   • Usando umbral óptimo calculado: {umbral_final:.3f}")
    
    # Aplicar umbral recomendado
    y_pred_final = y_pred_proba >= umbral_final
    accuracy_final = accuracy_score(y_test, y_pred_final)
    
    print(f"\n✅ RESULTADOS FINALES - RANDOM FOREST:")
    print(f"   • Umbral final: {umbral_final:.3f}")
    print(f"   • Exactitud (Accuracy): {accuracy_final:.4f}")
    print(f"   • AUC-ROC: {auc_score:.4f}")
    
    # 8. Validación cruzada
    print(f"\n📈 VALIDACIÓN CRUZADA:")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"   • Exactitud CV (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 9. Reporte detallado
    print(f"\n📋 REPORTE DE CLASIFICACIÓN DETALLADO:")
    print(classification_report(y_test, y_pred_final, 
                              target_names=['Sin Enfermedad', 'Con Enfermedad']))
    
    # 10. Importancia de variables
    importancia = model.feature_importances_
    importancia_df = pd.DataFrame({
        'Variable': feature_cols,
        'Importancia': importancia
    }).sort_values('Importancia', ascending=False)
    
    print(f"\n🏆 TOP 15 VARIABLES MÁS IMPORTANTES:")
    print(importancia_df.head(15).to_string(index=False))
    
    # 11. Interpretación de importancia
    top_variables = interpretar_importancia_rf(importancia_df, top_n=10)
    
    # 12. Visualizaciones
    plt.figure(figsize=(15, 10))
    
    # Matriz de confusión
    plt.subplot(2, 2, 1)
    cm_final = confusion_matrix(y_test, y_pred_final)
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
    plt.title(f'Matriz de Confusión\n(Umbral: {umbral_final:.3f})')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    # Gráfico de métricas vs umbral
    plt.subplot(2, 2, 2)
    umbrales = [r['umbral'] for r in resultados_umbrales]
    sensibilidad = [r['sensibilidad'] for r in resultados_umbrales]
    especificidad = [r['especificidad'] for r in resultados_umbrales]
    f1_scores_plot = [r['f1'] for r in resultados_umbrales]
    
    plt.plot(umbrales, sensibilidad, 'o-', label='Sensibilidad', linewidth=2)
    plt.plot(umbrales, especificidad, 'o-', label='Especificidad', linewidth=2)
    plt.plot(umbrales, f1_scores_plot, 'o-', label='F1-Score', linewidth=2)
    plt.axvline(umbral_final, color='red', linestyle='--', 
                label=f'Umbral seleccionado\n{umbral_final:.3f}')
    plt.xlabel('Umbral')
    plt.ylabel('Valor de Métrica')
    plt.title('Métricas por Umbral - Random Forest')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Importancia de variables
    plt.subplot(2, 2, 3)
    top_10_vars = importancia_df.head(10)
    plt.barh(top_10_vars['Variable'], top_10_vars['Importancia'], color='lightcoral')
    plt.xlabel('Importancia')
    plt.title('Top 10 Variables Más Importantes')
    plt.gca().invert_yaxis()
    
    # Curva ROC (opcional)
    plt.subplot(2, 2, 4)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'Random Forest (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Gráfico detallado de métricas vs umbral
    plot_metricas_umbral(y_test, y_pred_proba, umbral_final)
    
    # Gráfico detallado de importancia
    plot_importancia_variables_rf(model, feature_cols, top_n=15)
    
    # 13. Retornar resultados completos
    resultados = {
        'modelo': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_final': y_pred_final,
        'y_pred_proba': y_pred_proba,
        'accuracy_final': accuracy_final,
        'auc_score': auc_score,
        'cv_scores': cv_scores,
        'optimal_threshold': umbral_final,
        'importancia_df': importancia_df,
        'feature_names': feature_cols,
        'resultados_umbrales': resultados_umbrales,
        'top_variables': top_variables
    }
    
    return resultados

# Función principal 
if __name__ == "__main__":
    from data_loader import main_data  
    df_cleveland = main_data()
    
    print("🌲 EJECUTANDO RANDOM FOREST - CLASIFICACIÓN BINARIA")
    print("="*60)
    
    # Clasificación binaria: sin enfermedad (0) vs con enfermedad (1-4)
    resultados_rf = RandomForestClasificacion(
        df_cleveland, 
        target_col='num', 
        binary_threshold=0,
        n_estimators=100,
        random_state=42
    )
    
    print("\n" + "="*60)
    print("🎯 RANDOM FOREST - EJECUCIÓN COMPLETADA")
    print("="*60)