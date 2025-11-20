import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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



def interpretar_coeficientes(importancia_df, top_n=10):
    """Interpreta los coeficientes del modelo de manera más detallada"""
    print(f"\n{'='*60}")
    print("INTERPRETACIÓN DE VARIABLES IMPORTANTES")
    print(f"{'='*60}")
    
    top_vars = importancia_df.head(top_n)
    
    for _, row in top_vars.iterrows():
        coef = row['Coeficiente']
        var = row['Variable']
        impacto = "POSITIVO" if coef > 0 else "NEGATIVO"
        magnitud = abs(coef)
        
        print(f"\n🔍 {var}:")
        print(f"   • Coeficiente: {coef:+.4f} ({impacto})")
        print(f"   • Magnitud: {magnitud:.4f}")
        
        # Interpretación general
        if coef > 0:
            print(f"   • Interpretación: Aumenta la probabilidad de enfermedad cardíaca")
        else:
            print(f"   • Interpretación: Disminuye la probabilidad de enfermedad cardíaca")
    
    return top_vars

def RegresionLogistica(df, target_col='num', binary_threshold=0):
    """
    Función mejorada para ejecutar regresión logística binaria
    
    Args:
        df (pd.DataFrame): Dataset 
        target_col (str): Columna objetivo
        binary_threshold (int): Umbral para binarizar (0: sin enfermedad, >0: con enfermedad)
    
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
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Estandarizar y entrenar modelo
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # 4. Predicciones y métricas base
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_default = y_pred_proba >= 0.55
    
    accuracy_default = accuracy_score(y_test, y_pred_default)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # 5. OPTIMIZAR UMBRAL - VERSIÓN MEJORADA
    print(f"\n🔍 OPTIMIZACIÓN INTELIGENTE DE UMBRAL:")
    optimal_threshold, max_f1 = optimizar_umbral_inteligente(y_test, y_pred_proba)
    
    # 6. Análisis comparativo de umbrales - MÁS DETALLADO
    print(f"\n📊 COMPARACIÓN DETALLADA DE UMBRALES:")
    
    # Incluir umbrales específicos de la zona de interés
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
    
    print(f" Umbral final: {umbral_final:.3f}")
    print(f" Exactitud (Accuracy): {accuracy_final:.4f}")
    print(f"  AUC-ROC: {auc_score:.4f}")
    
    # 8. Reporte detallado
    print(classification_report(y_test, y_pred_final, 
                              target_names=['Sin Enfermedad', 'Con Enfermedad']))
    
    # 9. Importancia de variables
    coeficientes = model.coef_[0]
    importancia_df = pd.DataFrame({
        'Variable': feature_cols,
        'Coeficiente': coeficientes,
        'Importancia_Abs': np.abs(coeficientes)
    }).sort_values('Importancia_Abs', ascending=False)
    
    print(f"\n🏆 TOP 10 VARIABLES MÁS IMPORTANTES:")
    print(importancia_df.head(10).to_string(index=False))
    
    # 10. Interpretación de coeficientes
    top_variables = interpretar_coeficientes(importancia_df, top_n=10)
    
    # 11. Visualizaciones
    # Matriz de confusión
    cm_final = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
    plt.title(f'Matriz de Confusión\n(Umbral: {umbral_final:.3f})')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    # Gráfico de métricas vs umbral
    plt.subplot(1, 2, 2)
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
    plt.title('Métricas por Umbral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
  
    # 12. Retornar resultados completos
    resultados = {
        'modelo': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_final': y_pred_final,
        'y_pred_proba': y_pred_proba,
        'accuracy_final': accuracy_final,
        'auc_score': auc_score,
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
    

    # Clasificación binaria: sin enfermedad (0) vs con enfermedad (1-4)
    resultados_binarios = RegresionLogistica(df_cleveland, target_col='num', binary_threshold=0)
    
