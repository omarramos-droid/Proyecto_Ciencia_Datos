import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def optimizar_umbral(y_test, y_pred_proba):
    """
    Optimiza el umbral de clasificación para maximizar el F1-Score
    """
    #umbrales para cálculo y visualización
    umbrales_prueba = np.linspace(0.01, 0.99, 100)
    f1_scores = []
    
    for umbral in umbrales_prueba:
        y_pred_temp = y_pred_proba >= umbral
        f1_scores.append(f1_score(y_test, y_pred_temp))
    
    # Encontrar el umbral que maximiza F1
    max_f1 = np.max(f1_scores)
    optimal_indices = np.where(f1_scores == max_f1)[0]
    best_idx = optimal_indices[0]
    optimal_threshold = umbrales_prueba[best_idx]
    
    return optimal_threshold, max_f1, umbrales_prueba, f1_scores

def RegresionLogistica(df, target_col='num', binary_threshold=0):
    """
    Regresión logística binaria con optimización de umbral 
    """
    # 1. Preparar datos para clasificación binaria
    df_clean = df.copy()
    df_clean['target_binary'] = (df_clean[target_col] > binary_threshold).astype(int)
    
    # Excluir columnas no predictoras
    exclude_cols = [target_col, 'dataset', 'target_binary']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean['target_binary']
    
    print(f" Sin enfermedad (0): {(y == 0).sum()} casos")
    print(f" Con enfermedad (1): {(y == 1).sum()} casos")
    print(f" Proporción: {(y == 1).mean()*100:.1f}% con enfermedad")
    
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
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # 5. Optimizar umbral
    optimal_threshold, max_f1, umbrales_prueba, f1_scores_list = optimizar_umbral(y_test, y_pred_proba)
    
    # Aplicar umbral recomendado
    y_pred_final = y_pred_proba >= optimal_threshold
    
    # Calcular métricas completas
    accuracy_final = accuracy_score(y_test, y_pred_final)
    precision_final = precision_score(y_test, y_pred_final)
    recall_final = recall_score(y_test, y_pred_final)
    
    # Matriz de confusión para métricas adicionales
    cm = confusion_matrix(y_test, y_pred_final)
    tn, fp, fn, tp = cm.ravel()
    sensibilidad = recall_final
    especificidad = tn / (tn + fp)
    
    print(f"\n RESULTADOS CON UMBRAL ÓPTIMO ({optimal_threshold:.4f}):")
    print(f"   • Accuracy:    {accuracy_final:.4f}")
    print(f"   • Precision:   {precision_final:.4f}")
    print(f"   • Recall:      {recall_final:.4f}")
    print(f"   • F1-Score:    {max_f1:.4f}")
    print(f"   • Sensibilidad: {sensibilidad:.4f}")
    print(f"   • Especificidad: {especificidad:.4f}")
    print(f"   • AUC-ROC:     {auc_score:.4f}")
    
    sensibilidad_list = []
    especificidad_list = []
    
    for umbral in umbrales_prueba:
        y_pred_temp = y_pred_proba >= umbral
        cm_temp = confusion_matrix(y_test, y_pred_temp)
        tn_temp, fp_temp, fn_temp, tp_temp = cm_temp.ravel()
        
        sensibilidad_list.append(tp_temp / (tp_temp + fn_temp))
        especificidad_list.append(tn_temp / (tn_temp + fp_temp))
    
    # 7.  Matriz de confusión + Métricas vs Umbral
    plt.figure(figsize=(12, 5))
    
    # Matriz de confusión (izquierda)
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
    plt.title(f'Matriz de Confusión\n(Umbral: {optimal_threshold:.4f})')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    # Gráfico de métricas vs umbr
    plt.subplot(1, 2, 2)
    
    # Graficar las curvas
    plt.plot(umbrales_prueba, sensibilidad_list, 'o-', label='Sensibilidad', linewidth=2, markersize=2, color='blue', alpha=0.7)
    plt.plot(umbrales_prueba, especificidad_list, 'o-', label='Especificidad', linewidth=2, markersize=2, color='green', alpha=0.7)
    plt.plot(umbrales_prueba, f1_scores_list, 'o-', label='F1-Score', linewidth=3, markersize=3, color='red')
    
    # Encontrar y marcar el punto de F1 máximo
    f1_max_idx = np.argmax(f1_scores_list)
    umbral_f1_max = umbrales_prueba[f1_max_idx]
    f1_max_valor = f1_scores_list[f1_max_idx]
    
   
    
    # Línea vertical en el umbral óptimo
    plt.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.8, linewidth=2,
                label=f'Umbral seleccionado = {optimal_threshold:.4f}')
    
    plt.xlabel('Umbral de Clasificación')
    plt.ylabel('Valor de Métrica')
    plt.title('Optimización de Umbral - Punto Óptimo F1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
 
    
    # 9. Retornar resultados completos
    return {
        'modelo': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_final': y_pred_final,
        'y_pred_proba': y_pred_proba,
        'accuracy_final': accuracy_final,
        'precision_final': precision_final,
        'recall_final': recall_final,
        'f1_score': max_f1,
        'auc_score': auc_score,
        'optimal_threshold': optimal_threshold,
        'sensibilidad': sensibilidad,
        'especificidad': especificidad,
        'feature_names': feature_cols
    }

# Función principal 
if __name__ == "__main__":
    from data_loader import main_data  
    df_cleveland = main_data()
    
    # Clasificación binaria: sin enfermedad (0) vs con enfermedad (1-4)
    resultados_binarios = RegresionLogistica(df_cleveland, target_col='num', binary_threshold=0)