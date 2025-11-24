import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

def optimizar_umbral_inteligente(y_test, y_pred_proba):
    """
    Optimiza el umbral de clasificación para maximizar el F1-Score
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    
    max_f1 = np.max(f1_scores)
    optimal_indices = np.where(f1_scores == max_f1)[0]
    
    best_idx = optimal_indices[0]
    optimal_threshold = thresholds[best_idx]
    
    return optimal_threshold, max_f1

def encontrar_k_optimo(X, y, k_range=range(1, 31)):
    """
    Encuentra la K óptima usando validación cruzada con F1-Score
    """
    from sklearn.model_selection import cross_val_score
    
    f1_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        # Usar F1-Score 
        cv_scores = cross_val_score(knn, X, y, cv=5, scoring='f1')
        f1_scores.append(cv_scores.mean())
    
    mejor_k = k_range[np.argmax(f1_scores)]
    mejor_f1 = max(f1_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, f1_scores, 'ro-', linewidth=2, markersize=6)
    plt.axvline(mejor_k, color='red', linestyle='--', label=f'K óptimo = {mejor_k}')
    plt.xlabel('Valor de K')
    plt.ylabel('F1-Score Validación Cruzada')
    plt.title('Búsqueda de K Óptimo para KNN (Maximizando F1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"🎯 K óptimo encontrado: {mejor_k} (F1-Score: {mejor_f1:.4f})")
    return mejor_k, f1_scores
def KNNClasificacion(df, target_col='num', binary_threshold=0, encontrar_k_auto=True):
    """
    K-Nearest Neighbors con optimización automática de K
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset completo con variables predictoras y variable objetivo
    target_col : str, default='num'
        Nombre de la columna objetivo
    binary_threshold : int, default=0
        Umbral para binarizar la variable objetivo
    encontrar_k_auto : bool, default=True
        Si True, encuentra K óptimo automáticamente
        
    Returns:
    --------
    dict : Diccionario con todos los resultados del modelo
    """
    
    # 1. Preparar datos para clasificación binaria
    df_clean = df.copy()
    df_clean['target_binary'] = (df_clean[target_col] > binary_threshold).astype(int)
    
    # Excluir columnas no predictoras
    exclude_cols = [target_col, 'dataset', 'target_binary']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean['target_binary']
    
    print(f" DISTRIBUCIÓN DE CLASES:")
    print(f" Sin enfermedad (0): {(y == 0).sum()} casos")
    print(f" Con enfermedad (1): {(y == 1).sum()} casos")
    print(f" Proporción: {(y == 1).mean()*100:.1f}% con enfermedad")
    
    # 2. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Escalar datos 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Encontrar K óptimo
    if encontrar_k_auto:
        print("\n🔍 BUSCANDO K ÓPTIMO...")
        k_optimo, scores = encontrar_k_optimo(X_train_scaled, y_train, range(1, 21))
    else:
        k_optimo = 5
    
    # 5. Entrenar modelo con K óptimo
    print(f"\n🌐 ENTRENANDO KNN CON K={k_optimo}")
    model = KNeighborsClassifier(
        n_neighbors=k_optimo, 
        weights='distance',
        metric='euclidean'
    )
    model.fit(X_train_scaled, y_train)
    
    # 6. Predicciones y probabilidades
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 7. Optimizar umbral
    optimal_threshold, max_f1 = optimizar_umbral_inteligente(y_test, y_pred_proba)
    
    # Aplicar umbral recomendado
    y_pred_final = y_pred_proba >= optimal_threshold
    
    # Calcular métricas completas
    accuracy_final = accuracy_score(y_test, y_pred_final)
    precision_final = precision_score(y_test, y_pred_final)
    recall_final = recall_score(y_test, y_pred_final)
    auc_score_val = roc_auc_score(y_test, y_pred_proba)
    
    # Matriz de confusión para métricas adicionales
    cm = confusion_matrix(y_test, y_pred_final)
    tn, fp, fn, tp = cm.ravel()
    sensibilidad = recall_final
    especificidad = tn / (tn + fp)
    
    print(f"\n RESULTADOS KNN (K={k_optimo}):")
    print(f"   • Umbral óptimo: {optimal_threshold:.3f}")
    print(f"   • Accuracy:    {accuracy_final:.4f}")
    print(f"   • Precision:   {precision_final:.4f}")
    print(f"   • Recall:      {recall_final:.4f}")
    print(f"   • F1-Score:    {max_f1:.4f}")
    print(f"   • Sensibilidad: {sensibilidad:.4f}")
    print(f"   • Especificidad: {especificidad:.4f}")
    print(f"   • AUC-ROC:     {auc_score_val:.4f}")
    
    # 8. Preparar datos para el gráfico de métricas vs umbral
    umbrales_prueba = np.linspace(0.1, 0.9, 50)
    sensibilidad_list = []
    especificidad_list = []
    f1_scores_list = []
    
    for umbral in umbrales_prueba:
        y_pred_temp = y_pred_proba >= umbral
        cm_temp = confusion_matrix(y_test, y_pred_temp)
        tn_temp, fp_temp, fn_temp, tp_temp = cm_temp.ravel()
        
        sensibilidad_list.append(tp_temp / (tp_temp + fn_temp))
        especificidad_list.append(tn_temp / (tn_temp + fp_temp))
        f1_scores_list.append(f1_score(y_test, y_pred_temp))
    
    # 9. Visualizaciones
    plt.figure(figsize=(12, 5))
    
    # Matriz de confusión
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
    plt.title(f'Matriz de Confusión KNN\n(K={k_optimo}, Umbral: {optimal_threshold:.3f})')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    # Gráfico de métricas vs umbral
    plt.subplot(1, 2, 2)
    plt.plot(umbrales_prueba, sensibilidad_list, 'o-', label='Sensibilidad', linewidth=2, markersize=3)
    plt.plot(umbrales_prueba, especificidad_list, 'o-', label='Especificidad', linewidth=2, markersize=3)
    plt.plot(umbrales_prueba, f1_scores_list, 'o-', label='F1-Score', linewidth=2, markersize=3)
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Umbral seleccionado\n{optimal_threshold:.3f}')
    plt.xlabel('Umbral')
    plt.ylabel('Valor de Métrica')
    plt.title('Métricas por Umbral - KNN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 10. Gráfico adicional: Comparación de diferentes valores de K
    if encontrar_k_auto:
        plt.figure(figsize=(10, 6))
        k_range = range(1, 21)
        k_scores = []
        
        for k in k_range:
            knn_temp = KNeighborsClassifier(n_neighbors=k, weights='distance')
            knn_temp.fit(X_train_scaled, y_train)
            y_pred_temp = knn_temp.predict(X_test_scaled)
            k_scores.append(accuracy_score(y_test, y_pred_temp))
        
        plt.plot(k_range, k_scores, 'go-', linewidth=2, markersize=6)
        plt.axvline(k_optimo, color='red', linestyle='--', label=f'K óptimo = {k_optimo}')
        plt.xlabel('Valor de K')
        plt.ylabel('Accuracy en Test')
        plt.title('Rendimiento de KNN con Diferentes Valores de K')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
  
    # 11. Retornar resultados completos
    return {
        'modelo': model,
        'scaler': scaler,
        'k_optimo': k_optimo,
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
        'auc_score': auc_score_val,
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
    resultados_knn = KNNClasificacion(df_cleveland, target_col='num', binary_threshold=0)