import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def encontrar_k_optimo(X, y, k_range=range(1, 31)):
    """
    Encuentra la K óptima usando validación cruzada con F1-Score
    """
    f1_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        cv_scores = cross_val_score(knn, X, y, cv=5, scoring='f1')
        f1_scores.append(cv_scores.mean())
    
    mejor_k = k_range[np.argmax(f1_scores)]
    mejor_f1 = max(f1_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, f1_scores, 'ro-', linewidth=2, markersize=6)
    plt.axvline(mejor_k, color='red', linestyle='--', label=f'K óptimo = {mejor_k}')
    plt.xlabel('Valor de K')
    plt.ylabel('F1-Score Validación Cruzada')
    plt.title('Búsqueda de K Óptimo para KNN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f" K óptimo encontrado: {mejor_k} (F1-Score: {mejor_f1:.4f})")
    return mejor_k, f1_scores

def KNNClasificacion(df, target_col='num', binary_threshold=0, encontrar_k_auto=True):
    """
    K-Nearest Neighbors - Solo métricas válidas del modelo
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
    
    # 3. Escalar datos 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Encontrar K óptimo
    if encontrar_k_auto:
        k_optimo, scores = encontrar_k_optimo(X_train_scaled, y_train, range(1, 21))
    else:
        k_optimo = 10
    
    # 5. Entrenar modelo con K óptimo
    model = KNeighborsClassifier(
        n_neighbors=k_optimo, 
        weights='distance',
        metric='euclidean'
    )
    model.fit(X_train_scaled, y_train)
    
    # 6. Predicción directa 
    y_pred = model.predict(X_test_scaled)
    
    # 7. Calcular  métricas 
    accuracy_final = accuracy_score(y_test, y_pred)
    precision_final = precision_score(y_test, y_pred)
    recall_final = recall_score(y_test, y_pred)
    f1_final = f1_score(y_test, y_pred)
    
    # Matriz de confusión para métricas 
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensibilidad = recall_final
    especificidad = tn / (tn + fp)
    
    print(f"\n MÉTRICAS KNN (K={k_optimo}):")
    print(f"   • Accuracy:     {accuracy_final:.4f}")
    print(f"   • Precision:    {precision_final:.4f}")
    print(f"   • Recall:       {recall_final:.4f}")
    print(f"   • F1-Score:     {f1_final:.4f}")
    print(f"   • Sensibilidad: {sensibilidad:.4f}")
    print(f"   • Especificidad: {especificidad:.4f}")
    
   
    
    # 8. Visualización
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
    plt.title(f'KNN - Matriz de Confusión (K={k_optimo})')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.show()
    
    # 9. Retornar resultados con solo métricas válidas
    return {
        'modelo': model,
        'scaler': scaler,
        'k_optimo': k_optimo,
        'metricas': {
            'accuracy': accuracy_final,
            'precision': precision_final,
            'recall': recall_final,
            'f1_score': f1_final,
            'sensibilidad': sensibilidad,
            'especificidad': especificidad
        },
        'matriz_confusion': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_names': feature_cols
    }

# Función principal 
if __name__ == "__main__":
    from data_loader import main_data  
    df_cleveland = main_data()
    
    # Clasificación binaria
    resultados_knn = KNNClasificacion(df_cleveland, target_col='num', binary_threshold=0)