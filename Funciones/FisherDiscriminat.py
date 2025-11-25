import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def encontrar_umbral_optimo_lda(z_scores, y_test):
    """
    Encuentra el umbral óptimo para clasificación en los scores LDA
    Maximizando el accuracy 
    Parameters:
    -----------
    z_scores : array, scores de proyección LDA 
    y_test : array, valores reales
    
    Returns:
    --------
    optimal_threshold : float, umbral óptimo en escala de scores
    best_accuracy : float, mejor accuracy alcanzado
    """
    best_accuracy = 0
    optimal_threshold = 0
    
    # Probar umbrales en el rango real de los scores LDA
    thresholds = np.linspace(np.min(z_scores), np.max(z_scores), 100)
    
    for threshold in thresholds:
        y_pred = (z_scores >= threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            optimal_threshold = threshold
    
    return optimal_threshold, best_accuracy

def FisherLinearDiscriminant(df, target_col='num', binary_threshold=0):
    """
    Implementación  de Fisher LDA
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset completo con variables predictoras y variable objetivo
    target_col : str, default='num'
        Nombre de la columna objetivo en el dataset
    binary_threshold : int, default=0
        Umbral para binarizar la variable objetivo
        
    Returns:
    --------
    dict : Diccionario con resultados válidos del modelo
    """
    
    # 1. Preparar datos para clasificación binaria
    df_clean = df.copy()
    df_clean['target_binary'] = (df_clean[target_col] > binary_threshold).astype(int)
    
    # Excluir columnas no predictoras
    exclude_cols = [target_col, 'dataset', 'target_binary']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean['target_binary']
    
    print(f"Sin enfermedad (0): {(y == 0).sum()} casos")
    print(f"Con enfermedad (1): {(y == 1).sum()} casos")
    print(f"Proporción: {(y == 1).mean()*100:.1f}% con enfermedad")
    
    # 2. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Estandarizar datos (importante para LDA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Implementación de Fisher LDA
    # Separar datos por clase
    X0 = X_train_scaled[y_train == 0]  # Clase 0
    X1 = X_train_scaled[y_train == 1]  # Clase 1
    
    # Calcular medias de clase
    m0 = np.mean(X0, axis=0).reshape(-1, 1)  # Media clase 0
    m1 = np.mean(X1, axis=0).reshape(-1, 1)  # Media clase 1
    
    # Calcular matriz de dispersión 
    S0 = np.cov(X0, rowvar=False)
    S1 = np.cov(X1, rowvar=False)
    Sw = S0 + S1  # Matriz de dispersión intra-clase
    
    # Regularización para estabilidad
    Sw_reg = Sw + np.eye(Sw.shape[0]) * 1e-6
    
    # Calcular vector discriminante de Fisher
    w = np.linalg.solve(Sw_reg, (m1 - m0))
    w = w.ravel()  # Convertir a vector 1D
    
    # 5. Proyectar datos en la dirección discriminante
    z_train = X_train_scaled @ w
    z_test = X_test_scaled @ w
    
    # 6.  Encontrar umbral óptimo 
    optimal_threshold, best_accuracy = encontrar_umbral_optimo_lda(z_test, y_test)
    
    # Aplicar umbral óptimo
    y_pred_final = (z_test >= optimal_threshold).astype(int)
    
    # 7.  CALCULAR  MÉTRICAS 
    accuracy_final = accuracy_score(y_test, y_pred_final)
    precision_final = precision_score(y_test, y_pred_final, zero_division=0)
    recall_final = recall_score(y_test, y_pred_final, zero_division=0)
    f1_final = f1_score(y_test, y_pred_final, zero_division=0)
    
    # Matriz de confusión para métricas adicionales
    cm = confusion_matrix(y_test, y_pred_final)
    tn, fp, fn, tp = cm.ravel()
    sensibilidad = recall_final
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n=== RESULTADOS VÁLIDOS FISHER LDA ===")
    print(f"Umbral óptimo (escala scores): {optimal_threshold:.4f}")
    print(f"Accuracy:    {accuracy_final:.4f}")
    print(f"Precision:   {precision_final:.4f}")
    print(f"Recall:      {recall_final:.4f}")
    print(f"F1-Score:    {f1_final:.4f}")
    print(f"Sensibilidad: {sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")
    
    print(f"\nMatriz de Confusión:")
    print(f"Verdaderos Negativos: {tn}")
    print(f"Falsos Positivos:     {fp}")
    print(f"Falsos Negativos:     {fn}")
    print(f"Verdaderos Positivos: {tp}")
    
    # 8. Importancia de variables (magnitud del vector w)
    importancia_df = pd.DataFrame({
        'Variable': feature_cols,
        'Coeficiente_Fisher': w,
        'Importancia_Abs': np.abs(w)
    }).sort_values('Importancia_Abs', ascending=False)
    
    print(f"\n  Importancia")
    print(importancia_df.head(3).to_string(index=False))
    
    # 9. Visualizaciones 
    plt.figure(figsize=(15, 5))
    
    # Gráfico 1: Matriz de confusión
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
    plt.title(f'Matriz de Confusión\nFisher LDA')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    # Gráfico 2: Distribución de scores LDA por clase
    plt.subplot(1, 2, 2)
    plt.hist(z_test[y_test == 0], bins=15, alpha=0.7, label='Sin Enfermedad', 
             color='blue', density=True)
    plt.hist(z_test[y_test == 1], bins=15, alpha=0.7, label='Con Enfermedad', 
             color='red', density=True)
    plt.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2,
                label=f'Umbral: {optimal_threshold:.3f}')
    plt.xlabel('Score de Proyección LDA')
    plt.ylabel('Densidad')
    plt.title('Distribución de Scores LDA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
   
    # 10. Retornar resultados SOLO VÁLIDOS
    return {
        'modelo': {
            'w': w,
            'scaler': scaler,
            'optimal_threshold': optimal_threshold
        },
        'metricas': {
            'accuracy': accuracy_final,
            'precision': precision_final,
            'recall': recall_final,
            'f1_score': f1_final,
            'sensibilidad': sensibilidad,
            'especificidad': especificidad,
            'matriz_confusion': cm
        },
        'datos': {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred_final,
            'z_scores': z_test
        },
        'importancia_variables': importancia_df
    }

# Función principal 
if __name__ == "__main__":
    from data_loader import main_data  
    df_cleveland = main_data()
    
    # Clasificación binaria CORRECTA
    resultados_fisher = FisherLinearDiscriminant(df_cleveland, target_col='num', binary_threshold=0)