import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

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

def RedNeuronalClasificacion(df, target_col='num', binary_threshold=0):
    """
    Red Neuronal para clasificación binaria con 2 salidas
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset completo con variables predictoras y variable objetivo
    target_col : str, default='num'
        Nombre de la columna objetivo
    binary_threshold : int, default=0
        Umbral para binarizar la variable objetivo
        
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
    
    # 3. Escalar datos (CRÍTICO para redes neuronales)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Construir red neuronal
    input_dim = X_train_scaled.shape[1]
    
    model = Sequential([
        # Capa de entrada
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capa oculta 1
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        # Capa oculta 2
        Dense(16, activation='relu'),
        Dropout(0.1),
        
        # Capa de salida - 1 neurona con sigmoid para binaria
        Dense(1, activation='sigmoid')  # 2 salidas implícitas: [prob_clase_0, prob_clase_1]
    ])
    
    # 5. Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    
    print(f"\n🧠 ARQUITECTURA DE LA RED NEURONAL:")
    print(f"   • Capa entrada: {input_dim} neuronas")
    print(f"   • Capa oculta 1: 64 neuronas (ReLU)")
    print(f"   • Capa oculta 2: 32 neuronas (ReLU)") 
    print(f"   • Capa oculta 3: 16 neuronas (ReLU)")
    print(f"   • Capa salida: 1 neurona (Sigmoid)")
    print(f"   • Dropout: 30%/20%/10% para regularización")
    
    # 6. Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # 7. Entrenar modelo
    print(f"\n🚀 ENTRENANDO RED NEURONAL...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=150,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 8. Predicciones
    y_pred_proba = model.predict(X_test_scaled).flatten()
    
    # 9. Optimizar umbral
    optimal_threshold, max_f1 = optimizar_umbral_inteligente(y_test, y_pred_proba)
    
    # Aplicar umbral recomendado
    y_pred_final = (y_pred_proba >= optimal_threshold).astype(int)
    
    # 10. Calcular métricas
    accuracy_final = accuracy_score(y_test, y_pred_final)
    precision_final = precision_score(y_test, y_pred_final)
    recall_final = recall_score(y_test, y_pred_final)
    auc_score_val = roc_auc_score(y_test, y_pred_proba)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_final)
    tn, fp, fn, tp = cm.ravel()
    sensibilidad = recall_final
    especificidad = tn / (tn + fp)
    
    print(f"\n🧠 RESULTADOS RED NEURONAL:")
    print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
    print(f"   • Umbral óptimo: {optimal_threshold:.3f}")
    print(f"   • Accuracy:    {accuracy_final:.4f}")
    print(f"   • Precision:   {precision_final:.4f}")
    print(f"   • Recall:      {recall_final:.4f}")
    print(f"   • F1-Score:    {max_f1:.4f}")
    print(f"   • Sensibilidad: {sensibilidad:.4f}")
    print(f"   • Especificidad: {especificidad:.4f}")
    print(f"   • AUC-ROC:     {auc_score_val:.4f}")
    
    # 11. Preparar datos para gráficos
    umbrales_prueba = np.linspace(0.1, 0.9, 50)
    sensibilidad_list = []
    especificidad_list = []
    f1_scores_list = []
    
    for umbral in umbrales_prueba:
        y_pred_temp = (y_pred_proba >= umbral).astype(int)
        cm_temp = confusion_matrix(y_test, y_pred_temp)
        tn_temp, fp_temp, fn_temp, tp_temp = cm_temp.ravel()
        
        sensibilidad_list.append(tp_temp / (tp_temp + fn_temp))
        especificidad_list.append(tn_temp / (tn_temp + fp_temp))
        f1_scores_list.append(f1_score(y_test, y_pred_temp))
    
    # 12. Visualizaciones
    plt.figure(figsize=(15, 10))
    
    # Gráfico 1: Matriz de confusión
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
    plt.title(f'Matriz de Confusión - Red Neuronal\n(Umbral: {optimal_threshold:.3f})')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    # Gráfico 2: Métricas vs umbral
    plt.subplot(2, 2, 2)
    plt.plot(umbrales_prueba, sensibilidad_list, 'o-', label='Sensibilidad', linewidth=2, markersize=3)
    plt.plot(umbrales_prueba, especificidad_list, 'o-', label='Especificidad', linewidth=2, markersize=3)
    plt.plot(umbrales_prueba, f1_scores_list, 'o-', label='F1-Score', linewidth=2, markersize=3)
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Umbral seleccionado\n{optimal_threshold:.3f}')
    plt.xlabel('Umbral')
    plt.ylabel('Valor de Métrica')
    plt.title('Métricas por Umbral - Red Neuronal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 3: Pérdida durante entrenamiento
    plt.subplot(2, 2, 3)
    plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida Validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Evolución de la Pérdida durante Entrenamiento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 4: Precisión durante entrenamiento
    plt.subplot(2, 2, 4)
    plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión Validación')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Evolución de la Precisión durante Entrenamiento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 13. Retornar resultados completos
    return {
        'modelo': model,
        'scaler': scaler,
        'history': history,
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
    from sklearn.model_selection import train_test_split
    
    df_cleveland = main_data()
    
    # Clasificación binaria: sin enfermedad (0) vs con enfermedad (1-4)
    resultados_nn = RedNeuronalClasificacion(df_cleveland, target_col='num', binary_threshold=0)