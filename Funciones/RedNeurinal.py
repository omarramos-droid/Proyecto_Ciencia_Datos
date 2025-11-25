import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def optimizar_umbral(y_test, y_pred_proba):
    """
    Optimiza el umbral de clasificación para maximizar el F1-Score
    """
    umbrales = np.linspace(0.1, 0.9, 100)
    mejores_metricas = {
        'umbral': 0.5,
        'f1': 0,
        'precision': 0,
        'recall': 0
    }
    
    for umbral in umbrales:
        y_pred_temp = (y_pred_proba >= umbral).astype(int)
        precision_temp = precision_score(y_test, y_pred_temp, zero_division=0)
        recall_temp = recall_score(y_test, y_pred_temp, zero_division=0)
        f1_temp = f1_score(y_test, y_pred_temp)
        
        if f1_temp > mejores_metricas['f1']:
            mejores_metricas = {
                'umbral': umbral,
                'f1': f1_temp,
                'precision': precision_temp,
                'recall': recall_temp
            }
    
    return mejores_metricas['umbral'], mejores_metricas['f1']


def RedNeuronalClasificacionCompleja(df, target_col='num', binary_threshold=0):
    """
    Red Neuronal para clasificación binaria con optimización automática del umbral 
    """
    
    # 1. Binarizar la variable objetivo
    df_clean = df.copy()
    df_clean['target_binary'] = (df_clean[target_col] > binary_threshold).astype(int)
    
    # Excluir columnas no predictoras
    exclude_cols = [target_col, 'dataset', 'target_binary']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean['target_binary']
    
    print(f"   Clase 0 (sin enfermedad): {(y == 0).sum()} casos - {(y == 0).mean()*100:.1f}%")
    print(f"   Clase 1 (con enfermedad): {(y == 1).sum()} casos - {(y == 1).mean()*100:.1f}%")
    
    # 2. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=13, stratify=y
    )
    
    # 3. Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
   # 4. Construir red neuronal con más capas
    input_dim = X_train_scaled.shape[1]

    model = Sequential([    
    Dense(64, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),  # Capa adicional
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),
    
    Dense(64, activation='relu'),   # Capa adicional
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),   # Capa adicional
    BatchNormalization(),
    Dropout(0.15),
    
    Dense(8, activation='relu'),    # Capa adicional 
    Dropout(0.1),
    
    Dense(1, activation='sigmoid')
])

# 5. Compilar modelo (manteniendo tu configuración)
    model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
    )
    
 
    
    # 6. Entrenar
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=500, 
        batch_size=10,
        callbacks=[early_stop], verbose=1
    )
    
    # 7. Predicciones y optimización de umbral
    y_pred_proba = model.predict(X_test_scaled).flatten()
    
    # Obtener umbral que maximiza F1
    optimal_threshold, max_f1 = optimizar_umbral(y_test, y_pred_proba)
    
  
    
   
    # Aplicar umbral óptimo
    y_pred_final = (y_pred_proba >= optimal_threshold).astype(int)
    
    # 8. Calcular métricas finales
    accuracy_final = accuracy_score(y_test, y_pred_final)
    precision_final = precision_score(y_test, y_pred_final, zero_division=0)
    recall_final = recall_score(y_test, y_pred_final, zero_division=0)
    auc_score_val = roc_auc_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred_final)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"   • Umbral óptimo: {optimal_threshold:.3f}")
    print(f"   • Accuracy:    {accuracy_final:.4f}")
    print(f"   • Precision:   {precision_final:.4f}") 
    print(f"   • Recall:      {recall_final:.4f}")
    print(f"   • F1-Score:    {max_f1:.4f}")
    print(f"   • AUC-ROC:     {auc_score_val:.4f}")
    print(f"   • Sensibilidad: {recall_final:.4f}")
    print(f"   • Especificidad: {tn/(tn+fp):.4f}")
    
    # 9. Preparar datos para gráficos avanzados - 
    umbrales_prueba = np.linspace(0.1, 0.9, 100)  
    sensibilidad_list = []
    especificidad_list = []
    f1_scores_list = []
    precision_list = []
    
    for umbral in umbrales_prueba:
        y_pred_temp = (y_pred_proba >= umbral).astype(int)
        cm_temp = confusion_matrix(y_test, y_pred_temp)
        tn_temp, fp_temp, fn_temp, tp_temp = cm_temp.ravel()
        
        sensibilidad_list.append(tp_temp / (tp_temp + fn_temp) if (tp_temp + fn_temp) > 0 else 0)
        especificidad_list.append(tn_temp / (tn_temp + fp_temp) if (tn_temp + fp_temp) > 0 else 0)
        f1_scores_list.append(f1_score(y_test, y_pred_temp))
        precision_list.append(precision_score(y_test, y_pred_temp, zero_division=0))
    

 # 10. Visualizaciones completas
    plt.figure(figsize=(15, 10))
    
    # Gráfico 1: Matriz de confusión
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
    plt.title(f'Matriz de Confusión\n(Umbral óptimo: {optimal_threshold:.3f})')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    # Gráfico 2: Métricas vs umbral
    plt.subplot(2, 2, 2)
    plt.plot(umbrales_prueba, sensibilidad_list, 'o-', label='Sensibilidad', linewidth=2, markersize=3)
    plt.plot(umbrales_prueba, especificidad_list, 'o-', label='Especificidad', linewidth=2, markersize=3)
    plt.plot(umbrales_prueba, f1_scores_list, 'o-', label='F1-Score', linewidth=2, markersize=3)
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Umbral óptimo\n{optimal_threshold:.3f}')
    plt.xlabel('Umbral')
    plt.ylabel('Valor de Métrica')
    plt.title('Métricas por Umbral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 3: Pérdida durante entrenamiento
    plt.subplot(2, 2, 3)
    plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida Validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Evolución de la Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 4: Precisión durante entrenamiento
    plt.subplot(2, 2, 4)
    plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión Validación')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Evolución de la Precisión')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'modelo': model,
        'scaler': scaler,
        'optimal_threshold': optimal_threshold,
        'metricas': {
            'accuracy': accuracy_final,
            'precision': precision_final,
            'recall': recall_final,
            'f1': max_f1,
            'auc': auc_score_val
        },
        'y_pred_proba': y_pred_proba,
        'y_pred_final': y_pred_final,
        'tipo': 'compleja',
        'umbrales_grafica': umbrales_prueba,
        'f1_scores_grafica': f1_scores_list
    }


# Función principal 
if __name__ == "__main__":
    from data_loader import main_data
    
    df_cleveland = main_data()
    

    
    # Versión Compleja
    resultados_compleja = RedNeuronalClasificacionCompleja(
        df_cleveland, 
        target_col='num', 
        binary_threshold=0
    )
