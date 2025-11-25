import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split




def crear_clases_multiclass(df, target_col='num'):
    """
    Crea 3 clases a partir de la variable objetivo:
    - Clase 0: sin enfermedad (valor original 0)
    - Clase 1: enfermedad leve/moderada (valores originales 1, 2)  
    - Clase 2: enfermedad severa (valores originales 3, 4)
    """
    df_clean = df.copy()
    
    # Mapear a 3 clases
    condiciones = [
        df_clean[target_col] == 0,  # Sin enfermedad
        df_clean[target_col].isin([1]),  # Enfermedad leve/moderada
        df_clean[target_col].isin([2,3, 4])   # Enfermedad severa
    ]
    
    opciones = [0, 1, 2]  # Nuevas etiquetas
    
    df_clean['target_multiclass'] = np.select(condiciones, opciones, default=0)
    
    return df_clean

def RedNeuronalMulticlase(df, target_col='num'):
    """
    Red Neuronal para clasificación MULTICLASE con 3 clases
    """
    
    # 1. Crear las 3 clases
    df_processed = crear_clases_multiclass(df, target_col)
    
    # Excluir columnas no predictoras
    exclude_cols = [target_col, 'dataset', 'target_multiclass']
    feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
    
    X = df_processed[feature_cols]
    y = df_processed['target_multiclass']
    
    # 2. Mostrar distribución de clases
    print(f"📊 DISTRIBUCIÓN DE LAS 3 CLASES:")
    clases_nombres = ['Sin enfermedad (0)', 'Enfermedad leve/moderada (1,2)', 'Enfermedad severa (3,4)']
    for i, nombre in enumerate(clases_nombres):
        count = (y == i).sum()
        porcentaje = (y == i).mean() * 100
        print(f"   • {nombre}: {count} casos - {porcentaje:.1f}%")
    
    # 3. Codificar etiquetas para multiclase
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=3)
    
    # 4. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Convertir a categorical para el entrenamiento
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    # 5. Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Construir red neuronal para MULTICLASE
    input_dim = X_train_scaled.shape[1]
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dropout(0.1),
        
        # CORRECCIÓN: 3 neuronas con softmax para multiclase
        Dense(3, activation='softmax')  # 3 clases
    ])
    
    # 7. Compilar modelo para multiclase
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',  # Cambiado a categorical_crossentropy
        metrics=['accuracy', 'Precision', 'Recall']
    )
    
    print(f"\n🧠 ARQUITECTURA DE LA RED NEURONAL (MULTICLASE):")
    print(f"   • Entrada: {input_dim} características")
    print(f"   • Capas ocultas: 64 → 32 → 16 neuronas (ReLU)")
    print(f"   • Salida: 3 neuronas (Softmax) → Probabilidades para 3 clases")
    print(f"   • Función de pérdida: Categorical Crossentropy")
    
    # 8. Entrenar
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train_cat,
        validation_data=(X_test_scaled, y_test_cat),
        epochs=150, batch_size=16,
        callbacks=[early_stop], verbose=1
    )
    
    # 9. Predicciones
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)  # Clase con mayor probabilidad
    
    # 10. Calcular métricas
    accuracy_final = accuracy_score(y_test, y_pred)
    
    # Métricas por clase
    precision_por_clase = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_por_clase = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_por_clase = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Métricas promediadas
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # 11. Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    nombres_clases = ['Sin enfermedad', 'Leve/Moderada', 'Severa']
    
    print(f"\n🎯 RESULTADOS - CLASIFICACIÓN MULTICLASE:")
    print(f"   • Accuracy total: {accuracy_final:.4f}")
    print(f"   • Precision macro: {precision_macro:.4f}")
    print(f"   • Recall macro: {recall_macro:.4f}")
    print(f"   • F1-Score macro: {f1_macro:.4f}")
    
    print(f"\n📊 MÉTRICAS POR CLASE:")
    for i, nombre in enumerate(nombres_clases):
        print(f"   • {nombre}:")
        print(f"     - Precision: {precision_por_clase[i]:.4f}")
        print(f"     - Recall:    {recall_por_clase[i]:.4f}")
        print(f"     - F1-Score:  {f1_por_clase[i]:.4f}")
    
    # 12. Reporte de clasificación detallado
    print(f"\n📋 REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_pred, target_names=nombres_clases, zero_division=0))
    
    # 13. Visualizaciones
    plt.figure(figsize=(18, 12))
    
    # Gráfico 1: Matriz de confusión
    plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=nombres_clases, 
                yticklabels=nombres_clases)
    plt.title('Matriz de Confusión - 3 Clases')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Gráfico 2: Métricas por clase
    plt.subplot(2, 3, 2)
    x_pos = np.arange(len(nombres_clases))
    width = 0.25
    
    plt.bar(x_pos - width, precision_por_clase, width, label='Precision', alpha=0.8)
    plt.bar(x_pos, recall_por_clase, width, label='Recall', alpha=0.8)
    plt.bar(x_pos + width, f1_por_clase, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Clases')
    plt.ylabel('Valor')
    plt.title('Métricas por Clase')
    plt.xticks(x_pos, nombres_clases, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 3: Pérdida durante entrenamiento
    plt.subplot(2, 3, 3)
    plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida Validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Evolución de la Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 4: Precisión durante entrenamiento
    plt.subplot(2, 3, 4)
    plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión Validación')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Evolución de la Precisión')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 5: Distribución de probabilidades por clase real
    plt.subplot(2, 3, 5)
    colores = ['blue', 'orange', 'red']
    
    for clase in range(3):
        # Filtrar probabilidades para ejemplos de esta clase real
        mascara_clase_real = (y_test == clase)
        probabilidades_clase_predicha = y_pred_proba[mascara_clase_real, clase]
        
        plt.hist(probabilidades_clase_predicha, alpha=0.6, bins=20, 
                color=colores[clase], label=nombres_clases[clase])
    
    plt.xlabel('Probabilidad de Clase Correcta')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Probabilidades\n(por clase real)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 6: Precisión y Recall por clase
    plt.subplot(2, 3, 6)
    x = np.arange(len(nombres_clases))
    
    plt.plot(x, precision_por_clase, 'o-', linewidth=2, markersize=8, label='Precision')
    plt.plot(x, recall_por_clase, 's-', linewidth=2, markersize=8, label='Recall')
    plt.plot(x, f1_por_clase, '^-', linewidth=2, markersize=8, label='F1-Score')
    
    plt.xlabel('Clases')
    plt.ylabel('Valor')
    plt.title('Precision vs Recall por Clase')
    plt.xticks(x, nombres_clases, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 14. Análisis de errores comunes
    print(f"\n🔍 ANÁLISIS DE ERRORES:")
    for i in range(3):
        for j in range(3):
            if i != j and cm[i, j] > 0:
                print(f"   • {nombres_clases[i]} → {nombres_clases[j]}: {cm[i, j]} casos")
    
    return {
        'modelo': model,
        'scaler': scaler,
        'label_encoder': le,
        'metricas': {
            'accuracy': accuracy_final,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_por_clase': precision_por_clase,
            'recall_por_clase': recall_por_clase,
            'f1_por_clase': f1_por_clase
        },
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'nombres_clases': nombres_clases,
        'matriz_confusion': cm,
        'history': history
    }

def predecir_nueva_muestra(modelo, scaler, label_encoder, muestra, nombres_clases):
    """
    Función para predecir una nueva muestra
    """
    # Escalar la muestra
    muestra_scaled = scaler.transform([muestra])
    
    # Predecir probabilidades
    probabilidades = modelo.predict(muestra_scaled)[0]
    
    # Obtener clase predicha
    clase_predicha = np.argmax(probabilidades)
    
    print(f"\n🔮 PREDICCIÓN PARA NUEVA MUESTRA:")
    print(f"   Probabilidades:")
    for i, prob in enumerate(probabilidades):
        print(f"   • {nombres_clases[i]}: {prob:.4f} ({prob*100:.1f}%)")
    
    print(f"   Clase predicha: {nombres_clases[clase_predicha]}")
    
    return clase_predicha, probabilidades


# Función principal 
if __name__ == "__main__":
    from data_loader import main_data
    
    df_cleveland = main_data()
    
    print("🚀 ENTRENANDO RED NEURONAL MULTICLASE (3 CLASES)...")
    print("="*70)
    
    # Entrenar modelo multiclase
    resultados_multiclase = RedNeuronalMulticlase(
        df_cleveland, 
        target_col='num'
    )
    
    print(f"\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"   • Épocas entrenadas: {len(resultados_multiclase['history'].history['loss'])}")
    print(f"   • Accuracy final: {resultados_multiclase['metricas']['accuracy']:.4f}")
    print(f"   • F1-Score macro: {resultados_multiclase['metricas']['f1_macro']:.4f}")

