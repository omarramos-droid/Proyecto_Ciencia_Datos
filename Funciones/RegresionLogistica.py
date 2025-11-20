import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def RegresionLogistica(df):
    """
    Función para ejecutar regresión logística
+    
    Args:
        df (pd.DataFrame): Dataset 
    
    Returns:
        dict: Todos los resultados del modelo
    """
   
    # 1. Preparar datos para clasificacion binaria
    df_clean = df.copy()
    df_clean['target_binary'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)
    
    # Excluir columnas no predictoras
    exclude_cols = ['num', 'dataset', 'target_binary']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean['target_binary']
    
    print(f"Distribución : {y.value_counts()[0]} sin enfermedad vs {y.value_counts()[1]} con enfermedad")
    
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
    y_pred_default = y_pred_proba >= 0.5
    
    accuracy_default = accuracy_score(y_test, y_pred_default)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # 5. OPTIMIZAR UMBRAL

    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Aplicar umbral optimizado
    y_pred_optimized = y_pred_proba >= optimal_threshold
    
    # 6. Comparar métricas
    print(f"COMPARACIÓN DE UMBRALES:")
    print(f"{'Umbral':>8} | {'Accuracy':>10} | {'Sensibilidad':>12} | {'Especificidad':>13} | {'F1-Score':>9}")
    print("-" * 75)
    
    for umbral in [0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9]:
        y_pred_temp = y_pred_proba >= umbral
        cm = confusion_matrix(y_test, y_pred_temp)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy_temp = accuracy_score(y_test, y_pred_temp)
        sensibilidad = tp / (tp + fn)
        especificidad = tn / (tn + fp)
        f1_temp = f1_score(y_test, y_pred_temp)
        
     
    # 7. Usar mejor umbral
    y_pred_final = y_pred_proba >= optimal_threshold
    

    
    accuracy_final = accuracy_score(y_test, y_pred_final)
    
    print(f"Umbral seleccionado: {optimal_threshold:.3f}")
    print(f"Exactitud (Accuracy): {accuracy_final:.4f}")
    print(f"AUC-ROC: {auc_score:.4f}")
    print(f"\nREPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_pred_final, target_names=['Sin Enfermedad', 'Con Enfermedad']))
    
    # 9. Importancia de variables
    coeficientes = model.coef_[0]
    importancia_df = pd.DataFrame({
        'Variable': feature_cols,
        'Coeficiente': coeficientes,
        'Importancia_Abs': np.abs(coeficientes)
    }).sort_values('Importancia_Abs', ascending=False)
    
    print(f"\nTOP 5 VARIABLES MÁS IMPORTANTES:")
    print(importancia_df.head(5).to_string(index=False))
    
    # 10. Matriz de confusión
    cm_final = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
    plt.title(f'Matriz de Confusión - Umbral {optimal_threshold:.3f}')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.show()
    
    umbrales = np.linspace(0, 1, 200)
    sensibilidad_list = []
    especificidad_list = []
    f1_list = []

    for t in umbrales:
        y_temp = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_temp).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        esp = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1   = f1_score(y_test, y_temp)

        sensibilidad_list.append(sens)
        especificidad_list.append(esp)
        f1_list.append(f1)

    plt.figure(figsize=(10,6))
    plt.plot(umbrales, sensibilidad_list, label='Sensibilidad')
    plt.plot(umbrales, especificidad_list, label='Especificidad')
    plt.plot(umbrales, f1_list, label='F1-Score')

    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Umbral óptimo = {optimal_threshold:.2f}')

    plt.title("Métricas según el umbral de clasificación")
    plt.xlabel("Umbral")
    plt.ylabel("Valor Métrica")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

    # 11. Retornar resultados completos
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
        'importancia_df': importancia_df,
        'feature_names': feature_cols
    }
    
    return resultados

# Función principal 
if __name__ == "__main__":
     from data_loader import main_data  
     df_cleveland = main_data()
     resultados = RegresionLogistica(df_cleveland)

        
       