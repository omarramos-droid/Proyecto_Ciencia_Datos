import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluar_umbrales_random_forest(y_test, y_pred_proba):
    """
    Evalúa diferentes umbrales para Random Forest 
    """
    umbrales = np.linspace(0.1, 0.9, 50)
    resultados = []
    
    for umbral in umbrales:
        y_pred = y_pred_proba >= umbral
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        resultados.append({
            'umbral': umbral,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    df_resultados = pd.DataFrame(resultados)
    
    # Encontrar mejor umbral por F1 
    mejor_f1_idx = df_resultados['f1'].idxmax()
    umbral_optimo = df_resultados.loc[mejor_f1_idx, 'umbral']
    f1_optimo = df_resultados.loc[mejor_f1_idx, 'f1']
    
     
    return umbral_optimo, f1_optimo


def RandomForestClasificacion(df, target_col='num', binary_threshold=0, n_estimators=100, random_state=42):
    """
    Ejecuta Random Forest para clasificación binaria
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
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # 3. Entrenar modelo Random Forest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # 4. Predicciones
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)  # Predicción con umbral 0.5
    
    # 4. Importancia de variables (Random Forest)
    importancias = model.feature_importances_

    importancia_df = pd.DataFrame({
    'Variable': feature_cols,
    'Importancia_RF': importancias
    }).sort_values('Importancia_RF', ascending=False)

    print("\n  Importancia de Variables (Random Forest):")
    print(importancia_df.head(5).to_string(index=False))

    
    # 5. Evaluar diferentes umbrales
    optimal_threshold, max_f1 = evaluar_umbrales_random_forest(y_test, y_pred_proba)
    
    # Aplicar umbral seleccionado
    if optimal_threshold == 0.5:
        y_pred_final = y_pred_default
    else:
        y_pred_final = y_pred_proba >= optimal_threshold
    
    # 6. Calcular métricas completas
    accuracy_final = accuracy_score(y_test, y_pred_final)
    precision_final = precision_score(y_test, y_pred_final)
    recall_final = recall_score(y_test, y_pred_final)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_final)
    tn, fp, fn, tp = cm.ravel()
    sensibilidad = recall_final
    especificidad = tn / (tn + fp)
    
    print(f"   • Accuracy:    {accuracy_final:.4f}")
    print(f"   • Precision:   {precision_final:.4f}")
    print(f"   • Recall:      {recall_final:.4f}")
    print(f"   • F1-Score:    {max_f1:.4f}")
    print(f"   • Sensibilidad: {sensibilidad:.4f}")
    print(f"   • Especificidad: {especificidad:.4f}")
    print(f"   • AUC-ROC:     {auc_score:.4f}")
    print(f"   • Umbral usado: {optimal_threshold:.3f}")


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
    plt.figure(figsize=(15, 5))
    
    # Matriz de confusión
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
    plt.title(f'Matriz de Confusión\n(Umbral: {optimal_threshold:.3f})')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    # Métricas vs umbral
    plt.subplot(1, 2, 2)
    plt.plot(umbrales_prueba, sensibilidad_list, 'o-', label='Sensibilidad', linewidth=2, markersize=2)
    plt.plot(umbrales_prueba, especificidad_list, 'o-', label='Especificidad', linewidth=2, markersize=2)
    plt.plot(umbrales_prueba, f1_scores_list, 'o-', label='F1-Score', linewidth=3, markersize=3)
    plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Umbral seleccionado\n{optimal_threshold:.3f}')
    plt.xlabel('Umbral')
    plt.ylabel('Valor de Métrica')
    plt.title('Métricas por Umbral')
    plt.legend()
    plt.grid(True, alpha=0.3)
  
    
    return {
        'modelo': model,
        'metricas': {
            'accuracy': accuracy_final,
            'precision': precision_final,
            'recall': recall_final,
            'f1_score': max_f1,
            'auc_score': auc_score,
            'sensibilidad': sensibilidad,
            'especificidad': especificidad,
            'umbral_used': optimal_threshold
        },
        'y_test': y_test,
        'y_pred': y_pred_final,
        'y_pred_proba': y_pred_proba
    }

# Función principal 
if __name__ == "__main__":
    from data_loader import main_data  
    df_cleveland = main_data()
    
    resultados_rf = RandomForestClasificacion(
        df_cleveland, 
        target_col='num', 
        binary_threshold=0,
        n_estimators=100,
        random_state=42
    )