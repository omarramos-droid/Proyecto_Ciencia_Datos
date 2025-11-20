import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def knn_multiclase(df, target_col='num'):
    """
    K-Nearest Neighbors para clasificación multiclase (5 clases)
    """
    
    # 1. Preparar datos manteniendo todas las clases
    df_clean = df.copy()
    
    exclude_cols = [target_col, 'dataset']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    print("🎯 K-NEAREST NEIGHBORS - CLASIFICACIÓN MULTICLASE")
    print("="*50)
    
    # Distribución de clases
    distribucion = y.value_counts().sort_index()
    for clase, count in distribucion.items():
        porcentaje = count / len(y) * 100
        print(f"Clase {clase}: {count} casos ({porcentaje:.1f}%)")
    
    # 2. Dividir y escalar datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Encontrar K óptimo usando validación cruzada
    print(f"\n🔍 BUSCANDO K ÓPTIMO PARA MULTICLASE...")
    
    k_range = range(1, 31)
    k_scores_accuracy = []
    k_scores_f1 = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Accuracy en CV
        scores_acc = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
        k_scores_accuracy.append(scores_acc.mean())
        
        # F1 macro en CV
        scores_f1 = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='f1_macro')
        k_scores_f1.append(scores_f1.mean())
    
    # Encontrar K óptimo (máximo F1 macro)
    optimal_k_f1 = k_range[np.argmax(k_scores_f1)]
    optimal_k_acc = k_range[np.argmax(k_scores_accuracy)]
    
    # Preferir F1 macro para multiclase desbalanceada
    optimal_k = optimal_k_f1
    
    print(f"K óptimo (F1 macro): {optimal_k_f1}")
    print(f"K óptimo (Accuracy): {optimal_k_acc}")
    print(f"K seleccionado: {optimal_k}")
    print(f"F1 macro máximo en CV: {np.max(k_scores_f1):.4f}")
    
    # 4. Gráfico de selección de K
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(k_range, k_scores_accuracy, 'bo-', linewidth=2, markersize=6, label='Accuracy')
    plt.plot(k_range, k_scores_f1, 'ro-', linewidth=2, markersize=6, label='F1 Macro')
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'K óptimo = {optimal_k}')
    plt.xlabel('Número de Vecinos (K)')
    plt.ylabel('Score en Validación Cruzada')
    plt.title('Selección de K Óptimo - Multiclase')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Entrenar modelo con K óptimo
    knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_optimal.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred = knn_optimal.predict(X_test_scaled)
    y_pred_proba = knn_optimal.predict_proba(X_test_scaled)
    
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_macro_test = f1_score(y_test, y_pred, average='macro')
    f1_weighted_test = f1_score(y_test, y_pred, average='weighted')
    
    # 6. Métricas detalladas
    print(f"\n✅ RESULTADOS CON K = {optimal_k}")
    print("="*50)
    print(f"Accuracy en test: {accuracy_test:.4f}")
    print(f"F1 Macro en test: {f1_macro_test:.4f}")
    print(f"F1 Weighted en test: {f1_weighted_test:.4f}")
    
    # 7. Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    nombres_clases = {
        0: 'Sin Enfermedad',
        1: 'Enfermedad Leve', 
        2: 'Enfermedad Moderada',
        3: 'Enfermedad Severa',
        4: 'Enfermedad Muy Severa'
    }
    
    plt.subplot(1, 3, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[nombres_clases[i] for i in range(5)],
                yticklabels=[nombres_clases[i] for i in range(5)])
    plt.title(f'Matriz de Confusión - KNN (K={optimal_k})')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 8. Reporte de clasificación
    print(f"\n📋 REPORTE DE CLASIFICACIÓN - KNN MULTICLASE")
    print("="*50)
    print(classification_report(y_test, y_pred, 
                              target_names=[nombres_clases[i] for i in range(5)]))
    
    # 9. Análisis de F1 por clase
    f1_por_clase = f1_score(y_test, y_pred, average=None)
    
    plt.subplot(1, 3, 3)
    plt.bar(range(5), f1_por_clase, color='lightcoral', alpha=0.7)
    plt.xlabel('Clase')
    plt.ylabel('F1-Score')
    plt.title('F1-Score por Clase')
    plt.xticks(range(5), [nombres_clases[i] for i in range(5)], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # 10. Análisis de sensibilidad a K
    print(f"\n🔎 ANÁLISIS DE SENSIBILIDAD A K")
    print("="*50)
    
    k_df = pd.DataFrame({
        'K': list(k_range), 
        'Accuracy_CV': k_scores_accuracy,
        'F1_Macro_CV': k_scores_f1
    })
    k_df_top = k_df.sort_values('F1_Macro_CV', ascending=False).head(10)
    print("Top 10 K values por F1 Macro:")
    print(k_df_top.to_string(index=False))
    
    # 11. Retornar resultados
    resultados = {
        'modelo': knn_optimal,
        'scaler': scaler,
        'optimal_k': optimal_k,
        'accuracy_test': accuracy_test,
        'f1_macro_test': f1_macro_test,
        'f1_weighted_test': f1_weighted_test,
        'f1_por_clase': f1_por_clase,
        'k_scores_accuracy': k_scores_accuracy,
        'k_scores_f1': k_scores_f1,
        'k_range': list(k_range),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'nombres_clases': nombres_clases
    }
    
    return resultados

def kmeans_multiclase(df, max_k=10):
    """
    K-Means clustering no supervisado con dataset completo (todas las clases)
    """
    
    # 1. Preparar datos (excluir target)
    df_clean = df.copy()
    exclude_cols = ['num', 'dataset']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    
    print("🎯 K-MEANS CLUSTERING - NO SUPERVISADO (DATASET COMPLETO)")
    print("="*50)
    print(f"Variables utilizadas: {len(feature_cols)}")
    print(f"Muestras: {X.shape[0]}")
    
    # 2. Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Método del codo para encontrar K óptimo
    print(f"\n🔍 APLICANDO MÉTODO DEL CODO...")
    
    inertia = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        
        # Calcular silhouette score
        if k > 1:
            silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
    
    # 4. Encontrar K óptimo
    # Método del codo: calcular segunda derivada
    differences = np.diff(inertia)
    second_diff = np.diff(differences)
    optimal_k_elbow = np.argmax(np.abs(second_diff)) + 3
    
    # Silhouette score
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    print(f"K sugerido por método del codo: {optimal_k_elbow}")
    print(f"K sugerido por silhouette score: {optimal_k_silhouette}")
    
    # Para dataset médico, podríamos probar con 5 clusters (una por clase)
    optimal_k = optimal_k_silhouette
    
    # 5. Gráficos de evaluación
    plt.figure(figsize=(15, 5))
    
    # Gráfico del método del codo
    plt.subplot(1, 3, 1)
    plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=6)
    plt.axvline(optimal_k_elbow, color='red', linestyle='--', 
                label=f'Codo sugerido: K={optimal_k_elbow}')
    plt.axvline(optimal_k_silhouette, color='green', linestyle='--',
                label=f'Mejor silhouette: K={optimal_k_silhouette}')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para K-Means')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de silhouette scores
    plt.subplot(1, 3, 2)
    plt.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=6)
    plt.axvline(optimal_k_silhouette, color='red', linestyle='--',
                label=f'Mejor silhouette: K={optimal_k_silhouette}')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score por Número de Clusters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Entrenar modelo final con K óptimo
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    
    # 7. Análisis de los clusters
    df_clustered = df_clean.copy()
    df_clustered['cluster'] = cluster_labels
    
    print(f"\n📊 ANÁLISIS DE CLUSTERS (K={optimal_k})")
    print("="*50)
    
    # Distribución de clusters
    cluster_dist = df_clustered['cluster'].value_counts().sort_index()
    for cluster, count in cluster_dist.items():
        print(f"Cluster {cluster}: {count} casos ({count/len(df_clustered)*100:.1f}%)")
    
    # 8. Relación con la variable objetivo REAL
    print(f"\n🔗 RELACIÓN CLUSTERS vs ENFERMEDAD CARDÍACA REAL")
    print("="*50)
    
    nombres_clases = {
        0: 'Sin Enfermedad',
        1: 'Enfermedad Leve', 
        2: 'Enfermedad Moderada',
        3: 'Enfermedad Severa',
        4: 'Enfermedad Muy Severa'
    }
    
    # Crear tabla de contingencia
    contingency_table = pd.crosstab(df_clustered['cluster'], 
                                  df_clustered['num'],
                                  normalize='index') * 100
    
    plt.subplot(1, 3, 3)
    sns.heatmap(contingency_table, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=[nombres_clases[i] for i in range(5)])
    plt.title(f'Distribución de Enfermedad Real por Cluster (%)\n(K={optimal_k})')
    plt.ylabel('Cluster')
    plt.xlabel('Nivel de Enfermedad Real')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("Distribución porcentual de enfermedad real por cluster:")
    print(contingency_table.round(1))
    
    # 9. Análisis de correspondencia clusters-enfermedad
    print(f"\n🎯 CORRESPONDENCIA CLUSTERS - ENFERMEDAD REAL")
    print("="*50)
    
    # Para cada cluster, encontrar la clase más frecuente
    for cluster in range(optimal_k):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster]
        clase_predominante = cluster_data['num'].mode()[0]
        proporcion = (cluster_data['num'] == clase_predominante).mean() * 100
        
        print(f"Cluster {cluster}: {clase_predominante} ({nombres_clases[clase_predominante]}) - {proporcion:.1f}%")
    
    # 10. Visualización de clusters con PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('Clusters Encontrados por K-Means')
    
    # Añadir centroides
    centroids_pca = pca.transform(kmeans_final.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               marker='X', s=200, c='red', label='Centroides')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Visualización con colores por enfermedad real
    plt.subplot(1, 2, 2)
    scatter_real = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_clean['num'], 
                              cmap='plasma', alpha=0.7, s=50)
    plt.colorbar(scatter_real, label='Enfermedad Real')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('Enfermedad Real de los Pacientes')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 11. Retornar resultados
    resultados = {
        'modelo': kmeans_final,
        'scaler': scaler,
        'optimal_k': optimal_k,
        'cluster_labels': cluster_labels,
        'inertia': inertia,
        'silhouette_scores': silhouette_scores,
        'df_clustered': df_clustered,
        'pca': pca,
        'X_pca': X_pca,
        'contingency_table': contingency_table
    }
    
    return resultados

def comparativa_multiclase_completa(df):
    """
    Ejecuta ambos métodos con el dataset completo (5 clases)
    """
    
    print("🎯 COMPARATIVA COMPLETA: KNN vs K-MEANS (5 CLASES)")
    print("="*60)
    
    # 1. KNN Supervisado Multiclase
    print("\n" + "🔍 KNN - CLASIFICACIÓN SUPERVISADA MULTICLASE")
    print("="*50)
    resultados_knn = knn_multiclase(df)
    
    # 2. K-Means No Supervisado
    print("\n" + "🎯 K-MEANS - CLUSTERING NO SUPERVISADO")
    print("="*50)
    resultados_kmeans = kmeans_multiclase(df)
    
    # 3. Comparativa final
    print("\n" + "📊 COMPARATIVA FINAL - 5 CLASES")
    print("="*50)
    print(f"KNN (Supervisado - Multiclase):")
    print(f"  • K óptimo: {resultados_knn['optimal_k']}")
    print(f"  • Accuracy: {resultados_knn['accuracy_test']:.4f}")
    print(f"  • F1 Macro: {resultados_knn['f1_macro_test']:.4f}")
    print(f"  • F1 Weighted: {resultados_knn['f1_weighted_test']:.4f}")
    
    print(f"\nK-Means (No supervisado):")
    print(f"  • K óptimo: {resultados_kmeans['optimal_k']}")
    print(f"  • Silhouette Score: {np.max(resultados_kmeans['silhouette_scores']):.4f}")
    
    # Análisis de correspondencia
    print(f"\n🔍 ANÁLISIS DE CORRESPONDENCIA:")
    contingency = resultados_kmeans['contingency_table']
    for cluster in range(resultados_kmeans['optimal_k']):
        clase_predominante = contingency.iloc[cluster].idxmax()
        proporcion = contingency.iloc[cluster].max()
        print(f"  • Cluster {cluster} → Clase {clase_predominante} ({proporcion:.1f}%)")
    
    return resultados_knn, resultados_kmeans

# Ejecutar todo con dataset completo
if __name__ == "__main__":
    from data_loader import main_data
    df_cleveland = main_data()
    
    # Usar el dataset completo con todas las clases
    resultados_knn_multi, resultados_kmeans_multi = comparativa_multiclase_completa(df_cleveland)