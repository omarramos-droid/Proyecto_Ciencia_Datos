import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def knn_clasificacion(df, target_col='num', binary_threshold=0):
    """
    K-Nearest Neighbors para clasificación con selección de K óptimo
    """
    
    # 1. Preparar datos
    df_clean = df.copy()
    df_clean['target_binary'] = df_clean[target_col].apply(
        lambda x: 0 if x <= binary_threshold else 1
    )
    
    exclude_cols = [target_col, 'dataset', 'target_binary']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean['target_binary']
    
    print("🎯 K-NEAREST NEIGHBORS - CLASIFICACIÓN SUPERVISADA")
    print("="*50)
    print(f"Clase 0 (Sin enfermedad): {np.sum(y == 0)} casos")
    print(f"Clase 1 (Con enfermedad): {np.sum(y == 1)} casos")
    
    # 2. Dividir y escalar datos (CRUCIAL para KNN)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Encontrar K óptimo usando validación cruzada
    print(f"\n🔍 BUSCANDO K ÓPTIMO...")
    
    k_range = range(1, 31)
    k_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    
    # Encontrar K óptimo (máxima accuracy)
    optimal_k = k_range[np.argmax(k_scores)]
    max_accuracy = np.max(k_scores)
    
    print(f"K óptimo encontrado: {optimal_k}")
    print(f"Accuracy máxima en CV: {max_accuracy:.4f}")
    
    # 4. Gráfico de la "regla del codo" para KNN
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, k_scores, 'bo-', linewidth=2, markersize=6)
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'K óptimo = {optimal_k}')
    plt.xlabel('Número de Vecinos (K)')
    plt.ylabel('Accuracy en Validación Cruzada')
    plt.title('Selección de K Óptimo - Regla del Codo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Entrenar modelo con K óptimo
    knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_optimal.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred = knn_optimal.predict(X_test_scaled)
    y_pred_proba = knn_optimal.predict_proba(X_test_scaled)[:, 1]
    
    accuracy_test = accuracy_score(y_test, y_pred)
    
    # 6. Métricas detalladas
    print(f"\n✅ RESULTADOS CON K = {optimal_k}")
    print("="*50)
    print(f"Accuracy en test: {accuracy_test:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'])
    plt.title(f'Matriz de Confusión - KNN (K={optimal_k})')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    plt.tight_layout()
    plt.show()
    
    # 7. Reporte de clasificación
    print(f"\n📋 REPORTE DE CLASIFICACIÓN - KNN")
    print("="*50)
    print(classification_report(y_test, y_pred, 
                              target_names=['Sin Enfermedad', 'Con Enfermedad']))
    
    # 8. Análisis de sensibilidad a K
    print(f"\n🔎 ANÁLISIS DE SENSIBILIDAD A K")
    print("="*50)
    
    # Mostrar top 5 K values
    k_df = pd.DataFrame({'K': list(k_range), 'Accuracy_CV': k_scores})
    k_df = k_df.sort_values('Accuracy_CV', ascending=False).head(10)
    print(k_df.to_string(index=False))
    
    # 9. Retornar resultados
    resultados = {
        'modelo': knn_optimal,
        'scaler': scaler,
        'optimal_k': optimal_k,
        'accuracy_test': accuracy_test,
        'k_scores': k_scores,
        'k_range': list(k_range),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test
    }
    
    return resultados






from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
def kmeans_clustering(df, max_k=10):
    """
    K-Means clustering no supervisado con método del codo
    """
    
    # 1. Preparar datos (excluir target)
    df_clean = df.copy()
    exclude_cols = ['num', 'dataset']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    
    print("🎯 K-MEANS CLUSTERING - NO SUPERVISADO")
    print("="*50)
    print(f"Variables utilizadas: {len(feature_cols)}")
    print(f"Muestras: {X.shape[0]}")
    
    # 2. Escalar datos (IMPORTANTE para K-Means)
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
        if k > 1:  # silhouette necesita al menos 2 clusters
            silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
    
    # 4. Encontrar K óptimo (método del codo)
    # Calcular la segunda derivada para encontrar el "codo"
    differences = np.diff(inertia)
    second_diff = np.diff(differences)
    optimal_k_elbow = np.argmax(np.abs(second_diff)) + 3  # +3 por los diffs
    
    # También considerar silhouette score
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    print(f"K sugerido por método del codo: {optimal_k_elbow}")
    print(f"K sugerido por silhouette score: {optimal_k_silhouette}")
    
    # Usar el que tenga mejor silhouette score
    optimal_k = optimal_k_silhouette
    
    # 5. Gráficos de evaluación
    plt.figure(figsize=(15, 5))
    
    # Gráfico del método del codo
    plt.subplot(1, 3, 1)
    plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=6)
    plt.axvline(optimal_k_elbow, color='red', linestyle='--', 
                label=f'Codo sugerido: K={optimal_k_elbow}')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia (Within-Cluster Sum of Squares)')
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
    
    # 8. Relación con la variable objetivo (si existe)
    if 'num' in df_clustered.columns:
        print(f"\n🔗 RELACIÓN CLUSTERS vs ENFERMEDAD CARDÍACA")
        print("="*50)
        
        # Crear tabla de contingencia
        contingency_table = pd.crosstab(df_clustered['cluster'], 
                                      df_clustered['num'],
                                      normalize='index') * 100
        
        plt.subplot(1, 3, 3)
        sns.heatmap(contingency_table, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Distribución de Enfermedad por Cluster (%)')
        plt.ylabel('Cluster')
        plt.xlabel('Nivel de Enfermedad')
        
        print("Distribución porcentual de enfermedad por cluster:")
        print(contingency_table.round(1))
    
    plt.tight_layout()
    plt.show()
    
    # 9. Caracterización de clusters
    print(f"\n🎯 CARACTERIZACIÓN DE CLUSTERS")
    print("="*50)
    
    # Calcular promedios por cluster para las variables más importantes
    cluster_means = df_clustered.groupby('cluster')[feature_cols].mean()
    
    # Identificar variables más discriminantes entre clusters
    std_by_cluster = cluster_means.std()
    top_discriminative_vars = std_by_cluster.nlargest(5).index.tolist()
    
    print("Variables más discriminantes entre clusters:")
    for var in top_discriminative_vars:
        print(f"   • {var}: {std_by_cluster[var]:.3f}")
    
    # 10. Visualización de clusters (usando PCA)
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('Visualización de Clusters usando PCA')
    
    # Añadir centroides
    centroids_pca = pca.transform(kmeans_final.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               marker='X', s=200, c='red', label='Centroides')
    plt.legend()
    plt.grid(True, alpha=0.3)
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
        'X_pca': X_pca
    }
    
    return resultados
def comparativa_completa_clustering(df):
    """
    Ejecuta ambos métodos: KNN (supervisado) y K-Means (no supervisado)
    """
    
    print("🎯 COMPARATIVA COMPLETA: KNN vs K-MEANS")
    print("="*60)
    
    # 1. KNN Supervisado
    print("\n" + "🔍 KNN - CLASIFICACIÓN SUPERVISADA")
    print("="*40)
    resultados_knn = knn_clasificacion(df)
    
    # 2. K-Means No Supervisado
    print("\n" + "🎯 K-MEANS - CLUSTERING NO SUPERVISADO")
    print("="*40)
    resultados_kmeans = kmeans_clustering(df)
    
    # 3. Comparativa
    print("\n" + "📊 COMPARATIVA FINAL")
    print("="*40)
    print(f"KNN (Supervisado):")
    print(f"  • K óptimo: {resultados_knn['optimal_k']}")
    print(f"  • Accuracy: {resultados_knn['accuracy_test']:.4f}")
    
    print(f"\nK-Means (No supervisado):")
    print(f"  • K óptimo: {resultados_kmeans['optimal_k']}")
    print(f"  • Silhouette Score: {np.max(resultados_kmeans['silhouette_scores']):.4f}")
    
    return resultados_knn, resultados_kmeans

# Ejecutar todo
if __name__ == "__main__":
    from data_loader import main_data
    df_cleveland = main_data()
    
    resultados_knn, resultados_kmeans = comparativa_completa_clustering(df_cleveland)