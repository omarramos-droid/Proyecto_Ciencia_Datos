from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def grafica_metodo_codo(X_scaled, max_k=8):
    """
    Genera la gráfica del método del codo para determinar K óptimo
    """
    inercia = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inercia.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inercia, 'bo-', linewidth=2, markersize=8, markerfacecolor='red')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia (Within-Cluster Sum of Squares)')
    plt.title('Método del Codo para Determinar K Óptimo')
    plt.grid(True, alpha=0.3)
    
    # Destacar todos los K que vamos a evaluar
    for k in [2, 3, 4, 5]:
        plt.axvline(x=k, color=['green', 'orange', 'blue', 'purple'][k-2], 
                   linestyle='--', alpha=0.7, label=f'K={k}')
    plt.legend()
    
    plt.show()
    
    return inercia

def calcular_metricas_clustering(X_scaled, cluster_labels, y_true=None):
    """
    Calcula todas las métricas de evaluación del clustering
    """
    metrics = {
        'silhouette': silhouette_score(X_scaled, cluster_labels)
    }
    
    # Métricas de sensibilidad si tenemos etiquetas reales
    if y_true is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Para clustering vs clasificación, necesitamos mapear clusters a clases
        # Usamos la clase mayoritaria en cada cluster
        df_temp = pd.DataFrame({'cluster': cluster_labels, 'true': y_true})
        cluster_to_class = df_temp.groupby('cluster')['true'].agg(lambda x: x.value_counts().index[0]).to_dict()
        y_pred = [cluster_to_class[c] for c in cluster_labels]
        
        metrics.update({
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        })
    
    return metrics

def evaluar_clusters_varios(df, target_col='num', ks=[2, 3, 4, 5]):
    """
    Evalúa métricas de clustering para varios valores de K
    """
    # Preparar datos
    df_clean = df.copy()
    df_clean['target_binary'] = (df_clean[target_col] > 0).astype(int)
    
    exclude_cols = [target_col, 'dataset', 'target_binary']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean['target_binary']
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Gráfica del método del codo
    inercia = grafica_metodo_codo(X_scaled, max_k=8)
    
    resultados = {}
    todos_labels = {}
    
    # 2. Evaluar cada K
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X_scaled)
        todos_labels[k] = labels
        
        # Calcular métricas
        metrics = calcular_metricas_clustering(X_scaled, labels, y)
        resultados[k] = metrics
        
        # Distribución por enfermedad
        distribucion = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            total = mask.sum()
            enfermos = y[mask].sum()
            tasa_enfermedad = enfermos / total * 100
            distribucion.append({
                'cluster': cluster_id,
                'total': total,
                'enfermos': enfermos,
                'tasa_enfermedad': tasa_enfermedad
            })
        
        resultados[k]['distribucion'] = distribucion
    
    # 3. Mostrar tabla comparativa completa SIN columna "Mejor K"
 
    print(f"{'Métrica':<20} {'K=2':<12} {'K=3':<12} {'K=4':<12} {'K=5':<12}")
   
    
    # Métricas a comparar
    metricas_comparar = {
        'Silhouette Score': 'silhouette',
        'Accuracy': 'accuracy',
        'Precision': 'precision',
        'Recall': 'recall',
        'F1-Score': 'f1_score'
    }
    
    for nombre_metrica, clave_metrica in metricas_comparar.items():
        if clave_metrica in resultados[2]:  # Verificar que la métrica existe
            valores = [resultados[k][clave_metrica] for k in ks]
            
            # Mostrar fila
            if nombre_metrica in ['Silhouette Score', 'Accuracy', 'Precision', 'Recall', 'F1-Score']:
                print(f"{nombre_metrica:<20} {valores[0]:<12.4f} {valores[1]:<12.4f} {valores[2]:<12.4f} {valores[3]:<12.4f}")
            else:
                print(f"{nombre_metrica:<20} {valores[0]:<12.2f} {valores[1]:<12.2f} {valores[2]:<12.2f} {valores[3]:<12.2f}")
    
    print("-" * 70)
    
    # 4. Análisis de distribución por enfermedad
    
    for k in ks:
        print(f"\nK = {k}:")
        for cluster_info in resultados[k]['distribucion']:
            print(f"  Cluster {cluster_info['cluster']}: {cluster_info['total']} casos, "
                  f"{cluster_info['tasa_enfermedad']:.1f}% con enfermedad")
    
    return {
        'resultados': resultados,
        'todos_labels': todos_labels,
        'X_scaled': X_scaled,
        'feature_names': feature_cols
    }

def plotear(X_scaled, labels_2, labels_3, labels_4):
    """
    Plot comparativo de K=2 vs K=3 vs K=4 
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # K=2 - Verde y Negro
    colors_k2 = ['green' if label == 0 else 'black' for label in labels_2]
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors_k2, alpha=0.7, s=50)
    axes[0].set_title('K-Means Clustering (K=2)')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].grid(True, alpha=0.3)
    
    # Leyenda para K=2
    from matplotlib.patches import Patch
    legend_elements_2 = [
        Patch(facecolor='green', label='Cluster 0'),
        Patch(facecolor='black', label='Cluster 1')
    ]
    axes[0].legend(handles=legend_elements_2, loc='upper right')
    
    # K=3 - Verde, Negro y Naranja
    colors_k3 = ['green' if label == 0 else 'black' if label == 1 else 'orange' for label in labels_3]
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors_k3, alpha=0.7, s=50)
    axes[1].set_title('K-Means Clustering (K=3)')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].grid(True, alpha=0.3)
    
    # Leyenda para K=3
    legend_elements_3 = [
        Patch(facecolor='green', label='Cluster 0'),
        Patch(facecolor='black', label='Cluster 1'),
        Patch(facecolor='orange', label='Cluster 2')
    ]
    axes[1].legend(handles=legend_elements_3, loc='upper right')
    
    # K=4 - Verde, Negro, Naranja y Azul
    colors_k4 = ['green' if label == 0 else 'black' if label == 1 else 'orange' if label == 2 else 'blue' for label in labels_4]
    scatter3 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=colors_k4, alpha=0.7, s=50)
    axes[2].set_title('K-Means Clustering (K=4)')
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[2].grid(True, alpha=0.3)
    
    # Leyenda para K=4
    legend_elements_4 = [
        Patch(facecolor='green', label='Cluster 0'),
        Patch(facecolor='black', label='Cluster 1'),
        Patch(facecolor='orange', label='Cluster 2'),
        Patch(facecolor='blue', label='Cluster 3')
    ]
    axes[2].legend(handles=legend_elements_4, loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Cómo usar la función en tu código principal:
if __name__ == "__main__":
    from data_loader import main_data
    
    # Cargar datos
    df_cleveland = main_data()
    
    # Evaluar clusters para K=2,3,4,5
    resultados_evaluacion = evaluar_clusters_varios(df_cleveland)
    
    # Plotear comparación 
    plotear(
       resultados_evaluacion['X_scaled'],
       resultados_evaluacion['todos_labels'][2],
       resultados_evaluacion['todos_labels'][3],
       resultados_evaluacion['todos_labels'][4]
   )