from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def grafica_metodo_codo(X_scaled, max_k=8):
    """Genera la gráfica del método del codo"""
    Values = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        Values.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), Values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para Determinar K Óptimo')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return Values

def evaluar_clusters_varios(df, target_col='num', ks=[2, 3, 4, 5]):
    """Evalúa clustering para varios valores de K"""
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
    grafica_metodo_codo(X_scaled, max_k=8)
    
    resultados = {}
    todos_labels = {}
    
    # 2. Evaluar cada K
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X_scaled)
        todos_labels[k] = labels
        
        # Calcular métricas básicas
        silhouette = silhouette_score(X_scaled, labels)
        
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
        
        resultados[k] = {
            'silhouette': silhouette,
            'distribucion': distribucion
        }
    
    # 3. Mostrar tabla comparativa simplificada

    print(f"{'K':<6} {'Silhouette':<12} {'Distribución'}")
    
    for k in ks:
        dist_str = ", ".join([f"C{i}: {d['total']}({d['tasa_enfermedad']:.1f}%)" 
                            for i, d in enumerate(resultados[k]['distribucion'])])
        print(f"{k:<6} {resultados[k]['silhouette']:<12.4f} {dist_str}")
    
    return {
        'resultados': resultados,
        'todos_labels': todos_labels,
        'X_scaled': X_scaled,
        'feature_names': feature_cols,
        'y_true': y
    }

def analizar_gravedad_por_cluster(df_clean, labels_k3, target_col='num'):
    """Analiza distribución de gravedad por cluster"""
    df_analysis = df_clean.copy()
    df_analysis['cluster'] = labels_k3
    
    print("\n Gravedad de casos para (K=3)")
    
    for cluster_id in range(3):
        cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
        total = len(cluster_data)
        
        print(f"\n CLUSTER {cluster_id} ({total} casos):")
        for severity in [0, 1, 2, 3, 4]:
            count = (cluster_data[target_col] == severity).sum()
            percentage = (count / total) * 100
            
            if severity == 0:
                label = "Sano"
            elif severity == 1:
                label = "Leve"
            elif severity == 2:
                label = "Moderado"
            else:
                label = "Severo"
                
            print(f"   {label}: {count} casos ({percentage:.1f}%)")

def analizar_caracteristicas_clusters(df_clean, labels_k3, feature_cols):
    """Analiza características distintivas por cluster"""
    df_analysis = df_clean.copy()
    df_analysis['cluster'] = labels_k3
    

    
    # Estadísticas por cluster
    stats_by_cluster = {}
    global_means = df_analysis[feature_cols].mean()
    
    for cluster_id in range(3):
        cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
        stats_by_cluster[cluster_id] = cluster_data[feature_cols].mean()
    
    # Identificar variables más distintivas

    
    for cluster_id in range(3):
        diferencias = []
        for variable in feature_cols:
            cluster_mean = stats_by_cluster[cluster_id][variable]
            global_mean = global_means[variable]
            diferencia_pct = ((cluster_mean - global_mean) / global_mean) * 100
            
            if abs(diferencia_pct) > 20:  # Diferencia significativa
                direccion = "↑" if diferencia_pct > 0 else "↓"
                diferencias.append((variable, diferencia_pct, direccion))
        
        # Ordenar por diferencia absoluta
        diferencias.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n  CLUSTER {cluster_id}:")
        if diferencias:
            for var, diff_pct, dir in diferencias[:5]:  # Top 5
                print(f"   {dir} {var}: {diff_pct:+.1f}%")
        else:
            print("   Sin variables muy distintivas")

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

if __name__ == "__main__":
    from data_loader import main_data
    
    
    # Cargar datos
    df_cleveland = main_data()
    
    # Evaluar clusters
    resultados = evaluar_clusters_varios(df_cleveland)
    
  # Evaluar clusters para K=2,3,4,5
    resultados_evaluacion = evaluar_clusters_varios(df_cleveland)
    
    # Plotear comparación 
    plotear(
       resultados_evaluacion['X_scaled'],
       resultados_evaluacion['todos_labels'][2],
       resultados_evaluacion['todos_labels'][3],
       resultados_evaluacion['todos_labels'][4]
   )
      
    # Análisis de gravedad
    df_clean = df_cleveland.copy()
    analizar_gravedad_por_cluster(df_clean, resultados['todos_labels'][3])
    
    # Análisis de características
    analizar_caracteristicas_clusters(df_clean, resultados['todos_labels'][3], resultados['feature_names'])
    
