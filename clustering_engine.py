"""
Clustering Engine for StyleFind AI
Enhanced with Random Forest Feature Weighting and Cluster Analysis.
Handles feature extraction, dimensionality reduction (PCA), clustering algorithms (K-Means, Hierarchical, DBSCAN),
evaluation metrics (Silhouette, Davies-Bouldin, Inertia), and Elbow Method calculation.
"""
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

def preprocess_single_image(img: Image.Image) -> np.ndarray:
    """
    Standardize any user or sample photo into a 28x28 grayscale normalized array.
    Light item on dark background to match Fashion-MNIST representation.
    """
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, (0, 0), img.convert("RGBA"))
        img = bg.convert("RGB")
    
    # Auto-contrast and convert to grayscale
    g = ImageOps.autocontrast(img.convert("L"))
    arr = np.asarray(g, dtype=np.float32)
    
    # Detect light background and invert polarity
    if arr[[0, -1], :].mean() > 127:
        g = ImageOps.invert(g)
        
    # Aspect-ratio preserving padding to 28x28
    g = ImageOps.pad(g, (28, 28), color=0)
    return np.asarray(g, dtype=np.float32)

def extract_features_batch(images_pil_or_arr, feature_extractor=None, rf_importances=None, use_rf_weighting=False):
    """
    Given a list of PIL Images or (28, 28) numpy arrays,
    extract flattened pixel vectors and project to feature vectors.
    If feature_extractor (e.g. model[:-1]) is provided, transform through MinMaxScaler + PCA(50).
    If use_rf_weighting is True, scale features by Random Forest importance weights.
    Also compute 2D PCA projection for interactive scatter visualization.
    """
    processed_28x28 = []
    flattened_784 = []
    
    for item in images_pil_or_arr:
        if isinstance(item, Image.Image):
            arr = preprocess_single_image(item)
        elif isinstance(item, np.ndarray):
            if item.shape == (28, 28):
                arr = item.astype(np.float32)
            elif item.shape == (784,):
                arr = item.reshape(28, 28).astype(np.float32)
            else:
                pil_img = Image.fromarray(item.astype(np.uint8))
                arr = preprocess_single_image(pil_img)
        else:
            pil_img = Image.open(item)
            arr = preprocess_single_image(pil_img)
            
        processed_28x28.append(arr)
        flattened_784.append(arr.reshape(-1))
        
    X_raw = np.array(flattened_784, dtype=np.float32)
    
    # Feature extraction: use trained pipeline's [MinMaxScaler + PCA(50)]
    if feature_extractor is not None:
        try:
            X_features = feature_extractor.transform(X_raw)
        except Exception:
            X_norm = X_raw / 255.0
            pca_fallback = PCA(n_components=min(50, X_norm.shape[0], X_norm.shape[1]), random_state=42)
            X_features = pca_fallback.fit_transform(X_norm)
    else:
        X_norm = X_raw / 255.0
        pca_fallback = PCA(n_components=min(50, X_norm.shape[0], X_norm.shape[1]), random_state=42)
        X_features = pca_fallback.fit_transform(X_norm)
        
    # Apply Random Forest Feature Weighting if selected
    if use_rf_weighting and rf_importances is not None:
        weights = np.asarray(rf_importances, dtype=np.float32)
        if len(weights) == X_features.shape[1]:
            # Scale features by square root of normalized weights
            norm_weights = np.sqrt(weights / (np.max(weights) + 1e-7))
            X_features = X_features * norm_weights
            
    # Compute 2D PCA coordinates strictly for 2D visualization
    n_samples = len(X_features)
    if n_samples >= 2:
        pca_2d = PCA(n_components=2, random_state=42)
        coords_2d = pca_2d.fit_transform(X_features)
        var_ratio = [float(v) for v in pca_2d.explained_variance_ratio_]
    else:
        coords_2d = np.zeros((n_samples, 2), dtype=np.float32)
        var_ratio = [0.0, 0.0]
        
    return {
        "processed_images": processed_28x28,
        "raw_784": X_raw,
        "features": X_features,
        "coords_2d": coords_2d,
        "explained_variance_2d": var_ratio,
        "n_samples": n_samples
    }

def analyze_with_random_forest(raw_784, rf_model, classes, class_icons=None):
    """
    Run the Random Forest Classifier on all batch images to get
    individual predictions, probabilities, and confidences.
    """
    probas = rf_model.predict_proba(raw_784)
    preds = np.argmax(probas, axis=1)
    confidences = np.max(probas, axis=1)
    
    results = []
    for i in range(len(preds)):
        c_idx = preds[i]
        c_name = classes[c_idx]
        icon = class_icons.get(c_name, "??") if class_icons else "??"
        results.append({
            "pred_idx": int(c_idx),
            "pred_name": c_name,
            "pred_icon": icon,
            "confidence": float(confidences[i]),
            "proba": probas[i]
        })
    return results

def compute_rf_cluster_insights(labels, rf_results, classes, class_icons=None):
    """
    Compute Random Forest dominant styles, purities, and average confidences for each cluster.
    """
    unique_clusters = np.unique(labels)
    unique_clusters = sorted(unique_clusters, key=lambda x: (x == -1, x))
    cluster_insights = {}
    
    for c_id in unique_clusters:
        mask = (labels == c_id)
        indices = np.where(mask)[0]
        c_count = len(indices)
        
        if c_count == 0:
            continue
            
        # Get RF predictions for members of this cluster
        member_preds = [rf_results[idx]["pred_name"] for idx in indices]
        member_confs = [rf_results[idx]["confidence"] for idx in indices]
        
        # Dominant category
        from collections import Counter
        counts = Counter(member_preds)
        dominant_style, dominant_count = counts.most_common(1)[0]
        purity_pct = (dominant_count / c_count) * 100.0
        avg_conf_pct = (np.mean(member_confs)) * 100.0
        dominant_icon = class_icons.get(dominant_style, "??") if class_icons else "??"
        
        cluster_insights[c_id] = {
            "dominant_style": dominant_style,
            "dominant_icon": dominant_icon,
            "purity_pct": round(purity_pct, 1),
            "avg_confidence_pct": round(avg_conf_pct, 1),
            "distribution": dict(counts)
        }
        
    return cluster_insights

def run_kmeans_clustering(features, n_clusters=3, random_state=42):
    """
    Run scikit-learn K-Means on extracted feature vectors.
    """
    n_samples = len(features)
    k = max(2, min(n_clusters, n_samples))
    
    start_time = time.perf_counter()
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=random_state)
    labels = kmeans.fit_predict(features)
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1 and n_samples > len(unique_labels):
        sil_score = float(silhouette_score(features, labels))
        db_score = float(davies_bouldin_score(features, labels))
    else:
        sil_score = 0.0
        db_score = 0.0
        
    return {
        "algorithm": "K-Means",
        "labels": labels,
        "n_clusters": k,
        "inertia": float(kmeans.inertia_),
        "centroids": kmeans.cluster_centers_,
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score,
        "execution_time_ms": round(latency_ms, 2),
        "model": kmeans
    }

def run_hierarchical_clustering(features, n_clusters=3):
    """
    Run Agglomerative (Hierarchical) Clustering on extracted feature vectors.
    """
    n_samples = len(features)
    k = max(2, min(n_clusters, n_samples))
    
    start_time = time.perf_counter()
    agg = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
    labels = agg.fit_predict(features)
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1 and n_samples > len(unique_labels):
        sil_score = float(silhouette_score(features, labels))
        db_score = float(davies_bouldin_score(features, labels))
    else:
        sil_score = 0.0
        db_score = 0.0
        
    return {
        "algorithm": "Hierarchical Clustering",
        "labels": labels,
        "n_clusters": k,
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score,
        "execution_time_ms": round(latency_ms, 2),
        "model": agg
    }

def run_dbscan_clustering(features, eps=None, min_samples=2):
    """
    Run DBSCAN density-based clustering on extracted feature vectors.
    """
    n_samples = len(features)
    if eps is None:
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(features)
        np.fill_diagonal(dists, np.nan)
        med_dist = np.nanmedian(dists)
        eps = float(med_dist * 0.75) if not np.isnan(med_dist) else 0.5
        
    start_time = time.perf_counter()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    non_noise_mask = (labels != -1)
    unique_non_noise = np.unique(labels[non_noise_mask])
    n_clusters_found = len(unique_non_noise)
    
    if n_clusters_found > 1 and np.sum(non_noise_mask) > n_clusters_found:
        sil_score = float(silhouette_score(features[non_noise_mask], labels[non_noise_mask]))
        db_score = float(davies_bouldin_score(features[non_noise_mask], labels[non_noise_mask]))
    else:
        sil_score = 0.0
        db_score = 0.0
        
    return {
        "algorithm": "DBSCAN",
        "labels": labels,
        "n_clusters": n_clusters_found,
        "noise_count": int(np.sum(labels == -1)),
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score,
        "execution_time_ms": round(latency_ms, 2),
        "eps_used": round(eps, 3),
        "model": dbscan
    }

def compute_elbow_curve(features, min_k=2, max_k=8):
    """
    Calculate K-Means inertia across different K values and mathematically
    locate the optimal elbow point using maximum perpendicular distance to the chord line.
    """
    n_samples = len(features)
    max_k_adjusted = min(max_k, n_samples - 1)
    if max_k_adjusted < min_k:
        max_k_adjusted = min_k
        
    k_values = list(range(min_k, max_k_adjusted + 1))
    inertias = []
    
    for k in k_values:
        km = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=42)
        km.fit(features)
        inertias.append(float(km.inertia_))
        
    if len(k_values) >= 3:
        p1 = np.array([k_values[0], inertias[0]])
        p2 = np.array([k_values[-1], inertias[-1]])
        line_vec = p2 - p1
        line_norm = np.linalg.norm(line_vec)
        
        distances = []
        for i in range(len(k_values)):
            p = np.array([k_values[i], inertias[i]])
            dist = np.abs(np.cross(p2 - p1, p1 - p)) / (line_norm if line_norm != 0 else 1.0)
            distances.append(dist)
            
        optimal_idx = int(np.argmax(distances))
        optimal_k = k_values[optimal_idx]
    else:
        optimal_k = k_values[0]
        
    return {
        "k_values": k_values,
        "inertias": inertias,
        "optimal_k": optimal_k,
        "explanation": f"Evaluated K-Means inertia across K={k_values[0]} to K={k_values[-1]}. The knee point calculation identifies K={optimal_k} as the point of diminishing marginal returns (maximum elbow curvature)."
    }

def benchmark_all_clustering(features, n_clusters=3):
    """
    Runs K-Means, Hierarchical, and DBSCAN on the same dataset and returns side-by-side metrics.
    """
    res_km = run_kmeans_clustering(features, n_clusters=n_clusters)
    res_agg = run_hierarchical_clustering(features, n_clusters=n_clusters)
    res_db = run_dbscan_clustering(features)
    
    summary = [
        {
            "Algorithm": "K-Means",
            "Type": "Centroid-Based Partitioning",
            "Clusters Formed": res_km["n_clusters"],
            "Silhouette Score": res_km["silhouette_score"],
            "Davies-Bouldin Index": res_km["davies_bouldin_score"],
            "Execution Time (ms)": res_km["execution_time_ms"],
            "Status": "Primary Implemented"
        },
        {
            "Algorithm": "Hierarchical Clustering",
            "Type": "Agglomerative (Ward Linkage)",
            "Clusters Formed": res_agg["n_clusters"],
            "Silhouette Score": res_agg["silhouette_score"],
            "Davies-Bouldin Index": res_agg["davies_bouldin_score"],
            "Execution Time (ms)": res_agg["execution_time_ms"],
            "Status": "Implemented"
        },
        {
            "Algorithm": "DBSCAN",
            "Type": "Density-Based Spatial Clustering",
            "Clusters Formed": f"{res_db['n_clusters']} (+{res_db['noise_count']} noise)" if res_db['noise_count'] > 0 else f"{res_db['n_clusters']}",
            "Silhouette Score": res_db["silhouette_score"],
            "Davies-Bouldin Index": res_db["davies_bouldin_score"],
            "Execution Time (ms)": res_db["execution_time_ms"],
            "Status": "Implemented"
        }
    ]
    
    return {
        "table": pd.DataFrame(summary),
        "kmeans": res_km,
        "hierarchical": res_agg,
        "dbscan": res_db
    }
