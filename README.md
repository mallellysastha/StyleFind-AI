# 🌐 StyleFind AI: Model Arena & Image Clustering

> **Live Public URL**: [https://mallellysastha.github.io/StyleFind-AI/](https://mallellysastha.github.io/StyleFind-AI/)


**StyleFind AI** is an interactive computer vision and machine learning platform combining **Supervised Classification** and **Unsupervised Image Clustering** powered by Scikit-Learn, Streamlit, and Plotly.

---

## ?? Live Access

The application is running locally at:
- **URL**: [http://localhost:8501](http://localhost:8501)
- If you need to restart it anytime:
  ```bash
  python -m streamlit run app.py
  ```

---

## ?? Key Application Sections

### 1. ?? Style Prediction (Supervised Learning)
- **Workflow**: Upload Image &rarr; Contrast/Inversion Preprocessing &rarr; 50-D PCA Feature Extraction &rarr; Scikit-Learn Model &rarr; Style Prediction &rarr; Confidence Score.
- **Input Sources**: User photo file upload (JPG/PNG/WEBP), live webcam snapshot, or 300 unseen Fashion-MNIST test samples.
- **Visual Feedback**: Real-time predicted class badge, confidence score, probability breakdown bars, 28x28 preprocessed model preview, and 5-NN visual similarity search.

### 2. ? AI Image Clustering (Unsupervised Learning)
- **Workflow**: Multi-image batch &rarr; Standardized Preprocessing &rarr; 50-D Feature Vectors &rarr; K-Means / Hierarchical / DBSCAN &rarr; 2D PCA Space &rarr; Interactive Cluster Visualizer &rarr; Cluster Gallery.
- **Dataset Options**: Upload 1050 custom images or 1-click Quick-Load curated Fashion Batches (15, 30, or 50 diverse items).
- **Interactive 2D Visualization**: Interactive Plotly scatter plot mapping images in PCA space with cluster color-coding and item hover details.
- **Elbow Method Optimizer**: "Find Optimal K" calculates inertia across $K=2 \dots 8$ and mathematically detects the optimal knee/elbow point.
- **Cluster Gallery Cards**: Visual card per cluster with thumbnail grids and expandable inspections.
- **Dynamic KPI Strip**: Images Uploaded, Selected Algorithm, Number of Clusters ($K$), Silhouette Score, and Execution Latency (ms).

### 3. ?? Algorithm Comparison & Benchmarks
- Side-by-side benchmark of **K-Means**, **Hierarchical Clustering (Agglomerative)**, and **DBSCAN** on the exact same feature vectors.
- Genuine calculated metrics: **Silhouette Score**, **Davies-Bouldin Index**, **Clusters Formed**, and **Execution Latency (ms)**.
- Educational Viva / College Presentation reference cards explaining algorithm type, purpose, inputs, outputs, and parameters.
- 6 Real-World Industry Application showcases (E-commerce grouping, visual search, recommendation systems, etc.).

---

## ?? File Structure

| File | Purpose |
|------|---------|
| [`app.py`](app.py) | Modern Streamlit dashboard with Dark Navy glassmorphic UI |
| [`clustering_engine.py`](clustering_engine.py) | Modular ML engine: feature extraction, PCA 2D projection, K-Means, Hierarchical, DBSCAN, Silhouette & DB metrics, Elbow locator |
| [`test_verification.py`](test_verification.py) | Verification test script validating clustering, metrics, and classification |
| [`model.joblib`](model.joblib) | Trained Scikit-Learn pipeline (`MinMaxScaler` &rarr; `PCA(50)` &rarr; `Classifier`) |
| [`assets.npz`](assets.npz) | Fashion-MNIST catalogue images, embeddings, and test samples |
| [`results.json`](results.json) | Cross-validation benchmarks and confusion matrix data |
| [`requirements.txt`](requirements.txt) | Project dependencies including Scikit-Learn, Streamlit, and Plotly |
