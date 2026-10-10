import joblib, numpy as np, pandas as pd
import clustering_engine as ce

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

A = np.load('assets.npz')
model = joblib.load('model.joblib')
fe = model[:-1]
rf = model.named_steps['clf']
rf_weights = rf.feature_importances_

# 1. Prediction pipeline
test_img = A['test_img'][0].reshape(28, 28)
pred_prob = model.predict_proba(test_img.reshape(1, -1))[0]
top_idx = np.argmax(pred_prob)
print(f"Prediction Pipeline: Predicted {CLASSES[top_idx]} ({pred_prob[top_idx]:.2%})")

# 2. Batch Feature Extraction with and without RF Weighting
batch = [A['test_img'][i].reshape(28, 28) for i in range(25)]
batch_std = ce.extract_features_batch(batch, feature_extractor=fe, use_rf_weighting=False)
batch_rf = ce.extract_features_batch(batch, feature_extractor=fe, rf_importances=rf_weights, use_rf_weighting=True)
assert batch_std['features'].shape == (25, 50)
assert batch_rf['features'].shape == (25, 50)
print("Batch Feature Extraction: Standard and RF-Weighted shapes valid (25, 50)")

# 3. Random Forest Batch Predictions & Cluster Purity Insights
rf_items = ce.analyze_with_random_forest(batch_rf['raw_784'], model, CLASSES)
km = ce.run_kmeans_clustering(batch_rf['features'], n_clusters=3)
insights = ce.compute_rf_cluster_insights(km['labels'], rf_items, CLASSES)
print("Random Forest Cluster Purity Insights computed:")
for c_id, c_data in insights.items():
    print(f"  Cluster {c_id + 1}: Dominant Style = {c_data['dominant_style']}, Purity = {c_data['purity_pct']}%, Avg Conf = {c_data['avg_confidence_pct']}%")

print("All tests passed successfully!")
