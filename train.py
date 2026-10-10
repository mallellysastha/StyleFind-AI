"""StyleFind training: compare 3 models, tune, save best pipeline with Joblib.
Run once:  python train.py   (downloads Fashion-MNIST from OpenML)"""
import json, time, joblib, numpy as np
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Loading Fashion-MNIST dataset...")
X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="auto")
X, y = X.astype("float32"), y.astype(int)
Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]
print(f"Data ready. Train: {Xtr.shape}, Test: {Xte.shape}")

def pipe(clf):  # scaling + PCA + classifier in ONE pipeline (raw 0-255 pixels in)
    return Pipeline([("scale", MinMaxScaler()), ("pca", PCA(50, random_state=0)), ("clf", clf)])

candidates = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=300)),
                            {"clf__C": [0.1, 1, 10]}),
    "K-Nearest Neighbours": (pipe(KNeighborsClassifier()),
                             {"clf__n_neighbors": [3, 5, 9], "clf__weights": ["uniform", "distance"]}),
    "Random Forest": (pipe(RandomForestClassifier(random_state=0, n_jobs=-1)),
                      {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 20]}),
}

results, best_name, best_cv, best_model = {}, None, -1, None
for name, (p, grid) in candidates.items():
    print(f"\n--- Training & Tuning {name} ---")
    base = p.fit(Xtr[:10000], ytr[:10000])  # untuned baseline
    base_acc = accuracy_score(yte, base.predict(Xte))
    print(f"Baseline accuracy (10k untuned): {base_acc:.4f}")
    
    # tune with CV inside the training data only (scaler/PCA refit in each fold)
    gs = GridSearchCV(p, grid, cv=3, scoring="f1_macro", n_jobs=-1).fit(Xtr[:10000], ytr[:10000])
    print(f"Best CV Macro F1: {gs.best_score_:.4f} with params: {gs.best_params_}")
    
    t = time.time(); model = gs.best_estimator_.fit(Xtr, ytr); train_s = time.time() - t
    t = time.time(); pred = model.predict(Xte); pred_s = time.time() - t
    test_acc = accuracy_score(yte, pred)
    test_f1 = f1_score(yte, pred, average="macro")
    print(f"Full fit Test Accuracy: {test_acc:.4f}, Test Macro F1: {test_f1:.4f}")
    
    results[name] = {
        "baseline_accuracy": float(base_acc), "cv_macro_f1": float(gs.best_score_),
        "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "train_seconds": float(train_s), "predict_seconds": float(pred_s),
        "confusion": confusion_matrix(yte, pred).tolist(),
    }
    if gs.best_score_ > best_cv:  # winner chosen by CV score, not by test score
        best_name, best_cv, best_model = name, gs.best_score_, model

results["_winner"] = best_name
with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
joblib.dump(best_model, "model.joblib")

# Assets for the website: catalogue for visual search + unseen test samples
print("\nExtracting PCA embeddings for 10k catalogue...")
emb = best_model[:-1].transform(Xtr[:10000])
np.savez_compressed("assets.npz", cat_img=Xtr[:10000].astype("uint8"), cat_lab=ytr[:10000],
                    cat_emb=emb.astype("float32"),
                    test_img=Xte[:300].astype("uint8"), test_lab=yte[:300],
                    counts=np.bincount(ytr))
print("Saved model.joblib, results.json, assets.npz | winner:", best_name)
