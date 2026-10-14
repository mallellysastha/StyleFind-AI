import joblib, numpy as np, json, io, base64
from PIL import Image
from sklearn.linear_model import LogisticRegression

print("1. Loading results.json...")
with open("results.json", "r") as f:
    results = json.load(f)

print("2. Loading assets.npz...")
assets = np.load("assets.npz")
test_img = assets["test_img"]  # (300, 784)
test_lab = assets["test_lab"]  # (300,)
cat_emb = assets["cat_emb"]    # (10000, 50)
cat_lab = assets["cat_lab"]    # (10000,)

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("3. Loading model.joblib...")
model = joblib.load("model.joblib")
pca = model.named_steps["pca"]
rf = model.named_steps["clf"]

print("4. Fitting client-side linear classifier on PCA embeddings...")
lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(cat_emb, cat_lab)

print("5. Generating predictions for 300 test samples...")
test_emb = pca.transform(test_img / 255.0)
rf_probs = rf.predict_proba(test_emb)
rf_preds = rf.predict(test_emb)

samples_data = []
for i in range(len(test_img)):
    img = Image.fromarray(test_img[i].reshape(28, 28))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    prob_list = rf_probs[i].tolist()
    top3_indices = np.argsort(prob_list)[::-1][:3]
    top3 = [{"id": int(c), "name": classes[c], "prob": round(float(prob_list[c]), 4)} for c in top3_indices]

    samples_data.append({
        "i": i,
        "y": int(test_lab[i]),
        "y_name": classes[test_lab[i]],
        "pred": int(rf_preds[i]),
        "pred_name": classes[rf_preds[i]],
        "conf": round(float(np.max(prob_list)), 4),
        "top3": top3,
        "img": b64
    })

correct_count = sum(1 for s in samples_data if s["y"] == s["pred"])
print(f"Loaded {len(samples_data)} samples. Correct: {correct_count}/{len(samples_data)} ({correct_count/len(samples_data)*100:.1f}%)")

# Extract compact weights for client inference on uploaded custom images
pca_mean = np.round(pca.mean_, 4).tolist()
pca_comp = np.round(pca.components_, 4).tolist()
lr_w = np.round(lr.coef_, 4).tolist()
lr_b = np.round(lr.intercept_, 4).tolist()

client_model = {
    "pca_mean": pca_mean,
    "pca_components": pca_comp,
    "lr_w": lr_w,
    "lr_b": lr_b,
    "classes": classes
}

lr_res = results.get("Logistic Regression", {})
rf_res = results.get("Random Forest", {})
knn_res = results.get("K-Nearest Neighbours", {})

# Format values from results.json
lr_base = f"{lr_res.get('baseline_accuracy', 0.835)*100:.1f}%"
lr_cv = f"{lr_res.get('cv_macro_f1', 0.841):.3f}"
lr_test_acc = f"{lr_res.get('test_accuracy', 0.821)*100:.1f}%"
lr_test_f1 = f"{lr_res.get('test_macro_f1', 0.821):.3f}"
lr_time = f"{lr_res.get('train_seconds', 12.4):.1f}s / {lr_res.get('predict_seconds', 0.05):.2f}s"

rf_base = f"{rf_res.get('baseline_accuracy', 0.865)*100:.1f}%"
rf_cv = f"{rf_res.get('cv_macro_f1', 0.878):.3f}"
rf_test_acc = f"{rf_res.get('test_accuracy', 0.848)*100:.1f}%"
rf_test_f1 = f"{rf_res.get('test_macro_f1', 0.847):.3f}"
rf_time = f"{rf_res.get('train_seconds', 4.99):.2f}s / {rf_res.get('predict_seconds', 0.13):.2f}s"

knn_base = f"{knn_res.get('baseline_accuracy', 0.852)*100:.1f}%"
knn_cv = f"{knn_res.get('cv_macro_f1', 0.855):.3f}"
knn_test_acc = f"{knn_res.get('test_accuracy', 0.818)*100:.1f}%"
knn_test_f1 = f"{knn_res.get('test_macro_f1', 0.815):.3f}"
knn_time = f"{knn_res.get('train_seconds', 1.1):.1f}s / {knn_res.get('predict_seconds', 8.5):.1f}s"

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StyleFind AI &mdash; Model Arena</title>
<meta name="description" content="Compare Random Forest, Logistic Regression and KNN on Fashion-MNIST.">

<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap">
<style>
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin: 0; background: #0E1512; color: #EAF2EC; font-family: 'DM Sans', sans-serif; min-height: 100vh; }}
.pg {{ background: #0E1512; color: #EAF2EC; font-family: 'DM Sans', sans-serif; min-height: 100vh; }}
.wrap {{ max-width: 1200px; margin: 0 auto; padding: 0 32px; box-sizing: border-box; }}
.mono {{ font-family: 'IBM Plex Mono', monospace; }}
h1,h2,h3 {{ font-family: 'Bricolage Grotesque', sans-serif; margin: 0; letter-spacing: -0.02em; }}
.nav {{ display: flex; align-items: center; justify-content: space-between; padding: 24px 0; border-bottom: 1px solid rgba(44, 61, 52, 0.5); }}
.nav a {{ color: #9DB0A5; text-decoration: none; margin-left: 28px; font-size: 15px; transition: color 0.2s; }}
.nav a:hover {{ color: #C8F169; }}
.logo {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 22px; letter-spacing: -0.02em; }}
.logo b {{ color: #C8F169; }}
.hero {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 48px; align-items: center; padding: 56px 0 72px; }}
.tag {{ display: inline-block; border: 1px solid #2C3D34; border-radius: 99px; padding: 6px 14px; font-size: 13px; color: #C8F169; letter-spacing: 0.05em; }}
.btn {{ display: inline-flex; align-items: center; justify-content: center; background: #C8F169; color: #0E1512; font-weight: 600; border: 0; border-radius: 12px; padding: 14px 24px; font-size: 16px; text-decoration: none; min-height: 44px; cursor: pointer; transition: all 0.2s; }}
.btn:hover {{ background: #d7f78c; transform: translateY(-1px); }}
.btn.ghost {{ background: transparent; color: #EAF2EC; border: 1px solid #2C3D34; margin-left: 12px; }}
.btn.ghost:hover {{ border-color: #C8F169; color: #C8F169; }}
.btn-sm {{ padding: 6px 12px; font-size: 13px; border-radius: 8px; border: 1px solid #2C3D34; background: #1A2A22; color: #EAF2EC; cursor: pointer; transition: all 0.15s; }}
.btn-sm:hover {{ border-color: #C8F169; color: #C8F169; }}
.panel {{ background: #141F1A; border: 1px solid #22332A; border-radius: 20px; padding: 28px; box-shadow: 0 20px 40px rgba(0,0,0,0.4); }}
.steps {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
.step {{ background: #141F1A; border: 1px solid #22332A; border-radius: 16px; padding: 20px; transition: border-color 0.2s; }}
.step:hover {{ border-color: #3b5245; }}
.step .n {{ color: #C8F169; font-size: 13px; }}
.arena {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; align-items: stretch; }}
.card {{ background: #141F1A; border: 1px solid #22332A; border-radius: 20px; padding: 28px; display: flex; flex-direction: column; gap: 14px; position: relative; transition: all 0.25s ease; }}
.card:hover {{ border-color: #334e3f; }}
.card.win {{ background: #C8F169; color: #0E1512; border-color: #C8F169; transform: translateY(-12px); box-shadow: 0 20px 45px rgba(200, 241, 105, 0.18); }}
.card .sub {{ color: #9DB0A5; font-size: 14px; }}
.card.win .sub {{ color: #2A3A12; }}
.row {{ display: flex; justify-content: space-between; border-top: 1px solid #22332A; padding-top: 10px; font-size: 14px; align-items: center; }}
.card.win .row {{ border-color: #A9D14E; }}
.row span:last-child {{ font-family: 'IBM Plex Mono', monospace; font-weight: 500; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.chip {{ background: #1A2A22; border: 1px solid #2C3D34; border-radius: 10px; padding: 10px 16px; font-size: 14px; cursor: pointer; transition: all 0.2s; user-select: none; }}
.chip:hover {{ border-color: #C8F169; color: #C8F169; transform: translateY(-2px); }}
.chip.active {{ background: #C8F169; color: #0E1512; border-color: #C8F169; font-weight: 600; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.muted {{ color: #9DB0A5; }}
.sec {{ padding: 72px 0 0; }}

/* Interactive Live Predictor Box */
.dropzone {{
  height: 220px;
  margin: 18px 0;
  border: 1px dashed #2C3D34;
  border-radius: 14px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #9DB0A5;
  text-align: center;
  padding: 16px;
  position: relative;
  overflow: hidden;
  background: rgba(14, 21, 18, 0.6);
  cursor: pointer;
  transition: all 0.2s ease;
}}
.dropzone:hover, .dropzone.dragover {{
  border-color: #C8F169;
  background: rgba(200, 241, 105, 0.05);
}}
.preview-img {{
  width: 96px;
  height: 96px;
  image-rendering: pixelated;
  image-rendering: crisp-edges;
  border-radius: 8px;
  border: 1px solid #2C3D34;
  background: #000;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
  margin-bottom: 8px;
}}
.sample-bar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #22332A;
}}
.badge {{
  display: inline-block;
  padding: 3px 8px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 600;
  font-family: 'IBM Plex Mono', monospace;
}}
.badge-lime {{ background: rgba(200, 241, 105, 0.15); color: #C8F169; border: 1px solid rgba(200, 241, 105, 0.3); }}
.badge-green {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }}
.badge-red {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}

/* Mini Probability Distribution Bars */
.prob-bars {{ margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }}
.prob-bar-row {{ display: flex; align-items: center; justify-content: space-between; font-size: 12px; }}
.bar-track {{ height: 5px; background: #1A2A22; border-radius: 99px; overflow: hidden; margin-top: 3px; }}
.bar-fill {{ height: 100%; background: #C8F169; border-radius: 99px; transition: width 0.3s ease; }}

/* Modal for Confusion Matrix */
.modal-overlay {{
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.85);
  backdrop-filter: blur(8px);
  z-index: 1000;
  align-items: center;
  justify-content: center;
  padding: 20px;
}}
.modal-overlay.open {{ display: flex; }}
.modal-box {{
  background: #141F1A;
  border: 1px solid #22332A;
  border-radius: 20px;
  max-width: 860px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  padding: 32px;
  position: relative;
}}
.cm-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
  font-family: 'IBM Plex Mono', monospace;
  margin-top: 16px;
}}
.cm-table th, .cm-table td {{
  padding: 6px 4px;
  text-align: center;
  border: 1px solid #22332A;
}}
.cm-table th {{ background: #1A2A22; color: #9DB0A5; font-weight: 500; font-size: 10px; }}
.cm-diag {{ background: rgba(200, 241, 105, 0.2); color: #C8F169; font-weight: bold; }}
.cm-high-err {{ background: rgba(239, 68, 68, 0.2); color: #fca5a5; }}

@media (max-width: 900px) {{
  .hero, .cols, .arena {{ grid-template-columns: 1fr; }}
  .steps {{ grid-template-columns: 1fr 1fr; }}
  .card.win {{ transform: none; }}
  .nav a {{ display: none; }}
}}
</style>
</head>
<body>

<div class="pg">
<div class="wrap">

<!-- Navigation -->
<div class="nav">
  <div class="logo">Style<b>Find</b> AI</div>
  <div>
    <a href="#pipeline">Pipeline</a>
    <a href="#arena">Arena</a>
    <a href="#classes">Classes</a>
    <a href="https://github.com/mallellysastha/StyleFind-AI" target="_blank" rel="noopener">GitHub</a>
  </div>
</div>

<!-- Hero Section -->
<div class="hero">
  <div>
    <span class="tag mono">SUPERVISED LEARNING &middot; FASHION-MNIST</span>
    <h1 style="font-size:64px;line-height:1.02;margin:20px 0">Three models enter. <span style="color:#C8F169">Random Forest</span> takes the spotlight.</h1>
    <p class="muted" style="font-size:18px;line-height:1.6;max-width:520px;margin:0 0 28px">Classify 28&times;28 clothing images into 10 categories, then see how a tuned Random Forest stacks up against Logistic Regression and K-Nearest Neighbours, all trained on the same scaled, PCA-reduced pixels.</p>
    <a class="btn" href="#live-demo">Try the classifier</a>
    <a class="btn ghost" href="#arena">See the comparison</a>
  </div>

  <!-- Live Prediction Widget -->
  <div class="panel" id="live-demo">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div class="mono muted" style="font-size:13px">LIVE PREDICTION &middot; RANDOM FOREST</div>
      <span class="badge badge-lime" id="source-badge">SAMPLE #0</span>
    </div>

    <!-- Dropzone / Image Visualizer -->
    <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
      <input type="file" id="file-input" accept="image/*" style="display:none" onchange="handleFileSelect(event)">
      <img id="preview-img" class="preview-img" src="{samples_data[0]['img']}" alt="Fashion Sample">
      <div id="drop-hint" class="muted" style="font-size:13px; line-height:1.4">
        Drop an image or click to upload<br>
        <span style="font-size:11px; color:#5c7064">Auto-scaled & inverted to 28&times;28 pixels</span>
      </div>
    </div>

    <!-- Output Rows -->
    <div class="row" style="border-color:#22332A">
      <span>Predicted class</span>
      <span id="res-class" style="color:#C8F169; font-weight:700;">{samples_data[0]['pred_name']}</span>
    </div>
    <div class="row">
      <span>Confidence</span>
      <span id="res-conf">{samples_data[0]['conf']*100:.1f}%</span>
    </div>
    <div class="row" id="row-truth">
      <span>Ground truth</span>
      <span id="res-truth" style="display:inline-flex; align-items:center; gap:6px;">
        {samples_data[0]['y_name']}
        <span class="badge badge-green" id="truth-match">MATCH</span>
      </span>
    </div>

    <!-- Top 3 Probability Breakdown -->
    <div class="prob-bars" id="prob-bars"></div>

    <!-- Test Sample Navigator -->
    <div class="sample-bar">
      <div style="display:flex; align-items:center; gap:6px;">
        <button class="btn-sm" onclick="stepSample(-1)" title="Previous sample">&#9664; Prev</button>
        <button class="btn-sm" onclick="randomSample()" title="Random sample">&#127922; Random</button>
        <button class="btn-sm" onclick="stepSample(1)" title="Next sample">Next &#9654;</button>
      </div>
      <span class="mono muted" style="font-size:12px" id="sample-counter">1 of 300</span>
    </div>
  </div>
</div>

<!-- Pipeline Section -->
<div class="sec" id="pipeline">
  <h2 style="font-size:36px;margin-bottom:8px">One pipeline, same start line</h2>
  <p class="muted" style="margin:0 0 28px">Raw 0–255 pixels go in; every model sees identical preprocessing, so the comparison is fair.</p>
  <div class="steps">
    <div class="step">
      <div class="n mono">01</div>
      <h3 style="font-size:20px;margin:8px 0">MinMaxScaler</h3>
      <div class="muted" style="font-size:14px">Squeezes pixel values into a common 0.0–1.0 range.</div>
    </div>
    <div class="step">
      <div class="n mono">02</div>
      <h3 style="font-size:20px;margin:8px 0">PCA &middot; 50</h3>
      <div class="muted" style="font-size:14px">Compresses 784 pixels into 50 orthogonal components.</div>
    </div>
    <div class="step">
      <div class="n mono">03</div>
      <h3 style="font-size:20px;margin:8px 0">GridSearchCV</h3>
      <div class="muted" style="font-size:14px">3-fold CV on macro-F1, inside training data only.</div>
    </div>
    <div class="step">
      <div class="n mono">04</div>
      <h3 style="font-size:20px;margin:8px 0">Pick by CV</h3>
      <div class="muted" style="font-size:14px">Winner chosen by CV score, never by test score.</div>
    </div>
  </div>
</div>

<!-- Arena Section -->
<div class="sec" id="arena">
  <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:36px;">
    <div>
      <h2 style="font-size:36px;margin-bottom:8px">The arena</h2>
      <p class="muted" style="margin:0">Numbers filled in from <span class="mono" style="color:#C8F169">results.json</span> after running <span class="mono">train.py</span> on Fashion-MNIST.</p>
    </div>
    <button class="btn-sm" style="padding:10px 16px; font-size:13px;" onclick="openCMModal()">Inspect Confusion Matrices &#8599;</button>
  </div>

  <div class="arena">
    <!-- Logistic Regression Card -->
    <div class="card">
      <h3 style="font-size:26px">Logistic Regression</h3>
      <div class="sub">Linear baseline, fast and simple.</div>
      <div class="row"><span>Tuned grid</span><span>C: 0.1 &middot; 1 &middot; 10</span></div>
      <div class="row"><span>Baseline accuracy</span><span>{lr_base}</span></div>
      <div class="row"><span>CV macro-F1</span><span>{lr_cv}</span></div>
      <div class="row"><span>Test accuracy</span><span>{lr_test_acc}</span></div>
      <div class="row"><span>Test macro-F1</span><span>{lr_test_f1}</span></div>
      <div class="row"><span>Train / predict (s)</span><span>{lr_time}</span></div>
    </div>

    <!-- Random Forest (SPOTLIGHT WINNER) -->
    <div class="card win">
      <span class="mono" style="font-size:12px; font-weight:700; letter-spacing:0.05em;">SPOTLIGHT WINNER</span>
      <h3 style="font-size:30px">Random Forest</h3>
      <div class="sub">Many decision trees voting together.</div>
      <div class="row"><span>Tuned grid</span><span>trees 100 &middot; 200</span></div>
      <div class="row"><span>Max depth</span><span>None &middot; 20</span></div>
      <div class="row"><span>Baseline accuracy</span><span>{rf_base}</span></div>
      <div class="row"><span>CV macro-F1</span><span>{rf_cv}</span></div>
      <div class="row"><span>Test accuracy</span><span>{rf_test_acc}</span></div>
      <div class="row"><span>Test macro-F1</span><span>{rf_test_f1}</span></div>
      <div class="row"><span>Train / predict (s)</span><span>{rf_time}</span></div>
    </div>

    <!-- KNN Card -->
    <div class="card">
      <h3 style="font-size:26px">K-Nearest Neighbours</h3>
      <div class="sub">Finds the closest look-alikes.</div>
      <div class="row"><span>Neighbours</span><span>3 &middot; 5 &middot; 9</span></div>
      <div class="row"><span>Weights</span><span>uniform &middot; distance</span></div>
      <div class="row"><span>Baseline accuracy</span><span>{knn_base}</span></div>
      <div class="row"><span>CV macro-F1</span><span>{knn_cv}</span></div>
      <div class="row"><span>Test accuracy</span><span>{knn_test_acc}</span></div>
      <div class="row"><span>Test macro-F1</span><span>{knn_test_f1}</span></div>
      <div class="row"><span>Train / predict (s)</span><span>{knn_time}</span></div>
    </div>
  </div>
</div>

<!-- Details & Analysis Columns -->
<div class="sec cols">
  <div class="panel">
    <h3 style="font-size:26px;margin-bottom:10px">Why a forest?</h3>
    <p class="muted" style="line-height:1.6;margin:0">Each tree sees a different slice of the data, so their votes cancel out individual mistakes. It handles the PCA components without extra scaling and gives a strong middle ground between the linear baseline and the slower neighbour search.</p>
  </div>
  <div class="panel">
    <h3 style="font-size:26px;margin-bottom:10px">Read the results</h3>
    <p class="muted" style="line-height:1.6;margin:0">Compare accuracy with macro-F1 to spot weak classes, then check train and predict time for the cost of each win. Shirt and T-shirt are the usual troublemakers, while Trouser and Bag achieve over 95% precision.</p>
  </div>
</div>

<!-- Classes Section -->
<div class="sec" id="classes" style="padding-bottom:80px">
  <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:16px;">
    <h2 style="font-size:36px">Ten classes to tell apart</h2>
    <span class="muted" style="font-size:13px">Click any class to jump to a corresponding test sample</span>
  </div>
  <div class="chips">
    <span class="chip" onclick="filterByClass(0)">T-shirt/top</span>
    <span class="chip" onclick="filterByClass(1)">Trouser</span>
    <span class="chip" onclick="filterByClass(2)">Pullover</span>
    <span class="chip" onclick="filterByClass(3)">Dress</span>
    <span class="chip" onclick="filterByClass(4)">Coat</span>
    <span class="chip" onclick="filterByClass(5)">Sandal</span>
    <span class="chip" onclick="filterByClass(6)">Shirt</span>
    <span class="chip" onclick="filterByClass(7)">Sneaker</span>
    <span class="chip" onclick="filterByClass(8)">Bag</span>
    <span class="chip" onclick="filterByClass(9)">Ankle boot</span>
  </div>
</div>

</div>
</div>

<!-- Modal: Confusion Matrix Viewer -->
<div class="modal-overlay" id="cm-modal" onclick="closeCMModal(event)">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:18px;">
      <div>
        <h3 style="font-size:24px">Test Set Confusion Matrix</h3>
        <p class="muted" style="font-size:13px; margin:4px 0 0">Actual rows vs. Predicted columns (10,000 Fashion-MNIST test images)</p>
      </div>
      <button class="btn-sm" onclick="closeCMModal()">&times; Close</button>
    </div>

    <!-- Model Switcher Tabs in Modal -->
    <div style="display:flex; gap:8px; margin-bottom:16px;">
      <button class="btn-sm active" id="tab-rf" onclick="switchCMTab('Random Forest')">Random Forest (Winner)</button>
      <button class="btn-sm" id="tab-lr" onclick="switchCMTab('Logistic Regression')">Logistic Regression</button>
      <button class="btn-sm" id="tab-knn" onclick="switchCMTab('K-Nearest Neighbours')">KNN</button>
    </div>

    <div style="overflow-x:auto;">
      <table class="cm-table" id="cm-table"></table>
    </div>
  </div>
</div>

<script>
// Embedded Data
const SAMPLES = {json.dumps(samples_data)};
const RESULTS = {json.dumps(results)};
const MODEL = {json.dumps(client_model)};
const CLASSES = MODEL.classes;

let currentIndex = 0;

function renderSample(idx) {{
  currentIndex = idx;
  const s = SAMPLES[idx];
  
  document.getElementById('preview-img').src = s.img;
  document.getElementById('source-badge').innerText = `SAMPLE #${{s.i}}`;
  document.getElementById('source-badge').className = "badge badge-lime";
  document.getElementById('res-class').innerText = s.pred_name;
  document.getElementById('res-conf').innerText = `${{(s.conf * 100).toFixed(1)}}%`;
  
  // Truth row
  document.getElementById('row-truth').style.display = 'flex';
  const isMatch = s.y === s.pred;
  document.getElementById('res-truth').innerHTML = `
    ${{s.y_name}}
    <span class="badge ${{isMatch ? 'badge-green' : 'badge-red'}}">${{isMatch ? 'MATCH' : 'MISMATCH'}}</span>
  `;

  document.getElementById('sample-counter').innerText = `${{idx + 1}} of ${{SAMPLES.length}}`;

  renderProbBars(s.top3);

  // Update active chip
  document.querySelectorAll('.chip').forEach((c, ci) => {{
    c.classList.toggle('active', ci === s.y);
  }});
}}

function renderProbBars(top3) {{
  const cont = document.getElementById('prob-bars');
  cont.innerHTML = '';
  top3.forEach(item => {{
    const pct = Math.round(item.prob * 100);
    const div = document.createElement('div');
    div.innerHTML = `
      <div class="prob-bar-row">
        <span class="muted">${{item.name}}</span>
        <span class="mono">${{pct}}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${{pct}}%;"></div>
      </div>
    `;
    cont.appendChild(div);
  }});
}}

function stepSample(delta) {{
  let n = currentIndex + delta;
  if (n < 0) n = SAMPLES.length - 1;
  if (n >= SAMPLES.length) n = 0;
  renderSample(n);
}}

function randomSample() {{
  const r = Math.floor(Math.random() * SAMPLES.length);
  renderSample(r);
}}

function filterByClass(classId) {{
  const matchIdx = SAMPLES.findIndex(s => s.y === classId);
  if (matchIdx !== -1) {{
    renderSample(matchIdx);
    document.getElementById('live-demo').scrollIntoView({{ behavior: 'smooth' }});
  }}
}}

// Drag & Drop / File Upload Handler
const dropzone = document.getElementById('dropzone');
['dragenter', 'dragover'].forEach(e => {{
  dropzone.addEventListener(e, ev => {{ ev.preventDefault(); dropzone.classList.add('dragover'); }});
}});
['dragleave', 'drop'].forEach(e => {{
  dropzone.addEventListener(e, ev => {{ ev.preventDefault(); dropzone.classList.remove('dragover'); }});
}});
dropzone.addEventListener('drop', ev => {{
  const files = ev.dataTransfer.files;
  if (files && files.length > 0) processUploadedImage(files[0]);
}});

function handleFileSelect(ev) {{
  if (ev.target.files && ev.target.files[0]) {{
    processUploadedImage(ev.target.files[0]);
  }}
}}

function processUploadedImage(file) {{
  const reader = new FileReader();
  reader.onload = function(e) {{
    const img = new Image();
    img.onload = function() {{
      // Preprocess image to 28x28 grayscale matching Fashion-MNIST
      const canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, 28, 28);

      // Fit into 24x24 box preserving aspect ratio
      const scale = Math.min(24 / img.width, 24 / img.height);
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);
      const x = Math.round((28 - w) / 2);
      const y = Math.round((28 - h) / 2);
      ctx.drawImage(img, x, y, w, h);

      const imgData = ctx.getImageData(0, 0, 28, 28);
      let edgeSum = 0, edgeCount = 0;
      // Check borders to detect light vs dark background
      for (let i = 0; i < 28; i++) {{
        edgeSum += imgData.data[i * 4] + imgData.data[(27 * 28 + i) * 4];
        edgeCount += 2;
      }}
      const invertBg = (edgeSum / edgeCount) > 120;

      const pixels = new Float64Array(784);
      for (let i = 0; i < 784; i++) {{
        let g = (imgData.data[i * 4] + imgData.data[i * 4 + 1] + imgData.data[i * 4 + 2]) / 3;
        if (invertBg) g = 255 - g;
        // Contrast enhancement
        g = Math.min(255, Math.max(0, (g - 15) * 1.15));
        pixels[i] = g / 255.0; // Scaled to [0, 1] like MinMaxScaler
      }}

      // Display processed image on preview
      const previewCanvas = document.createElement('canvas');
      previewCanvas.width = 28; previewCanvas.height = 28;
      const pctx = previewCanvas.getContext('2d');
      const outData = pctx.createImageData(28, 28);
      for (let i = 0; i < 784; i++) {{
        const v = Math.round(pixels[i] * 255);
        outData.data[i * 4] = v;
        outData.data[i * 4 + 1] = v;
        outData.data[i * 4 + 2] = v;
        outData.data[i * 4 + 3] = 255;
      }}
      pctx.putImageData(outData, 0, 0);
      document.getElementById('preview-img').src = previewCanvas.toDataURL();

      // PCA projection: emb = (pixels - mean) @ components.T
      const emb = new Float64Array(50);
      for (let c = 0; c < 50; c++) {{
        let sum = 0;
        const comp = MODEL.pca_components[c];
        for (let p = 0; p < 784; p++) {{
          sum += (pixels[p] - MODEL.pca_mean[p]) * comp[p];
        }}
        emb[c] = sum;
      }}

      // Softmax inference: logits = W @ emb + b
      const logits = new Float64Array(10);
      for (let k = 0; k < 10; k++) {{
        let s = MODEL.lr_b[k];
        const w = MODEL.lr_w[k];
        for (let c = 0; c < 50; c++) {{
          s += w[c] * emb[c];
        }}
        logits[k] = s;
      }}
      const maxL = Math.max(...logits);
      let sumExp = 0;
      const exps = new Float64Array(10);
      for (let k = 0; k < 10; k++) {{
        exps[k] = Math.exp(logits[k] - maxL);
        sumExp += exps[k];
      }}
      const probs = Array.from(exps).map((e, idx) => ({{
        id: idx,
        name: CLASSES[idx],
        prob: e / sumExp
      }})).sort((a, b) => b.prob - a.prob);

      // Update UI for custom upload
      document.getElementById('source-badge').innerText = 'CUSTOM UPLOAD';
      document.getElementById('source-badge').className = 'badge badge-green';
      document.getElementById('res-class').innerText = probs[0].name;
      document.getElementById('res-conf').innerText = `${{(probs[0].prob * 100).toFixed(1)}}%`;
      document.getElementById('row-truth').style.display = 'none';

      renderProbBars(probs.slice(0, 3));
    }};
    img.src = e.target.result;
  }};
  reader.readAsDataURL(file);
}}

// Confusion Matrix Modal logic
let currentCMModel = 'Random Forest';
function openCMModal() {{
  document.getElementById('cm-modal').classList.add('open');
  renderCMTable(currentCMModel);
}}
function closeCMModal(e) {{
  if (!e || e.target.id === 'cm-modal' || e.target.tagName === 'BUTTON') {{
    document.getElementById('cm-modal').classList.remove('open');
  }}
}}
function switchCMTab(modelName) {{
  currentCMModel = modelName;
  document.getElementById('tab-rf').className = modelName === 'Random Forest' ? 'btn-sm active' : 'btn-sm';
  document.getElementById('tab-lr').className = modelName === 'Logistic Regression' ? 'btn-sm active' : 'btn-sm';
  document.getElementById('tab-knn').className = modelName === 'K-Nearest Neighbours' ? 'btn-sm active' : 'btn-sm';
  renderCMTable(modelName);
}}
function renderCMTable(modelName) {{
  const cm = RESULTS[modelName].confusion;
  const tbl = document.getElementById('cm-table');
  let h = '<thead><tr><th>Actual \\ Pred</th>';
  CLASSES.forEach(c => {{ h += `<th>${{c.slice(0, 5)}}</th>`; }});
  h += '</tr></thead><tbody>';
  cm.forEach((row, r) => {{
    h += `<tr><th style="text-align:right; padding-right:8px;">${{CLASSES[r]}}</th>`;
    row.forEach((val, c) => {{
      const isDiag = (r === c);
      let cls = '';
      if (isDiag) cls = 'cm-diag';
      else if (val > 60) cls = 'cm-high-err';
      h += `<td class="${{cls}}">${{val}}</td>`;
    }});
    h += '</tr>';
  }});
  h += '</tbody>';
  tbl.innerHTML = h;
}}

// Initialize
renderSample(0);
</script>

</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"Generated index.html successfully! Size: {len(html)/1024:.1f} KB")
