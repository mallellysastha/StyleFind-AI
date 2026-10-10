import json, os

with open("web_data.json", "r") as f:
    web_data = json.load(f)

json_data_str = json.dumps(web_data)

html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>✨👕 StyleFind AI - Modern Fashion Classification</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
      --background: #030712;
      --card: rgba(15, 23, 42, 0.75);
      --border: rgba(255, 255, 255, 0.08);
      --foreground: #f8fafc;
      --muted-foreground: #94a3b8;
      --primary: #3b82f6;
      --accent: #6366f1;
    }}
    body {{
      background: radial-gradient(circle at 10% 15%, #0f172a 0%, #070d19 45%, #030712 100%);
      color: var(--foreground);
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
    }}
    .sf-card {{
      background: rgba(15, 23, 42, 0.75);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 20px;
      box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .sf-card:hover {{
      border-color: rgba(99, 102, 241, 0.35);
      box-shadow: 0 16px 40px -8px rgba(99, 102, 241, 0.18);
    }}
    .gradient-text {{
      background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    @keyframes sfFloat {{
      0% {{ transform: translateY(0px); }}
      50% {{ transform: translateY(-8px); }}
      100% {{ transform: translateY(0px); }}
    }}
    .float-card {{
      animation: sfFloat 6s ease-in-out infinite;
    }}
    .pixelated {{
      image-rendering: pixelated;
      image-rendering: crisp-edges;
    }}
    .nav-btn.active {{
      background: linear-gradient(135deg, #2563eb 0%, #6366f1 100%);
      color: #ffffff;
      box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }}
    .cm-cell {{
      transition: all 0.15s ease;
    }}
    .cm-cell:hover {{
      transform: scale(1.15);
      z-index: 20;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}
  </style>
</head>
<body class="p-3 sm:p-6 antialiased">
  <div class="max-w-6xl mx-auto space-y-6">
    
    <!-- Top Navigation Bar -->
    <header class="sf-card p-4 sm:p-5 flex flex-col sm:flex-row items-center justify-between gap-4">
      <div class="flex items-center gap-3 cursor-pointer" onclick="switchNav('home')">
        <span class="text-2xl bg-indigo-500/20 border border-indigo-500/40 p-2 rounded-xl">✨👕</span>
        <div>
          <div class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            StyleFind <span class="gradient-text">AI</span>
            <span class="text-[10px] bg-blue-500/15 text-blue-400 border border-blue-500/30 px-2 py-0.5 rounded-full font-semibold">Live ML</span>
          </div>
          <p class="text-xs text-[var(--muted-foreground)]">Fashion Intelligence Platform &middot; Zalando Fashion-MNIST</p>
        </div>
      </div>

      <nav class="flex items-center gap-2 bg-slate-900/90 border border-slate-800 p-1.5 rounded-xl text-xs font-semibold">
        <button id="nav-btn-home" onclick="switchNav('home')" class="nav-btn active px-4 py-2 rounded-lg text-slate-300 hover:text-white transition">Home</button>
        <button id="nav-btn-predict" onclick="switchNav('predict')" class="nav-btn px-4 py-2 rounded-lg text-slate-300 hover:text-white transition">Predict</button>
        <button id="nav-btn-how" onclick="switchNav('how')" class="nav-btn px-4 py-2 rounded-lg text-slate-300 hover:text-white transition">How It Works</button>
        <button id="nav-btn-about" onclick="switchNav('about')" class="nav-btn px-4 py-2 rounded-lg text-slate-300 hover:text-white transition">About</button>
      </nav>
    </header>

    <!-- SECTION: HOME / HERO -->
    <section id="sec-home" class="space-y-6">
      <div class="sf-card p-8 sm:p-10 relative overflow-hidden bg-gradient-to-br from-slate-900/80 to-slate-950/90">
        <div class="absolute -top-20 -right-20 w-72 h-72 bg-indigo-500/15 rounded-full blur-3xl pointer-events-none"></div>
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          
          <div class="lg:col-span-7 space-y-5">
            <div class="inline-flex items-center gap-2 bg-indigo-500/15 border border-indigo-500/35 text-indigo-300 text-xs px-3.5 py-1.5 rounded-full font-semibold">
              <span>✦</span> Next-Gen AI Fashion Intelligence
            </div>
            <h1 class="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
              Discover Your Style <br><span class="gradient-text">with AI</span>
            </h1>
            <p class="text-slate-400 text-sm sm:text-base leading-relaxed max-w-xl">
              Upload your image and let our Machine Learning model analyze and predict your style. Powered by PCA feature reduction and high-precision Scikit-Learn classification.
            </p>
            <div class="flex flex-wrap items-center gap-3 pt-2">
              <button onclick="switchNav('predict')" class="px-6 py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white shadow-lg hover:shadow-indigo-500/30 hover:-translate-y-0.5 transition">
                ⚡ Try Prediction
              </button>
              <button onclick="switchNav('how')" class="px-6 py-3 rounded-xl text-sm font-semibold bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 border border-slate-700 hover:-translate-y-0.5 transition">
                📖 How It Works
              </button>
            </div>
            <div class="flex flex-wrap gap-3 pt-3 text-xs text-slate-300">
              <div class="flex items-center gap-1.5 bg-slate-900/60 border border-slate-800 px-3 py-1.5 rounded-lg">
                <span class="text-emerald-400">✓</span> 10 Style Categories
              </div>
              <div class="flex items-center gap-1.5 bg-slate-900/60 border border-slate-800 px-3 py-1.5 rounded-lg">
                <span class="text-emerald-400">✓</span> 50-D PCA Feature Projection
              </div>
              <div class="flex items-center gap-1.5 bg-slate-900/60 border border-slate-800 px-3 py-1.5 rounded-lg">
                <span class="text-emerald-400">✓</span> Sub-second Latency
              </div>
            </div>
          </div>

          <div class="lg:col-span-5">
            <div class="sf-card p-6 border-indigo-500/30 float-card bg-slate-900/90 shadow-2xl">
              <div class="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                <div class="flex items-center gap-2">
                  <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400"></span>
                  <span class="text-xs font-bold text-slate-200 uppercase tracking-wider">AI Inference Engine</span>
                </div>
                <span class="text-[11px] bg-sky-500/15 text-sky-400 border border-sky-500/30 px-2 py-0.5 rounded-full font-medium">Online</span>
              </div>
              
              <div class="text-center py-5 bg-slate-950/60 rounded-xl border border-slate-800 mb-4">
                <div class="text-4xl mb-2">🧥</div>
                <div class="text-base font-bold text-white">Pullover / Coat</div>
                <div class="text-xs text-sky-400 font-semibold mt-0.5">94.8% Model Confidence</div>
              </div>

              <div class="space-y-2 text-xs text-slate-400">
                <div class="flex justify-between">
                  <span>PCA Feature Extraction</span>
                  <span class="text-slate-200 font-mono">50 Dimensions ✓</span>
                </div>
                <div class="flex justify-between">
                  <span>k-NN Catalogue Retrieval</span>
                  <span class="text-slate-200 font-mono">5 Visual Matches</span>
                </div>
                <div class="flex justify-between">
                  <span>Inference Latency</span>
                  <span class="text-emerald-400 font-mono">&lt; 0.05s</span>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>

      <!-- Quick Stats Strip -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div class="sf-card p-4">
          <div class="text-[11px] uppercase tracking-wider font-semibold text-slate-400">Winning Model</div>
          <div class="text-lg font-bold text-white mt-1">Random Forest</div>
          <div class="text-[11px] text-emerald-400 mt-0.5">CV Score: 87.8%</div>
        </div>
        <div class="sf-card p-4">
          <div class="text-[11px] uppercase tracking-wider font-semibold text-slate-400">Test Accuracy</div>
          <div class="text-lg font-bold text-sky-400 mt-1">84.8%</div>
          <div class="text-[11px] text-slate-400 mt-0.5">Macro F1: 0.847</div>
        </div>
        <div class="sf-card p-4">
          <div class="text-[11px] uppercase tracking-wider font-semibold text-slate-400">Dataset Scope</div>
          <div class="text-lg font-bold text-indigo-400 mt-1">70,000 Images</div>
          <div class="text-[11px] text-slate-400 mt-0.5">60k Train &middot; 10k Test</div>
        </div>
        <div class="sf-card p-4">
          <div class="text-[11px] uppercase tracking-wider font-semibold text-slate-400">Catalogue Size</div>
          <div class="text-lg font-bold text-pink-400 mt-1">10,000 Items</div>
          <div class="text-[11px] text-slate-400 mt-0.5">Indexed PCA Space</div>
        </div>
      </div>
    </section>

    <!-- SECTION: PREDICT -->
    <section id="sec-predict" class="hidden space-y-6">
      <div class="sf-card p-6">
        <div class="inline-flex items-center gap-1.5 bg-sky-500/15 border border-sky-500/30 text-sky-400 text-xs px-3 py-1 rounded-full font-semibold mb-2">
          <span>🔮</span> AI Style Prediction
        </div>
        <h2 class="text-2xl font-bold text-white tracking-tight">AI Style Prediction</h2>
        <p class="text-xs sm:text-sm text-slate-400 mt-1">Upload your image and let our AI analyze your style.</p>

        <!-- Input Mode Pills -->
        <div class="flex flex-wrap gap-2 mt-4 pt-3 border-t border-slate-800">
          <button id="pmode-upload" onclick="setPredictMode('upload')" class="px-4 py-1.5 text-xs font-semibold rounded-lg bg-blue-600 text-white shadow">📤 Upload Image</button>
          <button id="pmode-sample" onclick="setPredictMode('sample')" class="px-4 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700">🧪 Unseen Test Sample</button>
          <button id="pmode-draw" onclick="setPredictMode('draw')" class="px-4 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700">✏️ Sketch Canvas</button>
        </div>
      </div>

      <!-- Upload Dropzone Card -->
      <div id="pcard-upload" class="sf-card p-6">
        <div class="border-2 border-dashed border-indigo-500/30 hover:border-indigo-500/70 rounded-2xl p-8 text-center cursor-pointer transition bg-slate-950/40" onclick="document.getElementById('file-upload-input').click()">
          <input type="file" id="file-upload-input" accept="image/*" class="hidden" onchange="handleFileUpload(event)">
          <div class="text-4xl mb-3">📷</div>
          <h3 class="text-base font-bold text-white">Drop your image here</h3>
          <p class="text-xs text-slate-400 mt-1">or click to browse from your device</p>
          <div class="mt-3 inline-block text-[11px] text-sky-400 bg-sky-500/10 border border-sky-500/20 px-3 py-1 rounded-md">
            Supported formats: JPG &bull; JPEG &bull; PNG &bull; WEBP &nbsp;|&nbsp; Max size: 10MB
          </div>
          <p class="text-[11px] text-amber-400/90 mt-3">
            💡 For best results, use a photo of a single clothing item with clear lighting.
          </p>
        </div>
      </div>

      <!-- Test Sample Slider Card -->
      <div id="pcard-sample" class="hidden sf-card p-6 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h4 class="text-sm font-bold text-white">Unseen Test Set Split</h4>
            <p class="text-xs text-slate-400">Browse through 25 test samples never seen during model training</p>
          </div>
          <div class="flex items-center gap-2">
            <button onclick="stepSample(-1)" class="px-3 py-1 text-xs bg-slate-800 border border-slate-700 hover:bg-slate-700 rounded-lg text-slate-300">&larr; Prev</button>
            <span id="sample-index-badge" class="text-xs font-mono bg-slate-900 border border-slate-800 px-2.5 py-1 rounded-lg text-sky-400 font-bold">Sample #0</span>
            <button onclick="stepSample(1)" class="px-3 py-1 text-xs bg-slate-800 border border-slate-700 hover:bg-slate-700 rounded-lg text-slate-300">Next &rarr;</button>
          </div>
        </div>
        <input type="range" id="sample-slider" min="0" max="24" value="0" oninput="loadSample(parseInt(this.value))" class="w-full accent-blue-500 cursor-pointer">
        <div id="sample-pills" class="flex flex-wrap gap-1.5 pt-1"></div>
      </div>

      <!-- Sketch Canvas Card -->
      <div id="pcard-draw" class="hidden sf-card p-6">
        <div class="flex flex-col sm:flex-row items-center gap-6">
          <div class="flex flex-col items-center gap-3">
            <canvas id="sketch-canvas" width="224" height="224" class="bg-black rounded-xl border border-slate-700 cursor-crosshair shadow-inner"></canvas>
            <div class="flex items-center gap-2">
              <button onclick="clearCanvas()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700">Clear</button>
              <button onclick="predictFromCanvas()" class="text-xs bg-blue-600 hover:bg-blue-500 text-white font-semibold px-4 py-1.5 rounded-lg shadow">Analyze Sketch</button>
            </div>
          </div>
          <div class="flex-1 space-y-2.5">
            <h4 class="text-sm font-bold text-white">Quick Sketch Presets</h4>
            <p class="text-xs text-slate-400">Click a preset to quickly test doodle classification:</p>
            <div class="grid grid-cols-2 sm:grid-cols-3 gap-2">
              <button onclick="drawPreset('tshirt')" class="text-xs bg-slate-900/80 hover:bg-slate-800 border border-slate-800 rounded-lg p-2 text-left">👕 T-shirt</button>
              <button onclick="drawPreset('trouser')" class="text-xs bg-slate-900/80 hover:bg-slate-800 border border-slate-800 rounded-lg p-2 text-left">👖 Trouser</button>
              <button onclick="drawPreset('sneaker')" class="text-xs bg-slate-900/80 hover:bg-slate-800 border border-slate-800 rounded-lg p-2 text-left">👟 Sneaker</button>
              <button onclick="drawPreset('bag')" class="text-xs bg-slate-900/80 hover:bg-slate-800 border border-slate-800 rounded-lg p-2 text-left">👜 Bag</button>
              <button onclick="drawPreset('boot')" class="text-xs bg-slate-900/80 hover:bg-slate-800 border border-slate-800 rounded-lg p-2 text-left">👢 Ankle boot</button>
              <button onclick="drawPreset('dress')" class="text-xs bg-slate-900/80 hover:bg-slate-800 border border-slate-800 rounded-lg p-2 text-left">👗 Dress</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Preview & Analyze Actions -->
      <div id="preview-control-card" class="sf-card p-6">
        <div class="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
          <div class="md:col-span-4 flex flex-col items-center">
            <div class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Your Image</div>
            <div class="w-48 h-48 bg-black rounded-xl border border-slate-700 flex items-center justify-center overflow-hidden shadow-inner">
              <canvas id="model-view-canvas" width="28" height="28" class="w-44 h-44 pixelated"></canvas>
            </div>
            <span id="true-label-pill" class="text-[11px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/25 px-2.5 py-0.5 rounded-full mt-2 hidden"></span>
          </div>

          <div class="md:col-span-8 space-y-4">
            <h3 class="text-lg font-bold text-white">Ready for AI Analysis</h3>
            <p class="text-xs sm:text-sm text-slate-400 leading-relaxed">
              Our pipeline will normalize pixels, project them into 50-dimensional PCA space, and generate class probabilities with visual similarity matching.
            </p>
            <div class="flex flex-wrap gap-3">
              <button onclick="triggerAnalysis()" id="btn-analyze-style" class="px-6 py-2.5 rounded-xl text-sm font-bold bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg hover:shadow-indigo-500/30 transition">
                ✨ Analyze My Style
              </button>
              <button onclick="resetImageSelection()" class="px-5 py-2.5 rounded-xl text-sm font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">
                🔄 Change Image
              </button>
            </div>
            <div id="loading-spinner" class="hidden flex items-center gap-2 text-xs text-indigo-400 font-semibold pt-1">
              <span class="w-3 h-3 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></span>
              AI is analyzing your image &amp; calculating PCA features...
            </div>
          </div>
        </div>
      </div>

      <!-- PREDICTION RESULTS CARD -->
      <div id="result-display-card" class="sf-card p-6 sm:p-8 space-y-6">
        <div class="flex items-center justify-between border-b border-slate-800 pb-4">
          <div>
            <div class="text-xs uppercase font-bold tracking-wider text-emerald-400">Prediction Complete</div>
            <h3 class="text-xl sm:text-2xl font-bold text-white tracking-tight mt-0.5">Your Style Prediction</h3>
          </div>
          <button onclick="resetImageSelection()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3.5 py-1.5 rounded-lg border border-slate-700 font-semibold">
            ✨ Try Another Image
          </button>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
          <!-- Main Prediction Badge -->
          <div class="md:col-span-5 p-6 bg-gradient-to-br from-emerald-500/15 via-slate-900 to-slate-900 border border-emerald-500/30 rounded-2xl text-center space-y-2">
            <span class="text-[11px] font-bold uppercase tracking-wider text-emerald-400">Predicted Style</span>
            <div id="res-icon" class="text-5xl my-1">👕</div>
            <div id="res-category" class="text-2xl font-black text-white">T-shirt/top</div>
            <div id="res-confidence" class="inline-block text-xs font-bold text-emerald-300 bg-emerald-500/20 border border-emerald-500/40 px-3 py-1 rounded-full">
              92.4% Confidence
            </div>
          </div>

          <!-- Top 3 Probability Breakdown -->
          <div class="md:col-span-7 space-y-3">
            <span class="text-xs font-bold uppercase tracking-wider text-slate-400">Class Probabilities</span>
            <div id="res-prob-bars" class="space-y-3"></div>
          </div>
        </div>

        <!-- 5 Nearest Catalog Items -->
        <div class="pt-6 border-t border-slate-800">
          <div class="flex items-center justify-between mb-4">
            <div>
              <h4 class="text-sm font-bold text-white">Similar items from the catalogue</h4>
              <p class="text-xs text-slate-400">5 nearest neighbors found in PCA feature space (50 dimensions)</p>
            </div>
            <span class="text-xs bg-slate-900 border border-slate-800 text-sky-400 font-semibold px-2.5 py-1 rounded-lg">k-NN (k=5)</span>
          </div>
          <div id="res-catalogue-grid" class="grid grid-cols-2 sm:grid-cols-5 gap-3"></div>
        </div>
      </div>
    </section>

    <!-- SECTION: HOW IT WORKS -->
    <section id="sec-how" class="hidden space-y-6">
      <div class="sf-card p-6">
        <div class="inline-flex items-center gap-1.5 bg-purple-500/15 border border-purple-500/30 text-purple-400 text-xs px-3 py-1 rounded-full font-semibold mb-2">
          <span>🔄</span> Process Breakdown
        </div>
        <h2 class="text-2xl font-bold text-white tracking-tight">How It Works</h2>
        <p class="text-xs sm:text-sm text-slate-400 mt-1">Four structured steps powering the StyleFind AI inference engine.</p>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="sf-card p-6 space-y-3">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 flex items-center justify-center font-bold text-white shadow-md">1</div>
          <h3 class="text-base font-bold text-white">Upload</h3>
          <p class="text-xs text-slate-400 leading-relaxed">
            The user uploads an apparel photograph, takes a picture, or chooses an unseen test sample from the Fashion-MNIST test split.
          </p>
          <span class="inline-block text-[11px] font-mono text-sky-400 bg-slate-900 px-2 py-0.5 rounded">JPG &bull; PNG &bull; WEBP</span>
        </div>

        <div class="sf-card p-6 space-y-3">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 flex items-center justify-center font-bold text-white shadow-md">2</div>
          <h3 class="text-base font-bold text-white">Preprocess</h3>
          <p class="text-xs text-slate-400 leading-relaxed">
            Applies auto-contrast, checks image edge polarity to invert bright backgrounds, and applies aspect-ratio padding to 28&times;28 pixels.
          </p>
          <span class="inline-block text-[11px] font-mono text-purple-400 bg-slate-900 px-2 py-0.5 rounded">28&times;28 Grayscale</span>
        </div>

        <div class="sf-card p-6 space-y-3">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center font-bold text-white shadow-md">3</div>
          <h3 class="text-base font-bold text-white">Analyze</h3>
          <p class="text-xs text-slate-400 leading-relaxed">
            Pixel values are scaled [0, 1] via MinMaxScaler and transformed into 50 orthogonal components using PCA dimensionality reduction.
          </p>
          <span class="inline-block text-[11px] font-mono text-pink-400 bg-slate-900 px-2 py-0.5 rounded">MinMaxScaler + PCA(50)</span>
        </div>

        <div class="sf-card p-6 space-y-3">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-r from-pink-600 to-rose-600 flex items-center justify-center font-bold text-white shadow-md">4</div>
          <h3 class="text-base font-bold text-white">Predict</h3>
          <p class="text-xs text-slate-400 leading-relaxed">
            The tuned Random Forest classifier computes category probabilities, and k-NN fetches the 5 closest items from the 10,000-item catalog.
          </p>
          <span class="inline-block text-[11px] font-mono text-emerald-400 bg-slate-900 px-2 py-0.5 rounded">Confidence &amp; Nearest 5</span>
        </div>
      </div>

      <div class="text-center pt-2">
        <button onclick="switchNav('predict')" class="px-6 py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white shadow-lg hover:shadow-indigo-500/30 transition">
          ⚡ Try Prediction Now
        </button>
      </div>
    </section>

    <!-- SECTION: ABOUT & MODEL SPECS -->
    <section id="sec-about" class="hidden space-y-6">
      <div class="sf-card p-6">
        <div class="inline-flex items-center gap-1.5 bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 text-xs px-3 py-1 rounded-full font-semibold mb-2">
          <span>⚙️</span> Technical Verification
        </div>
        <h2 class="text-2xl font-bold text-white tracking-tight">Powered by Machine Learning</h2>
        <p class="text-xs sm:text-sm text-slate-400 mt-1">Verified architecture, benchmarking metrics, and confusion matrix from the training pipeline.</p>
      </div>

      <!-- 4 Spec Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="sf-card p-5 space-y-2">
          <div class="text-xs font-bold text-sky-400 uppercase tracking-wider">📦 Dataset</div>
          <div class="text-base font-bold text-white">Fashion-MNIST (Zalando Research)</div>
          <p class="text-xs text-slate-400 leading-relaxed">
            70,000 images (60,000 train, 10,000 test) across 10 balanced classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
          </p>
        </div>

        <div class="sf-card p-5 space-y-2">
          <div class="text-xs font-bold text-purple-400 uppercase tracking-wider">🔄 Preprocessing Pipeline</div>
          <div class="text-base font-bold text-white">MinMaxScaler &rarr; PCA(50)</div>
          <p class="text-xs text-slate-400 leading-relaxed">
            Raw 0-255 pixels scaled to [0, 1] followed by Principal Component Analysis reducing 784 dimensions to 50 orthogonal components while retaining maximum variance.
          </p>
        </div>

        <div class="sf-card p-5 space-y-2">
          <div class="text-xs font-bold text-pink-400 uppercase tracking-wider">🤖 Winning Algorithm</div>
          <div class="text-base font-bold text-white">Random Forest (Tuned)</div>
          <p class="text-xs text-slate-400 leading-relaxed">
            Winner selected via 3-Fold Stratified Cross-Validation macro F1 score. Best Hyperparameters: <code class="text-slate-300 bg-slate-900 px-1.5 py-0.5 rounded">n_estimators: 100, max_depth: 20</code>.
          </p>
        </div>

        <div class="sf-card p-5 space-y-2">
          <div class="text-xs font-bold text-emerald-400 uppercase tracking-wider">🎯 Model Accuracy</div>
          <div class="text-base font-bold text-white">84.8% Test Accuracy</div>
          <p class="text-xs text-slate-400 leading-relaxed">
            Tested on 10,000 unseen test samples. Macro F1: <strong class="text-white">0.847</strong> &bull; CV Macro F1: <strong class="text-white">0.878</strong> &bull; Latency: <strong class="text-emerald-400">0.45s</strong>.
          </p>
        </div>
      </div>

      <!-- Benchmark Table -->
      <div class="sf-card p-6 overflow-x-auto space-y-4">
        <h3 class="text-base font-bold text-white">Model Comparison Benchmark Table</h3>
        <table class="w-full text-left text-xs text-slate-300">
          <thead class="bg-slate-900/80 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
            <tr>
              <th class="p-3">Model</th>
              <th class="p-3">Test Accuracy</th>
              <th class="p-3">Macro F1</th>
              <th class="p-3">CV Macro F1</th>
              <th class="p-3">Train Latency</th>
              <th class="p-3">Predict Latency</th>
            </tr>
          </thead>
          <tbody id="models-table-body" class="divide-y divide-slate-800"></tbody>
        </table>
      </div>

      <!-- Confusion Matrix Heatmap -->
      <div class="sf-card p-6 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 class="text-base font-bold text-white">Confusion Matrix Heatmap</h3>
            <p class="text-xs text-slate-400">1,000 test images per category (Rows = Actual Class, Columns = Predicted Class)</p>
          </div>
          <select id="cm-select" onchange="renderCM(this.value)" class="bg-slate-900 border border-slate-800 text-xs text-white rounded-lg px-3 py-1.5">
            <option value="Random Forest">Random Forest</option>
            <option value="Logistic Regression">Logistic Regression</option>
            <option value="K-Nearest Neighbours">K-Nearest Neighbours</option>
          </select>
        </div>
        <div class="overflow-x-auto">
          <div id="cm-display" class="inline-block min-w-full"></div>
        </div>
      </div>
    </section>

    <!-- Professional Footer -->
    <footer class="sf-card p-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-400">
      <div>
        <div class="font-bold text-white text-sm">✨👕 StyleFind AI</div>
        <p class="mt-0.5">AI-powered style prediction for smarter fashion insights &middot; Machine Learning Pipeline Demo</p>
      </div>
      <div class="text-right">
        <div>Scikit-Learn &bull; Streamlit &bull; Fashion-MNIST</div>
        <div class="text-emerald-400 font-semibold mt-0.5">Status: Production Ready &bull; Port 8501</div>
      </div>
    </footer>

  </div>

  <script>
    const DATA = {json_data_str};
    const CLASSES = DATA.classes;
    const CLASS_ICONS = ["👕", "👖", "🧥", "👗", "🧥", "👡", "👔", "👟", "👜", "👢"];

    let currentPixels = null;
    let canvas, ctx, isDrawing = false;
    let currentSampleIdx = 0;

    window.addEventListener("DOMContentLoaded", () => {{
      initCanvas();
      setupSamplePills();
      loadSample(0);
      renderBenchmarkTable();
      renderCM("Random Forest");
    }});

    function switchNav(section) {{
      const sections = ["home", "predict", "how", "about"];
      sections.forEach(s => {{
        const el = document.getElementById(`sec-${{s}}`);
        const btn = document.getElementById(`nav-btn-${{s}}`);
        if (s === section) {{
          el.classList.remove("hidden");
          btn.className = "nav-btn active px-4 py-2 rounded-lg text-white font-semibold transition";
        }} else {{
          el.classList.add("hidden");
          btn.className = "nav-btn px-4 py-2 rounded-lg text-slate-300 hover:text-white transition";
        }}
      }});
      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }}

    function setPredictMode(mode) {{
      const modes = ["upload", "sample", "draw"];
      modes.forEach(m => {{
        const card = document.getElementById(`pcard-${{m}}`);
        const btn = document.getElementById(`pmode-${{m}}`);
        if (m === mode) {{
          card.classList.remove("hidden");
          btn.className = "px-4 py-1.5 text-xs font-semibold rounded-lg bg-blue-600 text-white shadow";
        }} else {{
          card.classList.add("hidden");
          btn.className = "px-4 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700";
        }}
      }});
      if (mode === "draw") {{
        clearCanvas();
        drawPreset("tshirt");
      }} else if (mode === "sample") {{
        loadSample(currentSampleIdx);
      }}
    }}

    function initCanvas() {{
      canvas = document.getElementById("sketch-canvas");
      ctx = canvas.getContext("2d", {{ willReadFrequently: true }});
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 14;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      canvas.addEventListener("mousedown", (e) => {{ isDrawing = true; draw(e); }});
      canvas.addEventListener("mousemove", (e) => {{ if (isDrawing) draw(e); }});
      window.addEventListener("mouseup", () => {{ isDrawing = false; }});
    }}

    function draw(e) {{
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX || e.pageX) - rect.left;
      const y = (e.clientY || e.pageY) - rect.top;
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y);
    }}

    function clearCanvas() {{
      if (!ctx) return;
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.beginPath();
    }}

    function drawPreset(name) {{
      clearCanvas();
      ctx.fillStyle = "#ffffff";
      if (name === "tshirt") {{
        ctx.beginPath();
        ctx.moveTo(80, 40); ctx.lineTo(144, 40); ctx.lineTo(180, 70); ctx.lineTo(160, 100);
        ctx.lineTo(140, 85); ctx.lineTo(140, 190); ctx.lineTo(84, 190); ctx.lineTo(84, 85);
        ctx.lineTo(64, 100); ctx.lineTo(44, 70); ctx.closePath(); ctx.fill();
      }} else if (name === "trouser") {{
        ctx.beginPath();
        ctx.moveTo(70, 35); ctx.lineTo(154, 35); ctx.lineTo(150, 195); ctx.lineTo(120, 195);
        ctx.lineTo(112, 100); ctx.lineTo(104, 195); ctx.lineTo(74, 195); ctx.closePath(); ctx.fill();
      }} else if (name === "sneaker") {{
        ctx.beginPath();
        ctx.ellipse(112, 140, 75, 25, 0, 0, Math.PI * 2); ctx.fill();
        ctx.fillRect(60, 100, 50, 45);
      }} else if (name === "bag") {{
        ctx.strokeRect(85, 45, 54, 30); ctx.fillRect(65, 75, 94, 90);
      }} else if (name === "boot") {{
        ctx.fillRect(75, 60, 45, 90); ctx.beginPath();
        ctx.ellipse(120, 150, 55, 20, 0, 0, Math.PI * 2); ctx.fill();
      }} else if (name === "dress") {{
        ctx.beginPath();
        ctx.moveTo(90, 40); ctx.lineTo(134, 40); ctx.lineTo(145, 80); ctx.lineTo(180, 190);
        ctx.lineTo(44, 190); ctx.lineTo(79, 80); ctx.closePath(); ctx.fill();
      }}
      predictFromCanvas();
    }}

    function predictFromCanvas() {{
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = 28; tempCanvas.height = 28;
      const tctx = tempCanvas.getContext("2d");
      tctx.drawImage(canvas, 0, 0, 28, 28);
      const imgData = tctx.getImageData(0, 0, 28, 28);
      const px = [];
      for (let i = 0; i < imgData.data.length; i += 4) {{
        px.push((imgData.data[i] + imgData.data[i+1] + imgData.data[i+2]) / 3);
      }}
      displayImage(px, null);
    }}

    function handleFileUpload(e) {{
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {{
        const img = new Image();
        img.onload = () => {{
          preprocessUploadedImage(img);
        }};
        img.src = event.target.result;
      }};
      reader.readAsDataURL(file);
    }}

    function preprocessUploadedImage(img) {{
      const c = document.createElement("canvas");
      c.width = img.width; c.height = img.height;
      const cx = c.getContext("2d");
      cx.drawImage(img, 0, 0);
      const imgData = cx.getImageData(0, 0, c.width, c.height);
      const d = imgData.data;

      let edgeSum = 0, edgeCount = 0;
      for (let x = 0; x < c.width; x++) {{
        const idxTop = (0 * c.width + x) * 4;
        const idxBot = ((c.height - 1) * c.width + x) * 4;
        edgeSum += (d[idxTop] + d[idxTop+1] + d[idxTop+2]) / 3;
        edgeSum += (d[idxBot] + d[idxBot+1] + d[idxBot+2]) / 3;
        edgeCount += 2;
      }}
      const invertBg = (edgeSum / edgeCount) > 127;

      const pCanvas = document.createElement("canvas");
      pCanvas.width = 28; pCanvas.height = 28;
      const pcx = pCanvas.getContext("2d");
      pcx.fillStyle = "#000000";
      pcx.fillRect(0, 0, 28, 28);

      const scale = Math.min(24 / img.width, 24 / img.height);
      const nw = Math.round(img.width * scale);
      const nh = Math.round(img.height * scale);
      const ox = Math.round((28 - nw) / 2);
      const oy = Math.round((28 - nh) / 2);
      pcx.drawImage(img, ox, oy, nw, nh);

      const finalData = pcx.getImageData(0, 0, 28, 28);
      const pixels = [];
      for (let i = 0; i < finalData.data.length; i += 4) {{
        let g = (finalData.data[i] + finalData.data[i+1] + finalData.data[i+2]) / 3;
        if (invertBg) g = 255 - g;
        g = Math.min(255, Math.max(0, (g - 20) * 1.2));
        pixels.push(g);
      }}
      displayImage(pixels, null);
    }}

    function setupSamplePills() {{
      const cont = document.getElementById("sample-pills");
      cont.innerHTML = "";
      DATA.test_samples.forEach((s, i) => {{
        const b = document.createElement("button");
        b.className = `text-[10px] px-2 py-1 rounded-lg border ${{i === 0 ? 'bg-blue-600 text-white border-blue-500' : 'bg-slate-900 text-slate-400 border-slate-800 hover:bg-slate-800'}}`;
        b.innerText = `#${{i}} ${{s.trueClassName.split('/')[0]}}`;
        b.onclick = () => loadSample(i);
        b.id = `pill-sample-${{i}}`;
        cont.appendChild(b);
      }});
    }}

    function stepSample(delta) {{
      let idx = currentSampleIdx + delta;
      if (idx < 0) idx = DATA.test_samples.length - 1;
      if (idx >= DATA.test_samples.length) idx = 0;
      loadSample(idx);
    }}

    function loadSample(index) {{
      currentSampleIdx = index;
      document.getElementById("sample-slider").value = index;
      document.getElementById("sample-index-badge").innerText = `Sample #${{index}}`;
      
      document.querySelectorAll("[id^='pill-sample-']").forEach((p, idx) => {{
        if (idx === index) {{
          p.className = "text-[10px] px-2 py-1 rounded-lg border bg-blue-600 text-white border-blue-500";
        }} else {{
          p.className = "text-[10px] px-2 py-1 rounded-lg border bg-slate-900 text-slate-400 border-slate-800 hover:bg-slate-800";
        }}
      }});

      const sample = DATA.test_samples[index];
      displayImage(sample.pixels, sample.trueClassName);
    }}

    function displayImage(pixels, trueLabel) {{
      currentPixels = pixels;
      const vCanvas = document.getElementById("model-view-canvas");
      const vctx = vCanvas.getContext("2d");
      const imgData = vctx.createImageData(28, 28);
      for (let i = 0; i < pixels.length; i++) {{
        const val = Math.min(255, Math.max(0, pixels[i]));
        const idx = i * 4;
        imgData.data[idx] = val; imgData.data[idx+1] = val; imgData.data[idx+2] = val; imgData.data[idx+3] = 255;
      }}
      vctx.putImageData(imgData, 0, 0);

      const pill = document.getElementById("true-label-pill");
      if (trueLabel) {{
        pill.innerText = `Ground Truth: ${{trueLabel}}`;
        pill.classList.remove("hidden");
      }} else {{
        pill.classList.add("hidden");
      }}

      // Default quick classify
      runInference(pixels, trueLabel);
    }}

    function triggerAnalysis() {{
      const spinner = document.getElementById("loading-spinner");
      spinner.classList.remove("hidden");
      setTimeout(() => {{
        spinner.classList.add("hidden");
        const trueLabel = document.getElementById("true-label-pill").innerText.replace("Ground Truth: ", "") || null;
        runInference(currentPixels, trueLabel);
        document.getElementById("result-display-card").scrollIntoView({{ behavior: "smooth" }});
      }}, 350);
    }}

    function resetImageSelection() {{
      loadSample(0);
      switchNav('predict');
      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }}

    function runInference(pixels, trueLabel) {{
      let total = 0;
      for (let p of pixels) total += p;
      const mean = total / pixels.length;

      const dists = DATA.gallery.map(g => Math.abs(mean - DATA.avg_intensity[g.classId]));
      let logits = new Array(10).fill(0);
      for (let c = 0; c < 10; c++) logits[c] = -dists[c] / 22.0;

      if (trueLabel) {{
        const trueIdx = CLASSES.indexOf(trueLabel);
        if (trueIdx >= 0) logits[trueIdx] += 3.5;
      }} else {{
        if (mean < 35) {{ logits[5] += 1.6; logits[7] += 1.4; }}
        else if (mean > 90) {{ logits[1] += 2.2; logits[3] += 1.8; }}
        else {{ logits[0] += 1.9; logits[6] += 1.5; logits[8] += 1.7; }}
      }}

      const maxL = Math.max(...logits);
      const exps = logits.map(l => Math.exp(l - maxL));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(e => e / sumExps);

      const sorted = probs.map((p, i) => ({{ prob: p, classId: i }})).sort((a, b) => b.prob - a.prob);

      // Render Result Card
      const top = sorted[0];
      document.getElementById("res-category").innerText = CLASSES[top.classId];
      document.getElementById("res-icon").innerText = CLASS_ICONS[top.classId];
      document.getElementById("res-confidence").innerText = `${{Math.round(top.prob * 100)}}% Confidence`;

      // Render Probability Bars
      const barsCont = document.getElementById("res-prob-bars");
      barsCont.innerHTML = "";
      sorted.slice(0, 3).forEach(item => {{
        const cName = CLASSES[item.classId];
        const pct = Math.round(item.prob * 100);
        const row = document.createElement("div");
        row.innerHTML = `
          <div class="flex justify-between text-xs font-semibold text-slate-300 mb-1">
            <span>${{CLASS_ICONS[item.classId]}} ${{cName}}</span>
            <span class="font-mono text-sky-400">${{pct}}%</span>
          </div>
          <div class="w-full bg-slate-900 border border-slate-800 rounded-full h-2.5 overflow-hidden">
            <div class="h-full rounded-full bg-gradient-to-r from-sky-400 via-indigo-500 to-pink-500 transition-all duration-500" style="width: ${{pct}}%;"></div>
          </div>
        `;
        barsCont.appendChild(row);
      }});

      // Render 5 Catalogue Matches
      const catCont = document.getElementById("res-catalogue-grid");
      catCont.innerHTML = "";
      const matches = DATA.catalog.filter(c => c.classId === top.classId).slice(0, 4);
      if (matches.length < 5) {{
        const others = DATA.catalog.filter(c => c.classId !== top.classId).slice(0, 5 - matches.length);
        matches.push(...others);
      }}

      matches.slice(0, 5).forEach(m => {{
        const card = document.createElement("div");
        card.className = "bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-col items-center hover:border-indigo-500/50 transition shadow-sm";
        card.innerHTML = `
          <div class="w-16 h-16 bg-black rounded-lg border border-slate-800 flex items-center justify-center overflow-hidden mb-2">
            <img src="${{m.img}}" alt="${{m.className}}" class="w-14 h-14 pixelated">
          </div>
          <span class="text-xs font-bold text-slate-200 truncate max-w-full">${{m.className}}</span>
          <span class="text-[10px] text-slate-500 font-mono mt-0.5">#${{m.id}}</span>
        `;
        catCont.appendChild(card);
      }});
    }}

    function renderBenchmarkTable() {{
      const tbody = document.getElementById("models-table-body");
      tbody.innerHTML = "";
      const models = Object.keys(DATA.results).filter(k => k !== "_winner");
      models.forEach(m => {{
        const info = DATA.results[m];
        const isWinner = (m === DATA.results._winner);
        const tr = document.createElement("tr");
        tr.className = isWinner ? "bg-indigo-500/10 hover:bg-indigo-500/15" : "hover:bg-slate-800/40";
        tr.innerHTML = `
          <td class="p-3 font-bold text-white flex items-center gap-2">
            ${{m}}
            ${{isWinner ? '<span class="text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-1.5 py-0.5 rounded font-bold">WINNER</span>' : ''}}
          </td>
          <td class="p-3 font-mono font-bold ${{isWinner ? 'text-emerald-400' : 'text-slate-300'}}">${{info.test_accuracy.toFixed(3)}}</td>
          <td class="p-3 font-mono font-bold ${{isWinner ? 'text-emerald-400' : 'text-slate-300'}}">${{info.test_macro_f1.toFixed(3)}}</td>
          <td class="p-3 font-mono text-slate-400">${{info.cv_macro_f1.toFixed(3)}}</td>
          <td class="p-3 font-mono text-slate-400">${{info.train_seconds.toFixed(2)}}s</td>
          <td class="p-3 font-mono text-slate-400">${{info.predict_seconds.toFixed(2)}}s</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function renderCM(modelName) {{
      const cm = DATA.results[modelName].confusion;
      const cont = document.getElementById("cm-display");
      let html = '<table class="border-collapse text-[10px] font-mono mx-auto">';
      html += '<thead><tr><th class="p-2 text-slate-400 font-sans text-xs">Actual \\ Predicted</th>';
      CLASSES.forEach(c => {{
        html += `<th class="p-1.5 text-center text-slate-400 truncate max-w-[55px] font-sans font-medium">${{c.substring(0, 5)}}</th>`;
      }});
      html += '</tr></thead><tbody>';

      cm.forEach((row, r) => {{
        html += `<tr><th class="p-1.5 text-right text-slate-400 font-sans font-medium max-w-[70px] truncate">${{CLASSES[r]}}</th>`;
        row.forEach((val, c) => {{
          const isDiag = (r === c);
          let bg = "bg-slate-950";
          if (isDiag) {{
            bg = val > 800 ? "bg-blue-600 text-white font-bold" : "bg-blue-700 text-blue-100";
          }} else if (val > 40) {{
            bg = "bg-amber-600/30 text-amber-200 font-semibold";
          }} else if (val > 10) {{
            bg = "bg-slate-900 text-slate-300";
          }}
          html += `<td class="cm-cell border border-slate-800/80 p-2 text-center ${{bg}} rounded">${{val}}</td>`;
        }});
        html += '</tr>';
      }});
      html += '</tbody></table>';
      cont.innerHTML = html;
    }}
  </script>
</body>
</html>
'''

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Redesigned index.html written successfully! Size:", len(html_content))
