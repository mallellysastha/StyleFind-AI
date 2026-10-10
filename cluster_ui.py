"""Clustering and algorithm-comparison views for StyleFind AI."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clustering_engine import (
    benchmark_all_clustering,
    compute_elbow_curve,
    extract_features_batch,
    run_dbscan_clustering,
    run_hierarchical_clustering,
    run_kmeans_clustering,
)

CLUSTER_COLORS = ["#818cf8", "#38bdf8", "#f472b6", "#34d399", "#fbbf24", "#22d3ee"]
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _fmt(value, kind="num"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if kind == "ms":
        return f"{value:.1f} ms"
    if kind == "int":
        return str(int(value))
    return f"{value:.3f}"


def _init_state():
    st.session_state.setdefault("cl_items", [])
    st.session_state.setdefault("cl_names", [])
    st.session_state.setdefault("cl_pack", None)
    st.session_state.setdefault("cl_result", None)
    st.session_state.setdefault("cl_elbow", None)
    st.session_state.setdefault("cl_compare", None)
    st.session_state.setdefault("cl_algo", "K-Means")
    st.session_state.setdefault("upload_sig", None)


def _set_dataset(images, names):
    st.session_state.cl_items = images
    st.session_state.cl_names = names
    st.session_state.cl_pack = None
    st.session_state.cl_result = None
    st.session_state.cl_elbow = None
    st.session_state.cl_compare = None


def _sample_catalogue(assets, n=30, seed=0):
    rng = np.random.default_rng(seed)
    n = min(int(n), len(assets["cat_img"]))
    idx = rng.choice(len(assets["cat_img"]), size=n, replace=False)
    images, names = [], []
    for i in idx:
        images.append(assets["cat_img"][i].reshape(28, 28).astype(np.float32))
        names.append(f"fnist_{int(i)}_{CLASSES[int(assets['cat_lab'][i])]}.png")
    return images, names


def _extract(model):
    pack = extract_features_batch(st.session_state.cl_items, feature_extractor=model[:-1])
    st.session_state.cl_pack = pack
    return pack


def _run(algo, k, pack):
    feats = pack["features"]
    if algo == "K-Means":
        return run_kmeans_clustering(feats, n_clusters=k)
    if algo == "Hierarchical Clustering":
        return run_hierarchical_clustering(feats, n_clusters=k)
    return run_dbscan_clustering(feats)


def _scatter(pack, result, names):
    labels = result["labels"]
    cluster_name = [f"Cluster {int(c) + 1}" if c >= 0 else "Noise" for c in labels]
    df = pd.DataFrame({
        "PC1": pack["coords_2d"][:, 0],
        "PC2": pack["coords_2d"][:, 1],
        "cluster": cluster_name,
        "file": names,
    })
    fig = px.scatter(
        df, x="PC1", y="PC2", color="cluster", hover_name="file",
        color_discrete_sequence=CLUSTER_COLORS,
        title="PCA → 2D Feature Space → Cluster Visualization",
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=0.5, color="#030712")))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        font_color="#e2e8f0",
        height=460,
        legend_title_text="",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    )
    return fig


def render_clustering(model, assets):
    _init_state()
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(129,140,248,0.14); border: 1px solid rgba(129,140,248,0.35); color: #c7d2fe; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; margin-bottom: 8px;">
                <span>◆</span> Unsupervised Learning
            </div>
            <h2 style="font-size: 2.1rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin-bottom: 6px;">AI Image Clustering</h2>
            <p style="font-size: 0.95rem; color: #94a3b8;">Upload multiple images and let AI automatically group visually similar images.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        "1 Upload Images",
        "2 Choose Algorithm",
        "3 Choose Number of Clusters",
        "4 Extract Features",
        "5 Run Clustering",
        "6 Visualize Clusters",
        "7 Explore Cluster Images",
    ]
    cols = st.columns(len(steps))
    for col, label in zip(cols, steps):
        num, rest = label.split(" ", 1)
        col.markdown(
            f"""<div class="sf-card" style="padding:12px; text-align:center;">
            <div class="step-number" style="width:28px;height:28px;font-size:0.8rem;margin:0 auto 8px;">{num}</div>
            <div style="font-size:0.72rem; font-weight:700; color:#e2e8f0;">{rest}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    algo = st.selectbox(
        "Select Algorithm",
        ["K-Means", "Hierarchical Clustering", "DBSCAN"],
        index=["K-Means", "Hierarchical Clustering", "DBSCAN"].index(st.session_state.cl_algo)
        if st.session_state.cl_algo in ("K-Means", "Hierarchical Clustering", "DBSCAN") else 0,
    )
    st.session_state.cl_algo = algo
    needs_k = algo != "DBSCAN"
    k = 3
    if needs_k:
        k = st.radio("Number of Clusters", [2, 3, 4, 5], index=1, horizontal=True)
    else:
        st.info("DBSCAN discovers the number of clusters from density. K is not selected.")

    with st.expander("About the Algorithm", expanded=True):
        if algo == "K-Means":
            st.markdown(
                """**K-Means Clustering**  
Type: Unsupervised Learning  
Purpose: Groups similar images based on feature similarity.  
Input: Image feature vectors  
Output: Cluster assignments  
Parameter: Number of clusters (K)"""
            )
        elif algo == "Hierarchical Clustering":
            st.markdown(
                """**Hierarchical Clustering (Ward)**  
Type: Unsupervised Learning  
Purpose: Merges closest groups until K clusters remain.  
Input: Image feature vectors  
Output: Cluster assignments  
Parameter: Number of clusters (K)"""
            )
        else:
            st.markdown(
                """**DBSCAN**  
Type: Unsupervised Learning  
Purpose: Finds dense groups and marks sparse points as noise.  
Input: Image feature vectors  
Output: Cluster assignments (K is discovered)  
Parameter: eps is estimated from pairwise distances on this feature set"""
            )

    src = st.radio("Image source", ["Upload images", "Fashion-MNIST sample from this project"], horizontal=True)
    if src.startswith("Upload"):
        files = st.file_uploader(
            "Upload 10–50 images (JPG/PNG)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
        if files:
            from PIL import Image
            sig = tuple(getattr(f, "name", "") for f in files)
            if st.session_state.upload_sig != sig:
                images = [Image.open(f) for f in files]
                names = [getattr(f, "name", f"upload_{i}.png") for i, f in enumerate(files)]
                _set_dataset(images, names)
                st.session_state.upload_sig = sig
            st.success(f"{len(st.session_state.cl_items)} image(s) loaded and ready for feature extraction.")
    else:
        n = st.slider("Sample size", 12, 80, 30)
        if st.button("Load project sample images"):
            images, names = _sample_catalogue(assets, n=n)
            _set_dataset(images, names)
            st.session_state.upload_sig = None
            st.success(f"Loaded {len(images)} images from assets.npz (actual Fashion-MNIST catalogue).")

    n_img = len(st.session_state.cl_items)
    if n_img:
        preview = st.columns(min(10, n_img))
        for i, col in enumerate(preview):
            item = st.session_state.cl_items[i]
            if hasattr(item, "mode"):
                col.image(item, width=70)
            else:
                col.image(np.asarray(item).astype("uint8"), width=70, clamp=True)

    c1, c2, c3 = st.columns([1.3, 1, 1])
    start = c1.button("Start Clustering", type="primary", use_container_width=True)
    find_k = c2.button("Find Optimal K", use_container_width=True, disabled=n_img < 4)
    if c3.button("Clear dataset", use_container_width=True):
        _set_dataset([], [])
        st.session_state.upload_sig = None
        st.rerun()

    if start:
        if n_img < 4:
            st.warning("Need at least 4 images so clustering can form groups.")
        elif needs_k and n_img <= int(k):
            st.warning(f"Need more images than K. You have {n_img} images and K={k}.")
        else:
            with st.spinner("Extracting features (existing scaler + PCA) and running clustering…"):
                pack = _extract(model)
                result = _run(algo, int(k), pack)
                st.session_state.cl_result = result
            st.success(
                f"{result['algorithm']} finished on {pack['n_samples']} feature vectors. "
                "Assignments come from those vectors, not random labels."
            )

    if find_k:
        if n_img < 4:
            st.warning("Need at least 4 images for an elbow curve.")
        else:
            with st.spinner("Computing K-Means inertia for K = 2…6…"):
                pack = _extract(model)
                st.session_state.cl_elbow = compute_elbow_curve(pack["features"], min_k=2, max_k=6)

    result = st.session_state.cl_result
    m1, m2, m3, m4, m5 = st.columns(5)
    stats = [
        ("Images Uploaded", str(n_img)),
        ("Algorithm", result["algorithm"] if result else algo),
        ("Clusters", _fmt(result["n_clusters"], "int") if result else "—"),
        ("Silhouette Score", _fmt(result["silhouette_score"]) if result else "—"),
        ("Processing Time", _fmt(result["execution_time_ms"], "ms") if result else "—"),
    ]
    for col, (label, value) in zip((m1, m2, m3, m4, m5), stats):
        col.markdown(
            f"""<div class="stat-box">
            <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">{label}</div>
            <div style="font-size:1.15rem; font-weight:700; color:#ffffff; margin-top:4px;">{value}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    if st.session_state.cl_elbow:
        elbow = st.session_state.cl_elbow
        st.markdown("#### Elbow Method for Optimal Cluster Selection")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=elbow["k_values"], y=elbow["inertias"], mode="lines+markers",
            name="Inertia", line=dict(color="#818cf8", width=3), marker=dict(size=10),
        ))
        fig.add_vline(x=elbow["optimal_k"], line_dash="dash", line_color="#38bdf8")
        fig.update_layout(
            title="Elbow Method for Optimal Cluster Selection",
            xaxis_title="K", yaxis_title="K-Means inertia",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.55)",
            font_color="#e2e8f0", height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            f"On this image set the elbow heuristic recommends **K = {elbow['optimal_k']}** "
            "(largest perpendicular distance from the inertia curve to the chord). "
            "That is a calculated estimate for this run, not a guaranteed true number of styles."
        )

    pack = st.session_state.cl_pack
    if result and pack is not None:
        extra = []
        if result.get("davies_bouldin_score") is not None:
            extra.append(f"Davies–Bouldin: {result['davies_bouldin_score']:.3f}")
        if result.get("inertia") is not None:
            extra.append(f"Inertia: {result['inertia']:.1f}")
        if result.get("noise_count"):
            extra.append(f"Noise: {result['noise_count']}")
        if result.get("eps_used") is not None:
            extra.append(f"eps: {result['eps_used']}")
        if extra:
            st.caption(" · ".join(extra) + " · Features: existing MinMaxScaler → PCA(50) pipeline")

        event = st.plotly_chart(
            _scatter(pack, result, st.session_state.cl_names),
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="cluster_scatter",
        )
        st.caption("Each point is one uploaded image. Hover for the filename. Click a point to preview it.")
        points = getattr(getattr(event, "selection", None), "points", None) or []
        if points:
            idx = int(points[0].get("point_index", points[0].get("pointNumber", 0)))
            idx = min(max(idx, 0), n_img - 1)
            p1, p2 = st.columns([1, 3])
            img = pack["processed_images"][idx]
            p1.image(np.asarray(img).astype("uint8"), clamp=True, width=140)
            lab = result["labels"][idx]
            p2.write(f"**{st.session_state.cl_names[idx]}**")
            p2.write(f"Assigned cluster: {int(lab) + 1 if lab >= 0 else 'Noise'}")

        st.markdown("#### Explore cluster images")
        labels = result["labels"]
        images = pack["processed_images"]
        for cid in sorted(set(int(c) for c in labels if c >= 0)):
            members = [i for i, lab in enumerate(labels) if int(lab) == cid]
            with st.expander(f"Cluster {cid + 1}  ·  Images: {len(members)}", expanded=cid == 0):
                st.markdown(
                    f"""<div class="sf-card" style="padding:14px;">
                    <b>Cluster {cid + 1}</b>
                    <p style="color:#94a3b8; margin:4px 0 0;">{len(members)} images grouped by feature similarity.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
                thumbs = st.columns(6)
                for j, i in enumerate(members):
                    thumbs[j % 6].image(
                        np.asarray(images[i]).astype("uint8"),
                        caption=st.session_state.cl_names[i][:22],
                        width=90,
                        clamp=True,
                    )
        noise = [i for i, lab in enumerate(labels) if int(lab) < 0]
        if noise:
            with st.expander(f"Noise  ·  Images: {len(noise)}"):
                thumbs = st.columns(6)
                for j, i in enumerate(noise):
                    thumbs[j % 6].image(
                        np.asarray(images[i]).astype("uint8"),
                        caption=st.session_state.cl_names[i][:22],
                        width=90,
                        clamp=True,
                    )


def render_comparison(model):
    _init_state()
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.3); color: #38bdf8; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; margin-bottom: 8px;">
                <span>⚖️</span> Measured on your current images
            </div>
            <h2 style="font-size: 2.1rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin-bottom: 6px;">Algorithm Comparison</h2>
            <p style="font-size: 0.95rem; color: #94a3b8;">K-Means is the primary algorithm. Hierarchical Clustering and DBSCAN run on the same extracted features. Scores are calculated — empty values stay as — when a metric is undefined.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    n_img = len(st.session_state.cl_items)
    if n_img < 6:
        st.warning("Load or upload at least 6 images in **Image Clustering**, then return here. Comparison uses that same dataset.")
        return

    k = st.radio("K for K-Means and Hierarchical Clustering", [2, 3, 4, 5], index=1, horizontal=True, key="cmp_k")
    if st.button("Run comparison on current images", type="primary"):
        with st.spinner("Fitting K-Means, Hierarchical Clustering, and DBSCAN on extracted features…"):
            pack = _extract(model)
            bench = benchmark_all_clustering(pack["features"], n_clusters=int(k))
            st.session_state.cl_compare = {"bench": bench, "n": n_img, "k": int(k)}

    data = st.session_state.cl_compare
    if not data:
        st.info("Click **Run comparison** after images are loaded. No placeholder scores are shown.")
        return

    df = data["bench"]["table"].copy()
    st.caption(f"{data['n']} images · requested K={data['k']} for K-Means and Hierarchical · features from the saved StyleFind pipeline")
    show = df.copy()
    for col in ("Silhouette Score", "Davies-Bouldin Index"):
        show[col] = show[col].map(lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.3f}")
    st.dataframe(show, use_container_width=True)
    st.markdown(
        """- **Silhouette Score**: higher is better.  
- **Davies–Bouldin Index**: lower is better.  
- **K-Means** is the main clustering method for this project. The other two are real sklearn runs on the same vectors."""
    )
