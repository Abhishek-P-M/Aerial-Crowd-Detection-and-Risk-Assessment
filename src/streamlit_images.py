# streamlit_images.py
import os, glob
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Image Risk Dashboard", layout="wide")
st.title("🛰️ Aerial Congestion & Risk — Images Only")

# --- Inputs (editable in UI) ---
default_metrics = "outputs/analytics/metrics.csv"
default_over    = "outputs/analytics/overlays"
default_hm      = "outputs/analytics/heatmaps"

metrics_csv = st.text_input("metrics.csv path", value=default_metrics)
over_dir    = st.text_input("Overlays dir", value=default_over)
hm_dir      = st.text_input("Heatmaps dir (optional)", value=default_hm)

# --- Load metrics ---
if not os.path.exists(metrics_csv):
    st.warning(f"metrics.csv not found at: {metrics_csv}\nRun image_analytics.py first or fix the path above.")
    st.stop()

df = pd.read_csv(metrics_csv)

# --- Sidebar filters ---
st.sidebar.header("Filters")
min_conf = st.sidebar.number_input("Min proximity risk (PRI)", 0.0, 9999.0, 0.0, 0.1)
min_ci   = st.sidebar.number_input("Min congestion index (CI)", 0.0, 9999.0, 0.0, 0.1)
sort_by  = st.sidebar.selectbox("Sort by", ["proximity_risk_index", "congestion_index", "occupancy_frac", "total_detections"])
top_n    = st.sidebar.slider("Show top N rows", 5, 200, 20)

mask = (df["proximity_risk_index"] >= min_conf) & (df["congestion_index"] >= min_ci)
df_f = df[mask].copy()
df_f = df_f.sort_values([sort_by, "congestion_index"], ascending=False)

# --- KPIs ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Images", len(df))
c2.metric("Mean PRI", f"{df['proximity_risk_index'].mean():.2f}")
c3.metric("Mean CI", f"{df['congestion_index'].mean():.2f}")
c4.metric("Mean Occupancy", f"{100*df['occupancy_frac'].mean():.1f}%")

# --- Top table ---
st.subheader("Top risky frames")
st.dataframe(df_f.head(top_n), use_container_width=True)

# --- Charts ---
st.subheader("Distributions")
cc1, cc2, cc3 = st.columns(3)
with cc1: st.bar_chart(df_f["proximity_risk_index"])
with cc2: st.bar_chart(df_f["congestion_index"])
with cc3: st.bar_chart(df_f["occupancy_frac"])

# --- Frame browser ---
st.subheader("Browse a frame")
imgs = sorted(glob.glob(os.path.join(over_dir, "*")))
if not imgs:
    st.info(f"No overlays found in {over_dir}. Run image_analytics.py or adjust the path.")
else:
    # try to sync selection between table and image list
    names_from_table = df_f["image"].tolist()
    # pick index from overlays folder
    options = [os.path.basename(p) for p in imgs]
    default_sel = 0
    if names_from_table:
        try:
            default_sel = options.index(names_from_table[0])
        except ValueError:
            default_sel = 0

    idx = st.slider("Index", 0, len(imgs)-1, default_sel)
    cur = imgs[idx]
    st.image(cur, caption=os.path.basename(cur), use_container_width=True)

    # Show heatmap if available
    if os.path.exists(hm_dir):
        base = os.path.basename(cur)
        cand_same = os.path.join(hm_dir, base)  # in case you saved same name/extension
        cand_png  = os.path.join(hm_dir, os.path.splitext(base)[0] + "_heatmap.png")  # as in image_analytics.py
        if os.path.exists(cand_same):
            st.image(cand_same, caption="Heatmap", use_container_width=True)
        elif os.path.exists(cand_png):
            st.image(cand_png, caption="Heatmap", use_container_width=True)
        else:
            st.caption("No heatmap found for this frame.")
