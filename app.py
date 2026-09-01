"""
PyShore GUI — Streamlit App
============================
Run with:  streamlit run app.py

Two analysis modes:
  🛰️  GEE Mode   — extract shorelines automatically from Google Earth Engine
  📂  Manual Mode — bring your own shoreline shapefiles, assign years, run analysis
"""

import os, sys, io, json, zipfile, warnings, contextlib
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PyShore — Shoreline Change Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-title{font-size:2rem;font-weight:700;color:#1a6fa8;margin-bottom:0}
  .sub-title{font-size:.95rem;color:#555;margin-top:2px;margin-bottom:.5rem}
  .stButton>button{width:100%}
  .mode-tab{background:#f0f6ff;border-radius:8px;padding:16px;border:1px solid #c8dff5}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌊 PyShore</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated Shoreline Change Analysis · '
            'powered by Python &amp; Google Earth Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">By NERCi, NERSC | Funding: RCN</p>', unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
for k, default in [("result_gdf", None), ("intersection_gdf", None),
                   ("manual_year_map", {}), ("active_mode", "GEE"),
                   ("save_dir_override", "")]:
    if k not in st.session_state:
        st.session_state[k] = default

TREND_COLORS = {
    "Eroding":   "#d73027",
    "Stable":    "#4575b4",
    "Accreting": "#1a9850",
    "Uncertain": "#fee08b",
    "No Data":   "#d9d9d9",
}

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR — shared parameters
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Shared Parameters")

    st.subheader("📁 Baseline & Transects")
    baseline_shp = st.text_input(
        "Baseline shapefile",
        value="",
        help="Shoreline used to generate perpendicular transects"
    )
    col3, col4 = st.columns(2)
    t_spacing = col3.number_input("Spacing (m)", value=100, min_value=10, max_value=1000)
    t_length  = col4.number_input("Length (m)",  value=2000, min_value=100, max_value=10000)
    epsg = st.number_input("EPSG", value=32643, help="Projected CRS — UTM 43N for Kerala")

    st.subheader("📊 Analysis")
    min_years   = st.slider("Min years for regression", 2, 5, 3)
    bootstrap_n = st.select_slider("Bootstrap iterations",
                                    options=[100, 200, 500, 1000], value=500)

    st.divider()
    save_btn = st.button("💾  Save Results", use_container_width=True,
                          disabled=(st.session_state.result_gdf is None))


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _run_stage(label, pct, progress_bar, status_box, fn, *args, **kwargs):
    progress_bar.progress(pct, text=label)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    out = buf.getvalue().strip()
    if out:
        status_box.code(out[-2000:], language=None)
    return result


def _display_results(result, inter, min_years):
    """Render the 5 result tabs — shared by both GEE and Manual modes."""
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Summary", "🗺️ Trend Map", "📈 Plots", "📋 Data Table", "📉 Time Series"]
    )

    # ── Summary ────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Overall Statistics")
        n = len(result)
        c1, c2, c3, c4 = st.columns(4)
        nsm_v = result["NSM"].dropna().mean() if "NSM" in result else np.nan
        epr_v = result["EPR"].dropna().mean() if "EPR" in result else np.nan
        lrr_v = result["LRR"].dropna().mean() if "LRR" in result else np.nan
        n_lrr = int(result["LRR"].notna().sum()) if "LRR" in result else 0
        c1.metric("Transects", n)
        c2.metric("Mean NSM",  f"{nsm_v:.1f} m"   if not np.isnan(nsm_v) else "—")
        c3.metric("Mean EPR",  f"{epr_v:.2f} m/yr" if not np.isnan(epr_v) else "—")
        c4.metric("Mean LRR",  f"{lrr_v:.2f} m/yr" if not np.isnan(lrr_v) else "—",
                  help=f"{n_lrr} transects with ≥{min_years} observations")

        st.divider()
        if "Trend_LRR" in result.columns:
            tc = result["Trend_LRR"].value_counts()
            valid_tc = tc[tc.index != "No Data"]
            if not valid_tc.empty:
                st.subheader("Trend Classification (LRR)")
                icons = {"Eroding":"🔴","Accreting":"🟢","Stable":"🔵","Uncertain":"🟡"}
                cols = st.columns(len(valid_tc))
                for col, (trend, cnt) in zip(cols, valid_tc.items()):
                    col.metric(f"{icons.get(trend,'')} {trend}", cnt, f"{100*cnt/n:.1f}%")
            else:
                st.info(f"Trend classification needs ≥{min_years} years per transect. "
                        "Check NSM/EPR in the Data Table tab — those are computed from 2 years.")

        st.divider()
        st.subheader("Metric Summary")
        cols_show = [c for c in ["NSM","EPR","SCE","LRR","WLR","LMS"] if c in result.columns]
        if cols_show:
            summary = result[cols_show].describe().T[["mean","std","min","50%","max"]]
            summary.columns = ["Mean","Std Dev","Min","Median","Max"]
            st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)

    # ── Trend Map ──────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Shoreline Change Trend Map")
        avail = [c for c in ["Trend_LRR","Trend_WLR","Trend_LMS","LRR","WLR","LMS","EPR","NSM"]
                 if c in result.columns]
        trend_col = st.selectbox("Colour by", avail, key="map_trend_sel")

        fig, ax = plt.subplots(figsize=(13, 8))
        if trend_col and trend_col.startswith("Trend_"):
            for trend, color in TREND_COLORS.items():
                sub = result[result[trend_col] == trend]
                if not sub.empty:
                    sub.plot(ax=ax, color=color, linewidth=1.5, label=trend)
            patches = [mpatches.Patch(color=c, label=t) for t, c in TREND_COLORS.items()
                       if t in result[trend_col].unique() and t != "No Data"]
            if patches:
                ax.legend(handles=patches, loc="lower left", fontsize=9)
        elif trend_col:
            valid_r = result.dropna(subset=[trend_col])
            if not valid_r.empty:
                vmin = valid_r[trend_col].quantile(0.05)
                vmax = valid_r[trend_col].quantile(0.95)
                valid_r.plot(ax=ax, column=trend_col, cmap="RdYlGn",
                             vmin=vmin, vmax=vmax, legend=True,
                             legend_kwds={"label": trend_col, "shrink": 0.6})
        ax.set_title(f"PyShore — {trend_col}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
        ax.axis("equal"); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Plots ──────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Rate Distributions")
        fig_h, axes_h = plt.subplots(1, 3, figsize=(15, 4))
        for ax_h, (col, clr) in zip(axes_h, [("LRR","#4393c3"),("WLR","#d6604d"),("LMS","#74c476")]):
            data = result[col].dropna() if col in result.columns else pd.Series(dtype=float)
            if data.empty: ax_h.set_title(f"{col} — no data"); continue
            sns.histplot(data, bins=30, kde=True, color=clr, ax=ax_h, edgecolor="white")
            ax_h.axvline(0, color="black", lw=0.8, ls="--")
            ax_h.set_xlabel(f"{col} (m/yr)"); ax_h.set_title(f"Distribution of {col}")
        plt.tight_layout(); st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)

        st.subheader("Along-Coast Profile (NSM & EPR)")
        df_s = result.sort_values("transect_id")
        fig_p, axes_p = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        for ax_p, col, label, clr in [
            (axes_p[0], "NSM", "NSM (m)",    "#4393c3"),
            (axes_p[1], "EPR", "EPR (m/yr)", "#d6604d"),
        ]:
            if col not in df_s.columns: continue
            xp = df_s["transect_id"].values; yp = df_s[col].values
            ax_p.fill_between(xp, yp, 0, where=(yp > 0), alpha=0.4, color="#1a9850", label="Accretion")
            ax_p.fill_between(xp, yp, 0, where=(yp < 0), alpha=0.4, color="#d73027", label="Erosion")
            ax_p.plot(xp, yp, color=clr, lw=0.7); ax_p.axhline(0, color="black", lw=0.8)
            ax_p.set_ylabel(label); ax_p.legend(fontsize=8)
        axes_p[1].set_xlabel("Transect ID"); plt.tight_layout()
        st.pyplot(fig_p, use_container_width=True); plt.close(fig_p)

        if "LRR_bs_ci_lo" in result.columns:
            st.subheader("LRR with Bootstrap 95% CI")
            df_bs = result.dropna(subset=["LRR","LRR_bs_ci_lo","LRR_bs_ci_hi"]).sort_values("transect_id")
            if not df_bs.empty:
                fig_bs, ax_bs = plt.subplots(figsize=(14, 4))
                xb = df_bs["transect_id"].values; yb = df_bs["LRR"].values
                lo = yb - df_bs["LRR_bs_ci_lo"].values; hi = df_bs["LRR_bs_ci_hi"].values - yb
                ax_bs.errorbar(xb, yb, yerr=[lo, hi], fmt="none", ecolor="steelblue", alpha=0.4, lw=0.6)
                cb = df_bs["Trend_LRR"].map(TREND_COLORS).fillna("#d9d9d9") \
                     if "Trend_LRR" in df_bs.columns else "steelblue"
                ax_bs.scatter(xb, yb, c=cb, s=8, zorder=3)
                ax_bs.axhline(0, color="black", lw=0.8, ls="--")
                ax_bs.set_xlabel("Transect ID"); ax_bs.set_ylabel("LRR (m/yr)")
                plt.tight_layout(); st.pyplot(fig_bs, use_container_width=True); plt.close(fig_bs)

    # ── Data Table ─────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Full Metrics Table")
        df_show = pd.DataFrame(result[[c for c in result.columns if c != "geometry"]])
        st.dataframe(
            df_show.style.format(
                {c: "{:.3f}" for c in df_show.select_dtypes("number").columns}, na_rep="—"
            ),
            use_container_width=True, height=500,
        )
        csv_bytes = df_show.to_csv(index=False, float_format="%.4f").encode()
        st.download_button("⬇️  Download metrics CSV", data=csv_bytes,
                           file_name="pyshore_metrics.csv", mime="text/csv")

    # ── Time Series ────────────────────────────────────────────────────────
    with tab5:
        st.subheader("Transect Time Series")
        if inter is not None:
            all_ids = sorted(result["transect_id"].dropna().astype(int).tolist())
            step = max(1, len(all_ids) // 5)
            selected = st.multiselect("Select transects", options=all_ids,
                                       default=all_ids[::step][:5], key="ts_sel")
            if selected:
                fig_ts, axes_ts = plt.subplots(1, len(selected),
                                                figsize=(5*len(selected), 4), squeeze=False)
                for ax_ts, tid in zip(axes_ts[0], selected):
                    grp = inter[inter["transect_id"] == tid].sort_values("year")
                    if grp.empty: ax_ts.set_title(f"T{tid} — no data"); continue
                    ax_ts.scatter(grp["year"], grp["distance_m"], color="steelblue", s=40, zorder=5)
                    ax_ts.plot(grp["year"], grp["distance_m"], color="steelblue", alpha=0.4)
                    row = result[result["transect_id"] == tid]
                    if not row.empty:
                        for col, clr, lbl in [("LRR","#4393c3","LRR"),
                                               ("WLR","#d6604d","WLR"),
                                               ("LMS","#74c476","LMS")]:
                            if col not in row.columns: continue
                            rate = row[col].values[0]
                            ic_col = f"{col}_intercept"
                            ic = row[ic_col].values[0] if ic_col in row.columns else np.nan
                            if np.isnan(rate) or np.isnan(ic): continue
                            xs = np.array([grp["year"].min(), grp["year"].max()])
                            ax_ts.plot(xs, ic + rate * xs, color=clr, lw=1.5,
                                       label=f"{lbl}: {rate:.2f} m/yr")
                    ax_ts.legend(fontsize=7)
                    ax_ts.set_title(f"Transect {tid}")
                    ax_ts.set_xlabel("Year"); ax_ts.set_ylabel("Distance (m)")
                plt.tight_layout(); st.pyplot(fig_ts, use_container_width=True); plt.close(fig_ts)
        else:
            st.info("Run analysis first.")


def _run_pipeline(ps, run_extraction, gee_project, log_box, progress, status_box):
    """Execute pipeline stages with progress feedback."""
    if run_extraction:
        _run_stage("🛰️ Extracting from GEE…", 10, progress, status_box,
                   ps._run_extraction, gee_project)
    else:
        progress.progress(10, text="Using existing shapefiles…")

    _run_stage("📂 Loading shorelines…",     25, progress, status_box, ps._load_shorelines)
    _run_stage("📏 Building transects…",      40, progress, status_box, ps._build_transects)
    _run_stage("🔗 Computing intersections…", 55, progress, status_box, ps._compute_intersections)
    _run_stage("📐 Computing metrics…",       70, progress, status_box, ps._compute_metrics)
    _run_stage("📉 Uncertainty analysis…",    85, progress, status_box, ps._compute_uncertainty)
    progress.progress(95, text="Assembling results…")
    ps._assemble_result()
    progress.progress(100, text="Done ✅")

    st.session_state.result_gdf       = ps.result_gdf
    st.session_state.intersection_gdf = ps.intersection_gdf
    log_box.success(f"✅ Analysis complete — {len(ps.result_gdf)} transects processed.")
    status_box.empty()
    st.rerun()   # re-render so sidebar Save button picks up the new result


# ═══════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════
if save_btn and st.session_state.result_gdf is not None:
    from pyshore.export.exporter import export_results
    from pyshore.visualization.plots import generate_all_plots
    # Use manual override if set, otherwise derive from sidebar baseline path
    save_dir = (st.session_state.save_dir_override.strip()
                or os.path.join(os.path.dirname(os.path.abspath(baseline_shp)),
                                "PyShore_output"))
    os.makedirs(save_dir, exist_ok=True)
    with st.spinner(f"Saving to {save_dir} …"):
        # ── Step 1: data files (CSV, Shapefile, GeoPackage) ──────────────
        try:
            export_results(st.session_state.result_gdf, st.session_state.intersection_gdf,
                           output_dir=save_dir, prefix="pyshore",
                           csv=True, shapefile=True, geopackage=True)
        except Exception as _export_err:
            st.error(f"❌ Export failed: {_export_err}")
            st.stop()

        # ── Step 2: plots (non-fatal — data files already written) ───────
        try:
            generate_all_plots(st.session_state.result_gdf, st.session_state.intersection_gdf,
                               output_dir=os.path.join(save_dir, "plots"))
        except Exception as _plot_err:
            st.warning(f"⚠️  Plots skipped ({_plot_err}). Data files were saved successfully.")

    # ── ZIP download ──────────────────────────────────────────────────────
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(save_dir):
            for f in files:
                fp = os.path.join(root, f)
                zf.write(fp, os.path.relpath(fp, save_dir))
    zip_buf.seek(0)
    st.success(f"✅ Saved to **{save_dir}**  "
               f"(CSV + Shapefile + GeoPackage + plots)")
    st.download_button("⬇️  Download all results as ZIP", data=zip_buf,
                       file_name="PyShore_results.zip", mime="application/zip")


# ═══════════════════════════════════════════════════════════════════════════
#  MODE SELECTOR
# ═══════════════════════════════════════════════════════════════════════════
mode = st.radio("Analysis mode", ["🛰️ GEE Extraction", "📂 Manual Shapefiles"],
                horizontal=True, label_visibility="collapsed")
st.divider()


# ═══════════════════════════════════════════════════════════════════════════
#  MODE A — GEE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
if mode == "🛰️ GEE Extraction":
    with st.expander("🛰️ GEE Settings", expanded=True):
        col_g1, col_g2 = st.columns(2)
        gee_project  = col_g1.text_input("GEE Project ID", value="rr-geefiles")
        water_index  = col_g2.selectbox("Water index", ["MNDWI", "NDWI"])
        col_g3, col_g4, col_g5 = st.columns(3)
        start_year   = col_g3.number_input("Start year", value=2017, min_value=1984, max_value=2024)
        end_year     = col_g4.number_input("End year",   value=2023, min_value=1985, max_value=2024)
        cloud_thresh = col_g5.slider("Max cloud %", 5, 50, 20)
        shoreline_dir = st.text_input("Shoreline output folder",
                                       value="shoreline_output")
        aoi_shp = st.text_input("AOI shapefile (leave blank to auto-derive from baseline)",
                                 value="")
        run_extraction = st.checkbox("Re-run GEE extraction", value=False,
                                      help="Uncheck to reuse existing shapefiles")

    run_gee_btn = st.button("▶  Run GEE Analysis", type="primary", use_container_width=True)

    if run_gee_btn:
        from pyshore import PyShoreConfig, ExtractionConfig, AnalysisConfig, OutputConfig
        from pyshore.pipeline import PyShore

        st.session_state.result_gdf = None
        log_box  = st.empty(); log_box.info("Starting…")
        progress = st.progress(0, text="Initialising…")
        status   = st.empty()

        try:
            cfg = PyShoreConfig(
                extraction=ExtractionConfig(
                    start_year=int(start_year), end_year=int(end_year),
                    output_dir=shoreline_dir, aoi_shapefile=aoi_shp,
                    baseline_shapefile=baseline_shp, aoi_buffer_m=5000,
                    water_index=water_index, threshold_method="otsu",
                    prefer_sentinel2=True, cloud_threshold=cloud_thresh,
                    target_epsg=int(epsg),
                ),
                analysis=AnalysisConfig(
                    baseline_shapefile=baseline_shp, shoreline_dir=shoreline_dir,
                    transect_spacing=float(t_spacing), transect_length=float(t_length),
                    min_years=int(min_years), min_detectable_change=5.0,
                    epsg=int(epsg), bootstrap_n=int(bootstrap_n), confidence_level=0.95,
                ),
                output=OutputConfig(output_dir="_tmp", csv=False, shapefile=False,
                                    geopackage=False, plots=False),
            )
            ps = PyShore(cfg)
            _run_pipeline(ps, run_extraction, gee_project, log_box, progress, status)
        except Exception as e:
            log_box.error(f"❌ {e}")
            progress.empty(); st.stop()

    if st.session_state.result_gdf is not None:
        st.divider()
        _display_results(st.session_state.result_gdf,
                         st.session_state.intersection_gdf, min_years)
    elif not run_gee_btn:
        st.info("👆 Configure settings above and click **▶ Run GEE Analysis**.")


# ═══════════════════════════════════════════════════════════════════════════
#  MODE B — MANUAL SHAPEFILES
# ═══════════════════════════════════════════════════════════════════════════
else:
    st.subheader("📂 Manual Shoreline Analysis")
    st.markdown(
        "Point to a folder containing your shoreline shapefiles "
        "(one per time period). PyShore will detect files automatically and let you "
        "assign years and positional uncertainty before running the analysis."
    )

    # ── Baseline configuration (prominent) ─────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 🗺️ Baseline Shoreline")
        st.caption(
            "The **baseline** is the reference line used to generate perpendicular transects. "
            "It is typically your oldest survey or a central reference shoreline. "
            "It is **not** one of the analysis shorelines — it only defines transect positions."
        )
        col_bl1, col_bl2 = st.columns([3, 1])
        manual_baseline = col_bl1.text_input(
            "Baseline shapefile path",
            value=baseline_shp,
            key="manual_baseline_path",
            help="A single LineString or MultiLineString shapefile along the coast"
        )
        col_bl2.markdown("<br>", unsafe_allow_html=True)
        if os.path.exists(manual_baseline):
            col_bl2.success("✅ File found")
        else:
            col_bl2.error("❌ Not found")

        st.markdown("#### 💾 Output Folder")
        col_out1, col_out2 = st.columns([3, 1])
        _bl_dir = os.path.dirname(os.path.abspath(manual_baseline)) \
                  if os.path.exists(manual_baseline) else "."
        _default_out = st.session_state.save_dir_override or \
                       os.path.join(_bl_dir, "PyShore_output")
        manual_output_dir = col_out1.text_input(
            "Save results to",
            value=_default_out,
            key="manual_output_dir",
            help="CSV, Shapefile, GeoPackage and plots will be saved here when you click 💾 Save Results"
        )
        st.session_state.save_dir_override = manual_output_dir
        col_out2.markdown("<br>", unsafe_allow_html=True)
        if os.path.isdir(manual_output_dir):
            col_out2.success("✅ Exists")
        else:
            col_out2.info("📁 Will be created")

    # ── Input: folder path ─────────────────────────────────────────────────
    manual_folder = st.text_input(
        "Folder containing shoreline shapefiles",
        value="",
        placeholder="e.g. D:/data/my_shorelines",
        help="All .shp files in this folder will be listed below"
    )

    shp_files = []
    if manual_folder and os.path.isdir(manual_folder):
        shp_files = sorted([f for f in os.listdir(manual_folder) if f.endswith(".shp")])

    if manual_folder and not os.path.isdir(manual_folder):
        st.warning("⚠️  Folder not found. Check the path.")

    # ── Upload fallback ────────────────────────────────────────────────────
    with st.expander("📤 Or upload shapefile ZIPs (one ZIP per shoreline period)", expanded=False):
        st.caption("Each ZIP should contain the .shp, .dbf, .shx, .prj files for one shoreline.")
        uploaded_zips = st.file_uploader(
            "Upload ZIP files", type="zip", accept_multiple_files=True, key="zip_upload"
        )
        if uploaded_zips:
            import tempfile as _tmp
            upload_dir = os.path.join(os.path.dirname(__file__), "_manual_upload")
            os.makedirs(upload_dir, exist_ok=True)
            for zf_obj in uploaded_zips:
                with zipfile.ZipFile(io.BytesIO(zf_obj.read())) as zf:
                    zf.extractall(upload_dir)
            if not manual_folder:
                manual_folder = upload_dir
                shp_files = sorted([f for f in os.listdir(upload_dir) if f.endswith(".shp")])
                st.success(f"Extracted {len(shp_files)} shapefile(s) to temporary folder.")

    # ── Year & uncertainty assignment table ────────────────────────────────
    if shp_files:
        st.subheader(f"📋 Found {len(shp_files)} shapefile(s) — assign years")
        st.caption("Edit the Year and Uncertainty columns. "
                   "Year = integer (e.g. 2017). "
                   "Uncertainty = positional accuracy in metres "
                   "(5 m for Sentinel-2, 15 m for Landsat, 10–30 m for GPS/manual).")

        # Auto-detect years from filename
        def _guess_year(fname):
            import re
            m = re.search(r"(19|20)\d{2}", fname)
            return int(m.group()) if m else None

        init_rows = []
        for f in shp_files:
            yr = _guess_year(f)
            init_rows.append({
                "File": f,
                "Year": yr if yr else 2000,
                "Uncertainty_m": 10.0,
                "Include": True,
            })

        edited = st.data_editor(
            pd.DataFrame(init_rows),
            column_config={
                "File":          st.column_config.TextColumn("Filename", disabled=True),
                "Year":          st.column_config.NumberColumn("Year", min_value=1980,
                                                                max_value=2030, step=1,
                                                                format="%d"),
                "Uncertainty_m": st.column_config.NumberColumn("Uncertainty (m)",
                                                                min_value=0.5, max_value=100.0,
                                                                step=0.5, format="%.1f"),
                "Include":       st.column_config.CheckboxColumn("Include?"),
            },
            use_container_width=True,
            num_rows="fixed",
            key="year_editor",
        )

        included = edited[edited["Include"] == True]
        if included.empty:
            st.warning("Select at least one file to include.")
        else:
            n_included = len(included)
            years_list = sorted(included["Year"].astype(int).tolist())
            if len(set(years_list)) != n_included:
                st.error("⚠️  Duplicate years detected — each file must have a unique year.")
            else:
                st.caption(f"✅ {n_included} shoreline(s) selected: {years_list}")

                # Baseline year selector
                st.markdown("**📌 Baseline year for NSM / EPR**")
                st.caption(
                    "NSM = distance moved *from* this year to the most recent year. "
                    "EPR = NSM ÷ elapsed years. Choose the earliest year as the reference."
                )
                baseline_year_sel = st.selectbox(
                    "Baseline year", options=years_list,
                    index=0, key="manual_baseline_year",
                    help="The year that acts as the 'zero' position for Net Shoreline Movement"
                )

                # ── Run Manual Analysis ────────────────────────────────────
                run_manual_btn = st.button("▶  Run Manual Analysis",
                                            type="primary", use_container_width=True)

                if run_manual_btn:
                    from pyshore.analysis.transects import generate_transects
                    from pyshore.analysis.intersection import intersect_transects
                    from pyshore.analysis.metrics import compute_all_metrics
                    from pyshore.analysis.uncertainty import compute_uncertainty

                    st.session_state.result_gdf = None
                    log_box  = st.empty(); log_box.info("Starting manual analysis…")
                    progress = st.progress(0, text="Loading files…")
                    status   = st.empty()

                    try:
                        # 1. Load baseline & build transects
                        progress.progress(10, text="Loading baseline & building transects…")
                        bl_path = manual_baseline if os.path.exists(manual_baseline) else baseline_shp
                        baseline_gdf = gpd.read_file(bl_path).to_crs(epsg=int(epsg))
                        transects = generate_transects(
                            baseline_gdf,
                            spacing=float(t_spacing),
                            length=float(t_length),
                        )

                        # 2. Load each selected shoreline shapefile
                        progress.progress(25, text="Loading shoreline shapefiles…")
                        shorelines = []
                        for _, row in included.iterrows():
                            fpath = os.path.join(manual_folder, row["File"])
                            gdf = gpd.read_file(fpath)
                            # Reproject to working CRS
                            if gdf.crs is None:
                                gdf = gdf.set_crs(epsg=4326)
                            gdf = gdf.to_crs(epsg=int(epsg))
                            gdf["year"]          = int(row["Year"])
                            gdf["uncertainty_m"] = float(row["Uncertainty_m"])
                            gdf["sensor"]        = "manual"
                            shorelines.append(gdf)

                        # CRS / bounds info
                        tb = transects.total_bounds
                        status.code(
                            f"Transects : x=[{tb[0]:.0f}, {tb[2]:.0f}]  "
                            f"y=[{tb[1]:.0f}, {tb[3]:.0f}]\n"
                            f"Shorelines: {len(shorelines)} file(s), "
                            f"years={sorted(years_list)}",
                            language=None
                        )

                        # 3. Intersect
                        progress.progress(45, text="Computing intersections…")
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            inter_gdf = intersect_transects(transects, shorelines)
                        status.code(buf.getvalue().strip(), language=None)


                        # 4. Metrics
                        progress.progress(65, text="Computing metrics…")
                        metrics_df = compute_all_metrics(
                            inter_gdf,
                            min_years=int(min_years),
                            min_detectable_change=5.0,
                            baseline_year=int(baseline_year_sel),
                        )

                        # 5. Uncertainty
                        progress.progress(80, text="Uncertainty analysis…")
                        metrics_df = compute_uncertainty(
                            inter_gdf, metrics_df,
                            bootstrap_n=int(bootstrap_n),
                            confidence=0.95,
                            baseline_year=int(baseline_year_sel),
                            min_years=int(min_years),
                        )

                        # 6. Assemble result GeoDataFrame
                        progress.progress(95, text="Assembling results…")
                        result_gdf = transects.copy()
                        result_gdf = result_gdf.merge(metrics_df, on="transect_id", how="left")

                        progress.progress(100, text="Done ✅")
                        st.session_state.result_gdf       = result_gdf
                        st.session_state.intersection_gdf = inter_gdf
                        log_box.success(
                            f"✅ Manual analysis complete — {len(result_gdf)} transects, "
                            f"{n_included} shorelines, years {sorted(years_list)}"
                        )
                        status.empty()
                        st.rerun()

                    except Exception as e:
                        import traceback
                        log_box.error(f"❌ {e}")
                        st.code(traceback.format_exc(), language="python")
                        progress.empty()

    elif manual_folder and os.path.isdir(manual_folder):
        st.info("No .shp files found in that folder.")
    else:
        st.info("👆 Enter a folder path or upload ZIP files above.")
        with st.expander("ℹ️ Supported formats & file naming", expanded=False):
            st.markdown("""
**Folder mode** — put all your shoreline shapefiles in one folder.
PyShore auto-detects the year from the filename.

**Positional uncertainty** — enter positional accuracy in metres:
- Sentinel-2 extracted: ~5 m
- Landsat extracted: ~15 m
- GPS survey: ~1–3 m
- Manual digitising: ~5–30 m
            """)

    # Show results if available
    if st.session_state.result_gdf is not None:
        st.divider()
        _display_results(st.session_state.result_gdf,
                         st.session_state.intersection_gdf, min_years)
