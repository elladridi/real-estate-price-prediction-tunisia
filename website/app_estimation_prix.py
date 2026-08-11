"""
STB Bank - Internal property valuation interface (villa / apartment models).

Uses the artifacts exported by the "Export des deux modeles + artefacts pour
l'interface" cell of the notebook:
    modele_villa.json (or .joblib), modele_appartement.json (or .joblib),
    villa_feature_columns.joblib, appartement_feature_columns.joblib,
    encodeur_gouvernorat.joblib, encodeur_ville.joblib,
    encodeur_code_postal.joblib, encodeur_region.joblib,
    controlled_trend.csv, reference_categories.json

Optional (95% confidence interval):
    quantile_villa.json / .joblib, quantile_appartement.json / .joblib
Without them, the app still works and just shows a point estimate.

Run with:
    pip install -r requirements.txt
    streamlit run app_estimation_prix.py

Note on the UI: the property-details card visually mirrors the STB Bank
design mockups (dark navy hero, pill toggle buttons, projection chart with a
confidence band). The postal-code field is used by the apartment model as a
real predictor, but there's currently no postal-code -> governorate/city
reverse lookup wired in, so governorate/city are still separate dropdowns
rather than auto-filled from the postal code. That's a natural next step -
see the README.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBRegressor

ARTIFACT_DIR = Path(__file__).parent

st.set_page_config(
    page_title="STB Bank - Property Valuation",
    page_icon="\U0001F3E6",
    layout="centered",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

def inject_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root{
            --navy:#0B1E3D;
            --navy-deep:#071429;
            --cyan:#2FB6E0;
            --cyan-deep:#1E93B8;
            --bg:#EEF3F8;
            --card:#FFFFFF;
            --ink:#16233B;
            --muted:#6B7A90;
            --blue-btn:#2E86D8;
            --blue-btn-deep:#1F63A8;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp{
            background:var(--bg) !important;
            color:var(--ink) !important;
            font-family:'Inter', sans-serif !important;
        }
        [data-testid="stHeader"]{ background:transparent !important; }
        .block-container{ padding-top:1.2rem !important; max-width:760px; }
        h1,h2,h3,h4{ font-family:'Inter', sans-serif !important; }

        /* ---- top navbar ---- */
        .navbar{
            display:flex; align-items:center; justify-content:space-between;
            background:var(--navy); border-radius:18px; padding:14px 22px;
            margin-bottom:18px; box-shadow:0 10px 24px rgba(11,30,61,0.18);
        }
        .navbar .brand{ display:flex; align-items:center; gap:10px; }
        .navbar .brand-icon{
            width:34px; height:34px; border-radius:10px; background:#fff;
            display:flex; align-items:center; justify-content:center; font-size:18px;
        }
        .navbar .brand-text b{ color:#fff; font-size:15px; display:block; line-height:1.1; }
        .navbar .brand-text span{ color:var(--cyan); font-size:10px; letter-spacing:.08em; font-weight:700; }
        .navbar .tabs{ display:flex; gap:22px; color:#B9C4D6; font-size:13.5px; font-weight:600; }
        .navbar .tabs .active{ color:#fff; border-bottom:2px solid var(--cyan); padding-bottom:4px; }
        .navbar .tabs .badge{
            background:var(--cyan); color:#04202B; border-radius:999px; padding:1px 7px;
            font-size:11px; font-weight:800; margin-left:4px;
        }

        /* ---- hero ---- */
        .hero{
            background:linear-gradient(135deg, var(--navy) 0%, var(--navy-deep) 100%);
            border-radius:22px; padding:30px 30px 26px; color:#fff; margin-bottom:20px;
            box-shadow:0 16px 34px rgba(11,30,61,0.22);
        }
        .hero .eyebrow{ color:var(--cyan); font-size:12px; font-weight:800; letter-spacing:.1em; margin-bottom:10px; }
        .hero h1{ font-size:26px; font-weight:800; line-height:1.25; margin:0 0 10px; color:#fff !important; }
        .hero p{ color:#AEB9CC; font-size:13.5px; line-height:1.6; max-width:520px; margin:0 0 18px; }
        .stat-row{ display:flex; gap:10px; flex-wrap:wrap; }
        .stat-tile{
            background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.08);
            border-radius:14px; padding:12px 16px; min-width:84px;
        }
        .stat-tile b{ color:var(--cyan); font-size:19px; display:block; }
        .stat-tile span{ color:#AEB9CC; font-size:11px; }

        /* ---- section card ---- */
        .section-card{
            background:var(--card); border-radius:22px; padding:22px 24px;
            box-shadow:0 10px 26px rgba(16,30,54,0.08); margin-bottom:18px;
        }
        .section-title{
            display:flex; align-items:center; gap:8px; font-weight:800; font-size:13px;
            letter-spacing:.05em; color:var(--ink); text-transform:uppercase; margin-bottom:14px;
            padding-bottom:12px; border-bottom:1px solid #EEF1F6;
        }
        .section-title .icon-box{
            width:26px; height:26px; border-radius:8px; background:var(--bg);
            display:flex; align-items:center; justify-content:center; font-size:13px;
        }

        /* ---- toggle buttons (villa/appartement) ---- */
        div[data-testid="stButton"] button{
            border-radius:14px !important; font-weight:700 !important; font-size:14.5px !important;
            padding:11px 0 !important; border:none !important;
        }
        div[data-testid="stButton"] button[kind="primary"]{
            background:linear-gradient(135deg, var(--blue-btn), var(--blue-btn-deep)) !important;
            color:#fff !important; box-shadow:0 6px 16px rgba(46,134,216,0.35) !important;
        }
        div[data-testid="stButton"] button[kind="secondary"]{
            background:#E9EEF6 !important; color:var(--muted) !important;
        }

        [data-testid="stWidgetLabel"] p{
            font-weight:700 !important; color:var(--ink) !important; font-size:13px !important;
        }
        div[data-baseweb="select"] > div, input[type="number"], input[type="text"]{
            border-radius:12px !important; border-color:#DCE3ED !important;
        }
        [data-testid="stSlider"] div[role="slider"]{ background-color:var(--cyan) !important; }

        /* ---- submit button ---- */
        div[data-testid="stFormSubmitButton"] button{
            background:linear-gradient(135deg, var(--blue-btn), var(--blue-btn-deep)) !important;
            color:#fff !important; border:none !important; border-radius:999px !important;
            padding:12px 0 !important; font-weight:700 !important; font-size:15.5px !important;
            width:100%; box-shadow:0 8px 20px rgba(46,134,216,0.35) !important;
        }

        /* ---- result banner ---- */
        .result-banner{
            background:linear-gradient(135deg, var(--navy) 0%, var(--navy-deep) 100%);
            color:#fff; border-radius:22px; padding:26px 26px 22px; text-align:center;
            box-shadow:0 16px 34px rgba(11,30,61,0.24); margin-bottom:14px;
        }
        .result-label{ color:var(--cyan); font-size:11.5px; font-weight:800; letter-spacing:.1em; margin-bottom:6px; }
        .result-value{ font-size:38px; font-weight:800; }
        .result-range{ color:#AEB9CC; font-size:13px; margin-top:4px; }
        .pill-row{ margin-top:14px; display:flex; gap:8px; justify-content:center; flex-wrap:wrap; }
        .pill{
            display:inline-block; padding:6px 14px; border-radius:999px; font-size:12.5px; font-weight:700;
        }
        .pill-dark{ background:rgba(255,255,255,0.08); color:#fff; border:1px solid rgba(255,255,255,0.15); }
        .pill-cyan{ background:rgba(47,182,224,0.18); color:var(--cyan); border:1px solid rgba(47,182,224,0.35); }

        /* ---- property summary rows ---- */
        .summary-row{
            display:flex; justify-content:space-between; padding:10px 14px; background:var(--bg);
            border-radius:12px; margin-bottom:8px; font-size:13.5px;
        }
        .summary-row span:first-child{ color:var(--muted); font-weight:600; }
        .summary-row span:last-child{ color:var(--ink); font-weight:800; }

        /* ---- projection card ---- */
        .proj-badge{
            background:var(--cyan); color:#04202B; font-weight:800; font-size:14px;
            padding:8px 16px; border-radius:999px; float:right;
        }
        .proj-note{ color:var(--muted); font-size:11.5px; margin-top:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Artifact loading (unchanged logic from the original app)
# --------------------------------------------------------------------------

def _safe_joblib_load(filename: str):
    path = ARTIFACT_DIR / filename
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(
            f"Could not load **{filename}** ({e.__class__.__name__}: {e}).\n\n"
            f"This file is likely corrupted or incomplete (often caused by an "
            f"interrupted download). Re-download just this file and replace it "
            f"in this folder, then restart the app."
        )
        st.stop()


def _load_xgb_model(base_name: str):
    json_path = ARTIFACT_DIR / f"{base_name}.json"
    joblib_path = ARTIFACT_DIR / f"{base_name}.joblib"

    if json_path.exists():
        try:
            model = XGBRegressor()
            model.load_model(str(json_path))
            return model
        except Exception as e:
            st.warning(f"{base_name}.json could not be loaded ({e}), trying {base_name}.joblib instead.")

    if joblib_path.exists():
        return _safe_joblib_load(f"{base_name}.joblib")

    st.error(f"No model file found for '{base_name}' (neither {base_name}.json nor {base_name}.joblib).")
    st.stop()


def _load_optional_xgb_model(base_name: str):
    json_path = ARTIFACT_DIR / f"{base_name}.json"
    joblib_path = ARTIFACT_DIR / f"{base_name}.joblib"

    if json_path.exists():
        try:
            model = XGBRegressor()
            model.load_model(str(json_path))
            return model
        except Exception as e:
            st.warning(f"{base_name}.json could not be loaded ({e}) - interval disabled for this segment.")
            return None

    if joblib_path.exists():
        try:
            return joblib.load(joblib_path)
        except Exception as e:
            st.warning(f"{base_name}.joblib could not be loaded ({e}) - interval disabled for this segment.")
            return None

    return None


@st.cache_resource
def load_artifacts():
    required = [
        "villa_feature_columns.joblib", "appartement_feature_columns.joblib",
        "encodeur_gouvernorat.joblib", "encodeur_ville.joblib",
        "encodeur_code_postal.joblib", "encodeur_region.joblib",
        "controlled_trend.csv", "reference_categories.json",
    ]
    missing = [f for f in required if not (ARTIFACT_DIR / f).exists()]
    has_villa_model = (ARTIFACT_DIR / "modele_villa.json").exists() or (ARTIFACT_DIR / "modele_villa.joblib").exists()
    has_appt_model = (ARTIFACT_DIR / "modele_appartement.json").exists() or (ARTIFACT_DIR / "modele_appartement.joblib").exists()
    if not has_villa_model:
        missing.append("modele_villa.json or modele_villa.joblib")
    if not has_appt_model:
        missing.append("modele_appartement.json or modele_appartement.joblib")
    if missing:
        st.error(
            "Missing files in this folder: " + ", ".join(missing) +
            "\n\nCopy all artifacts exported from the notebook next to this script."
        )
        st.stop()

    return {
        "model_villa": _load_xgb_model("modele_villa"),
        "model_appartement": _load_xgb_model("modele_appartement"),
        "quantile_model_villa": _load_optional_xgb_model("quantile_villa"),
        "quantile_model_appartement": _load_optional_xgb_model("quantile_appartement"),
        "villa_cols": _safe_joblib_load("villa_feature_columns.joblib"),
        "appartement_cols": _safe_joblib_load("appartement_feature_columns.joblib"),
        "ohe_gouvernorat": _safe_joblib_load("encodeur_gouvernorat.joblib"),
        "target_encoder_ville": _safe_joblib_load("encodeur_ville.joblib"),
        "target_encoder_postal": _safe_joblib_load("encodeur_code_postal.joblib"),
        "target_encoder_region": _safe_joblib_load("encodeur_region.joblib"),
        "controlled_trend": pd.read_csv(ARTIFACT_DIR / "controlled_trend.csv"),
        "reference": json.loads((ARTIFACT_DIR / "reference_categories.json").read_text(encoding="utf-8")),
    }


REGIONS = ["Grand Tunis", "Nord", "Cap Bon", "Sahel", "Centre", "Sfax", "Sud", "Autre"]

ZONES_TOURISTIQUES = {
    "hammamet", "yasmine hammamet", "nabeul", "la marsa", "gammarth", "carthage",
    "sidi bou said", "les berges du lac", "lac 1", "lac 2", "sousse", "port el kantaoui",
    "el kantaoui", "monastir", "skanes", "djerba", "jerba", "houmt souk", "midoun",
    "tabarka", "korbous", "mahdia",
}
INDICE5 = {"la marsa", "gammarth", "carthage", "sidi bou said", "les berges du lac", "lac 1", "lac 2"}
INDICE4 = {"hammamet", "yasmine hammamet", "port el kantaoui", "el kantaoui", "djerba", "jerba", "houmt souk", "midoun", "monastir", "skanes"}
INDICE3 = {"sousse", "nabeul", "sfax", "mahdia", "bizerte"}
INDICE2 = {"gabes", "gabès", "kairouan", "beja", "béja", "zaghouan"}


def region_from_gouvernorat(gouvernorat: str) -> str:
    g = (gouvernorat or "").lower().strip()
    if g in {"tunis", "ariana", "ben arous", "manouba"}:
        return "Grand Tunis"
    if g in {"bizerte", "béja", "beja", "jendouba", "le kef", "kef", "siliana", "zaghouan"}:
        return "Nord"
    if g in {"nabeul"}:
        return "Cap Bon"
    if g in {"sousse", "monastir", "mahdia"}:
        return "Sahel"
    if g in {"kairouan", "kasserine", "sidi bouzid"}:
        return "Centre"
    if g in {"sfax"}:
        return "Sfax"
    if g in {"gabès", "gabes", "medenine", "médenine", "tataouine", "tozeur", "kebili", "kébili", "gafsa"}:
        return "Sud"
    return "Autre"


def indice_attractivite(ville: str) -> int:
    v = (ville or "").lower().strip()
    if v in INDICE5:
        return 5
    if v in INDICE4:
        return 4
    if v in INDICE3:
        return 3
    if v in INDICE2:
        return 2
    return 1


def _prepare_input_df(art: dict, property_features: dict):
    """Builds the model-ready encoded row and returns (input_df, model, is_villa)."""
    is_villa = property_features["type_bien"] == "villa"
    model = art["model_villa"] if is_villa else art["model_appartement"]
    feature_cols = art["villa_cols"] if is_villa else art["appartement_cols"]

    input_df = pd.DataFrame([property_features])
    input_df["type_bien_is_villa"] = (input_df["type_bien"] == "villa").astype(int)
    input_df = input_df.drop(columns=["type_bien"], errors="ignore")

    gouv_enc = art["ohe_gouvernorat"].transform(input_df[["gouvernorat"]])
    gouv_df = pd.DataFrame(
        gouv_enc,
        columns=art["ohe_gouvernorat"].get_feature_names_out(["gouvernorat"]),
        index=input_df.index,
    )
    input_df = pd.concat([input_df.drop(columns=["gouvernorat"]), gouv_df], axis=1)

    global_mean = art["reference"]["global_mean_price_train"]

    input_df["ville_encoded"] = art["target_encoder_ville"].transform(input_df[["ville"]]).squeeze()
    input_df["ville_encoded"] = input_df["ville_encoded"].fillna(global_mean)

    if not is_villa:
        input_df["code_postal_encoded"] = art["target_encoder_postal"].transform(
            input_df[["code_postal"]]
        ).squeeze()
        input_df["code_postal_encoded"] = input_df["code_postal_encoded"].fillna(global_mean)

        input_df["region_encoded"] = art["target_encoder_region"].transform(
            input_df[["region"]]
        ).squeeze()
        input_df["region_encoded"] = input_df["region_encoded"].fillna(global_mean)

    input_df = input_df.drop(columns=["ville", "code_postal", "region"], errors="ignore")

    for c in set(feature_cols) - set(input_df.columns):
        input_df[c] = 0
    input_df = input_df[feature_cols]

    return input_df, model, is_villa


def base_prediction(art: dict, property_features: dict) -> dict:
    """Runs the point-estimate + quantile models ONCE and returns the raw
    (un-projected, deflated) values, so a year-by-year projection can reuse
    them without re-calling the model for every year."""
    input_df, model, is_villa = _prepare_input_df(art, property_features)
    predicted_deflated = model.predict(input_df)[0]

    quantile_model = art["quantile_model_villa"] if is_villa else art["quantile_model_appartement"]
    if quantile_model is not None:
        q_lower_deflated, _, q_upper_deflated = quantile_model.predict(input_df)[0]
        q_lower_deflated = min(q_lower_deflated, predicted_deflated)
        q_upper_deflated = max(q_upper_deflated, predicted_deflated)
    else:
        q_lower_deflated = q_upper_deflated = None

    return {
        "predicted_deflated": predicted_deflated,
        "q_lower_deflated": q_lower_deflated,
        "q_upper_deflated": q_upper_deflated,
    }


def _growth_factor(art: dict, years_from_base: float) -> float:
    trend_prices = art["controlled_trend"]["controlled_price_per_m2"].values
    yoy_growth = (trend_prices[1:] - trend_prices[:-1]) / trend_prices[:-1]
    avg_growth = np.nanmedian(yoy_growth) if len(yoy_growth) else 0.0
    return (1 + avg_growth) ** years_from_base


def forecast_price(art: dict, property_features: dict, years_ahead: int) -> dict:
    """Kept for compatibility / single-year use: point estimate + 95% CI for
    one specific horizon."""
    if years_ahead < 0:
        raise ValueError("years_ahead cannot be negative.")

    base = base_prediction(art, property_features)
    base_year = art["reference"]["base_year"]
    target_year = property_features.get("annee_publication", base_year) + years_ahead
    growth_factor = _growth_factor(art, target_year - base_year)

    result = {
        "point_estimate": max(0, base["predicted_deflated"] * growth_factor),
        "target_year": target_year,
    }
    if base["q_lower_deflated"] is not None:
        result["lower_bound_95_ci"] = max(0, base["q_lower_deflated"] * growth_factor)
        result["upper_bound_95_ci"] = max(0, base["q_upper_deflated"] * growth_factor)
    return result


def forecast_series(art: dict, property_features: dict, max_years: int) -> pd.DataFrame:
    """Year-by-year projection from 0 to max_years, reusing a single model
    call (see base_prediction) and only re-applying the trend growth factor
    per year - this is what powers the projection chart."""
    base = base_prediction(art, property_features)
    base_year = art["reference"]["base_year"]
    ref_year = property_features.get("annee_publication", base_year)

    rows = []
    for y in range(0, max_years + 1):
        target_year = ref_year + y
        growth_factor = _growth_factor(art, target_year - base_year)
        row = {
            "years_from_now": y,
            "target_year": target_year,
            "point": max(0, base["predicted_deflated"] * growth_factor),
        }
        if base["q_lower_deflated"] is not None:
            row["lower"] = max(0, base["q_lower_deflated"] * growth_factor)
            row["upper"] = max(0, base["q_upper_deflated"] * growth_factor)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# UI pieces
# --------------------------------------------------------------------------

def render_navbar():
    saved_count = len(st.session_state.get("saved", []))
    st.markdown(
        f"""
        <div class="navbar">
            <div class="brand">
                <div class="brand-icon">🏦</div>
                <div class="brand-text"><b>STB Bank</b><span>PROPERTY VALUATION</span></div>
            </div>
            <div class="tabs">
                <span class="active">🏠 Valuation</span>
                <span>💬 Forum</span>
                <span>🔖 Saved<span class="badge">{saved_count}</span></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">— PROPERTY VALUATION</div>
            <h1>Internal valuation engine<br/>for Tunisian real estate</h1>
            <p>Select property type and location, enter the details, and receive an
            indicative estimate backed by STB's proprietary models. Projections use
            the internal price-trend index built during the internship project.</p>
            <div class="stat-row">
                <div class="stat-tile"><b>2</b><span>ML Models</span></div>
                <div class="stat-tile"><b>24</b><span>Governorates</span></div>
                <div class="stat-tile"><b>265+</b><span>Cities</span></div>
                <div class="stat-tile"><b>10yr</b><span>Projection</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_property_type_toggle():
    if "type_bien" not in st.session_state:
        st.session_state.type_bien = "appartement"

    col1, col2 = st.columns(2)
    with col1:
        is_villa = st.session_state.type_bien == "villa"
        label = ("✓ " if is_villa else "") + "Villa"
        if st.button(label, key="btn_villa", use_container_width=True,
                     type="primary" if is_villa else "secondary"):
            st.session_state.type_bien = "villa"
            st.rerun()
    with col2:
        is_appt = st.session_state.type_bien == "appartement"
        label = ("✓ " if is_appt else "") + "Appartement"
        if st.button(label, key="btn_appt", use_container_width=True,
                     type="primary" if is_appt else "secondary"):
            st.session_state.type_bien = "appartement"
            st.rerun()


def render_projection_chart(art, property_features, base_year_value: int):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title"><span class="icon-box">📈</span> VALUE PROJECTION</div>',
        unsafe_allow_html=True,
    )

    horizon = st.slider("Projection horizon (years)", 0, 15, 5, key="years_ahead_slider")
    series = forecast_series(art, property_features, horizon)
    final_value = series.iloc[-1]["point"]

    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"**In {horizon} years**")
        st.caption("Based on internal price trend index")
    with right:
        st.markdown(
            f'<div class="proj-badge">{final_value:,.0f} TND</div>'.replace(",", " "),
            unsafe_allow_html=True,
        )

    fig = go.Figure()
    has_band = "lower" in series.columns

    if has_band:
        fig.add_trace(go.Scatter(
            x=list(series["years_from_now"]) + list(series["years_from_now"])[::-1],
            y=list(series["upper"]) + list(series["lower"])[::-1],
            fill="toself", fillcolor="rgba(47,182,224,0.18)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            name="95% confidence band", showlegend=True,
        ))

    fig.add_trace(go.Scatter(
        x=series["years_from_now"], y=series["point"],
        mode="lines+markers", line=dict(color="#2FB6E0", width=3),
        marker=dict(size=7, color="#2FB6E0"),
        name="Estimate",
        customdata=series["years_from_now"],
        hovertemplate="%{customdata}yr<br>%{y:,.0f} TND<extra></extra>",
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#16233B", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Years from now", showgrid=False, zeroline=False),
        yaxis=dict(title="Value (TND)", showgrid=True, gridcolor="#EEF1F6", zeroline=False),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<div class="proj-note">📌 Based on listing-price trends. Indicative only — internal use, not a formal appraisal.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    inject_style()
    art = load_artifacts()
    ref = art["reference"]

    render_navbar()
    render_hero()

    # ---- Property details card ----
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title"><span class="icon-box">🏠</span> PROPERTY DETAILS</div>',
        unsafe_allow_html=True,
    )
    render_property_type_toggle()
    type_bien = st.session_state.type_bien

    with st.form("property_form"):
        surface_habitable = st.slider(
            "Living area (m²)",
            min_value=15, max_value=max(200, ref["surface_max"]), value=120,
        )

        c1, c2 = st.columns(2)
        with c1:
            nb_pieces = st.number_input("Total rooms", min_value=1, max_value=ref["nb_pieces_max"] * 2, value=3)
        with c2:
            nb_salles_bain = st.number_input("Bathrooms", min_value=0, max_value=ref["nb_salles_bain_max"] * 2, value=1)

        c3, c4 = st.columns(2)
        with c3:
            nb_chambres = st.number_input("Bedrooms", min_value=0, max_value=ref["nb_chambres_max"] * 2, value=2)
        with c4:
            nb_etages = st.number_input("Floors", min_value=0, max_value=ref["nb_etages_max"] * 2, value=1)

        code_postal = st.text_input("Postal code", value="1000")
        c5, c6 = st.columns(2)
        with c5:
            gouvernorat = st.selectbox("Governorate", ref["gouvernorats"])
        with c6:
            ville = st.selectbox("City", ref["villes"])

        with st.expander("More details (optional)"):
            has_terrain = st.checkbox("Land plot included?")
            surface_terrain = st.number_input("Land area (m²)", min_value=0, value=0, disabled=not has_terrain)
            e1, e2 = st.columns(2)
            with e1:
                piscine = st.checkbox("Pool")
                garage = st.checkbox("Garage")
                jardin = st.checkbox("Garden", value=False)
            with e2:
                climatisation = st.checkbox("Air conditioning")
                terrasse = st.checkbox("Terrace")
                securite = st.checkbox("Security")
            annee_publication = st.number_input(
                "Reference year", min_value=2015, max_value=ref["current_year"] + 5,
                value=ref["current_year"],
            )

        submitted = st.form_submit_button("🔍 Estimate price")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        property_features = {
            "surface_habitable_m2": surface_habitable,
            "nb_pieces": nb_pieces,
            "type_bien": type_bien,
            "gouvernorat": gouvernorat,
            "ville": ville,
            "annee_publication": annee_publication,
            "surface_terrain_m2": surface_terrain if has_terrain else 0,
            "has_terrain": int(has_terrain),
            "nb_chambres": nb_chambres,
            "nb_salles_bain": nb_salles_bain,
            "nb_etages": nb_etages,
            "piscine": int(piscine),
            "garage": int(garage),
            "jardin": int(jardin or piscine),
            "climatisation": int(climatisation),
            "terrasse": int(terrasse),
            "securite": int(securite),
        }
        if type_bien == "appartement":
            property_features["code_postal"] = code_postal
            property_features["region"] = region_from_gouvernorat(gouvernorat)

        st.session_state.result_features = property_features
        st.session_state.result_code_postal = code_postal

    if "result_features" not in st.session_state:
        return

    property_features = st.session_state.result_features
    code_postal = st.session_state.result_code_postal

    # ---- Result banner (today's estimate, horizon = 0) ----
    today = forecast_price(art, property_features, 0)
    price_str = f"{today['point_estimate']:,.0f}".replace(",", " ")
    segment_badge = "Villa" if property_features["type_bien"] == "villa" else "Apartment + postal code"

    range_html = ""
    if "lower_bound_95_ci" in today:
        lower_str = f"{today['lower_bound_95_ci']:,.0f}".replace(",", " ")
        upper_str = f"{today['upper_bound_95_ci']:,.0f}".replace(",", " ")
        range_html = f'<div class="result-range">Indicative range: {lower_str} TND — {upper_str} TND</div>'

    st.markdown(
        f"""
        <div class="result-banner">
            <div class="result-label">ESTIMATED VALUE TODAY</div>
            <div class="result-value">{price_str} TND</div>
            {range_html}
            <div class="pill-row">
                <span class="pill pill-dark">{segment_badge}</span>
                <span class="pill pill-cyan">{property_features['ville']}, {property_features['gouvernorat']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Property summary + actions ----
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title"><span class="icon-box">📋</span> PROPERTY SUMMARY</div>',
        unsafe_allow_html=True,
    )
    rows = [
        ("Property type", property_features["type_bien"].capitalize()),
        ("Living area", f"{property_features['surface_habitable_m2']} m²"),
        ("Rooms / Bedrooms", f"{property_features['nb_pieces']} / {property_features['nb_chambres']}"),
        ("Bathrooms / Floors", f"{property_features['nb_salles_bain']} / {property_features['nb_etages']}"),
        ("Governorate", property_features["gouvernorat"]),
        ("City", property_features["ville"]),
    ]
    if property_features["type_bien"] == "appartement":
        rows.append(("Postal code", code_postal))

    for label, value in rows:
        st.markdown(
            f'<div class="summary-row"><span>{label}</span><span>{value}</span></div>',
            unsafe_allow_html=True,
        )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("💾 Save", use_container_width=True):
            st.session_state.setdefault("saved", []).append(dict(property_features))
            st.toast("Saved.")
            st.rerun()
    with b2:
        if st.button("📋 Copy", use_container_width=True):
            summary_text = "\n".join(f"{label}: {value}" for label, value in rows)
            st.code(summary_text, language=None)
    with b3:
        if st.button("＋ New", use_container_width=True):
            del st.session_state["result_features"]
            st.rerun()
    with b4:
        if st.button("🗑 Clear", use_container_width=True):
            del st.session_state["result_features"]
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Projection chart ----
    render_projection_chart(art, property_features, today["target_year"])


if __name__ == "__main__":
    main()
