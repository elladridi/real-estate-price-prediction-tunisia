# Predicting Future Real Estate Prices in Tunisia

ML pipeline forecasting 3/5-year future real estate prices in Tunisia — CRISP-DM
methodology, dual XGBoost models (villa/apartment), quantile-regression uncertainty
intervals, and a Streamlit valuation demo.

Six-week, internship project. Built with the **CRISP-DM** methodology:
7 raw sources merged into a 42,161-row dataset, cleaned down to 12,971 usable
listings, feature-engineered, and modeled with two specialized XGBoost regressors
plus a quantile-regression uncertainty layer.

## Results

| Segment    | R² (test set) | Test set size |
|------------|---------------|----------------|
| Villa      | **0.806**     | 653 listings   |
| Apartment  | **0.570**     | 1,942 listings |

- 95% prediction intervals cover the true price **93.9%** of the time on unseen data.
- Median interval width ≈ 534,191 TND (wide, reflecting the underlying R²).

## Architecture

```
7 raw sources ──► merge into shared schema (42,161 rows)
                          │
                          ▼
              clean, impute, deduplicate (12,971 rows)
                          │
                          ▼
        feature engineering: postal code, region, tourism-zone flag,
        controlled price-trend index (2018 base year)
                          │
                          ▼
     ┌────────────────────┴────────────────────┐
     ▼                                          ▼
 Villa XGBoost model                Apartment XGBoost model
 (no postal code/region —                (+ postal code, region —
  too little data to support it)          enough data to benefit)
     │                                          │
     └────────────────────┬────────────────────┘
                          ▼
           quantile-regression models (2.5th/97.5th pct)
                          │
                          ▼
         forecast_price(property_features, years_ahead)
                          │
                          ▼
                  Streamlit test interface
```

**Why two separate models instead of one?** Adding richer location features
(postal code, region) to a single combined model helped apartments slightly
(R² 0.536 → 0.550) but hurt villas badly (R² 0.734 → 0.641) — villa listings
are too sparse for fine-grained location categories, so the model overfit.
Splitting into two models, each with a feature set tailored to its segment,
fixed this without a tradeoff.

## Tech stack

- **Data handling:** pandas, NumPy
- **Scraping:** requests, BeautifulSoup, Tayara's internal JSON API
- **Modeling:** scikit-learn, XGBoost, LightGBM
- **Interpretation:** SHAP, statsmodels (VIF)
- **Persistence:** joblib
- **Deployment demo:** Streamlit, Plotly

## Repo structure

```
├── README.md
├── .gitignore
├── file/
│   └── requirements.txt
├── notebook/
│   └── stage_final.ipynb        # full CRISP-DM pipeline
├── reports/
│   └── report(46pages).pdf      # full written report
└── website/
    └── app_estimation_prix.py   # Streamlit demo interface
```

> **Note:** trained model artifacts (`.joblib` / `.json` model files, encoders,
> reference tables) and the raw/cleaned datasets (`data/`, `train/`, `test/`)
> are intentionally not tracked in this repo — see `.gitignore`. Re-run the
> notebook to regenerate them, then drop them next to `app_estimation_prix.py`
> before launching the app. Raw listing data isn't redistributed here since
> some source data was scraped from third-party sites (Tayara.tn, Mubawab.tn).

## Running the demo interface

```bash
pip install -r file/requirements.txt
streamlit run website/app_estimation_prix.py
```

The app expects the exported model artifacts (see the docstring at the top of
`app_estimation_prix.py` for the full file list) in the same `website/` folder.

## Key limitations

- The model predicts **listing/asking prices**, not official appraised values.
- The villa price trend is **borrowed from the apartment trend** — there isn't
  enough multi-year villa history to fit an independent one.
- Property condition and year built were **unavailable** (0% fill rate in the
  source data) despite likely being strong predictors.
- The forecasting component has **not been backtested** against real historical
  outcomes — it's a trend extrapolation, not a validated forecasting method yet.
- Data is heavily concentrated in Greater Tunis; under-represented governorates
  have less reliable estimates.

Full write-up, methodology, and all figures are in [`reports/report(46pages).pdf`](reports/report(46pages).pdf).

## Team

Built by a four-person intern team during a six-week data science internship,
academic year 2025/2026.

## License

Add a license of your choice (e.g. MIT) if you want this repo to be reusable
by others.
