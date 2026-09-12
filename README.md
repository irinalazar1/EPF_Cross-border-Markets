# EPF Expert Review (RLHF)

A Streamlit dashboard for human-in-the-loop review of day-ahead electricity
price forecasts for the Belgian market. Domain experts adjust a neural
network's forecast by dragging the curve on a chart, rate their confidence,
and submit. Once the delivery day settles, admins compare each expert's
adjustment against the realized price (MAE) and aggregate results across
experts. The research question: does human review measurably improve a
forecasting model's output — the same question RLHF asks for language
models, applied here to electricity prices.

## Origin

Forked from Margarida Mascarenhas's
[**EPF_Cross-border-Markets**](https://github.com/margaridamascarenhas/EPF_Cross-border-Markets)
to bootstrap the initial app, using the Belgian data bundled in that repo
to mimic her DNN forecast results as a starting point. That local-data
dependency was later removed in favor of pulling data directly and live
from a separate repo,
[`DAM_Forecast_V4`](https://github.com/margaridamascarenhas/DAM_Forecast_V4)
— the current and only live data source (see Architecture below). The fork
also vendored [epftoolbox](https://github.com/jeslago/epftoolbox)
(Lago et al., *Applied Energy* 2021), the library that originally produced
the DNN/LEAR forecasts; that code has since been deleted from this project
too, since the review app never imported it directly. Cite that paper if
any result from that lineage reaches a publication.

## Architecture

- **`app.py`** — the Streamlit app: auth, theming, all five pages.
- **`db.py`** — SQLite persistence (`epf_dashboard.db`, auto-created).
- **`draggable_curve/`** — custom React/TypeScript component (built with
  Vite) rendering the forecast chart as one drag-to-edit widget: forecast
  line, uncertainty band, flagged points, all in one view.
- **Data** — fetched directly from GitHub's raw-content URLs at runtime and
  parsed straight into memory (`requests` → `io.StringIO` → `pandas`, cached
  30 min via `st.cache_data`). Nothing is ever saved to a local folder in
  this repo:
  - `raw.githubusercontent.com/margaridamascarenhas/DAM_Forecast_V4/main/Forecast/DNN_forecasts_10AM.csv`
  - `raw.githubusercontent.com/margaridamascarenhas/DAM_Forecast_V4/main/datasets/Data_BE_UTC.csv`

## Project structure

```
.
├── app.py
├── db.py
├── requirements.txt
├── .env
├── epf_dashboard.db            # created automatically on first run
└── draggable_curve/
    ├── __init__.py
    └── frontend/
        ├── src/DraggableCurve.tsx
        ├── dist/                # built output — what actually runs
        └── package.json
```

## Setup

```bash
# 1. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Environment variables
cp .env.example .env   # then fill in GITHUB_TOKEN

# 3. Build the drag-to-edit component (one-time; only needs Node for this step)
cd draggable_curve/frontend && npm install && npm run build && cd ../..

# 4. Run
streamlit run app.py
```

The database and all tables are created automatically on first run.
Self-registration only creates `expert` accounts — insert an `admin` row
directly into `users` (bcrypt-hash the password first) to create one.

## Roles & pages

- **expert** — Review & Adjust and Deterministic Forecast Analysis only;
  can only view/submit their own work. Submissions are final.
- **admin** — everything above, plus Reveal & Evaluate, Expert Scoreboard,
  and Survey Results; can view but never submit on an expert's behalf.

| Page | Purpose |
|---|---|
| Review & Adjust | View forecast, drag to adjust, rate confidence, submit, answer reflection survey. |
| Deterministic Forecast Analysis | DNN forecast vs. actual, last 14 days. |
| Reveal & Evaluate (admin) | MAE for one expert's submission once the price settles. |
| Expert Scoreboard (admin) | Aggregated MAE improvement, win rate, per expert. |
| Survey Results (admin) | Reflection-survey data + experience profiles; CSV export. |

## Data model (SQLite)

| Table | Purpose |
|---|---|
| `users` | Username, email, bcrypt hash, role. |
| `feedback` | One row per 15-min slot per submission: forecast, adjusted, flagged, confidence. |
| `onboarding_status` | Research-disclaimer consent + tutorial completion, gates all access. |
| `user_profile` | Self-reported EPF experience, asked once per user. |
| `submission_survey` | Reflection-survey ratings per submission, joinable against `feedback` on `(username, forecast_date)`. |

## The uncertainty band

The chart's 80% band isn't a separate model — it's a single margin,
conformal-calibrated from the DNN forecast's own settled residuals via
Adaptive Conformal Inference (`get_aci_margin()`). It self-adjusts: recent
misses widen it, recent hits relax it, no manual recalibration needed.

## Known limitations

- A few CSS rules (toggle/radio "on" colors) target Streamlit's
  auto-generated class names, not stable attributes — Streamlit doesn't
  expose one for these. May need re-verifying after a Streamlit upgrade
  (right-click → Inspect on the live element).
- A skipped reflection survey produces zero rows, not a marked "skipped"
  row — response rate isn't directly computable from the data as-is.
- French demand isn't shown as context: it's on a different scale from
  Belgium's, so it wasn't useful for judging the Belgian forecast.
