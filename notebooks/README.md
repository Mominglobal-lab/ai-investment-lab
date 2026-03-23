# Notebooks

This folder is for research, validation, and idea testing.

Guidelines:

- Reuse code from `ai_models/`, `simulation/`, `reports/`, and `data_pipeline/`.
- Do not duplicate production logic in notebooks.
- Promote stable notebook logic into modules and add tests in `tests/`.
- Keep data access pointed at the existing cached parquet/json artifacts when possible.

Suggested workflow:

1. Explore an idea in a notebook.
2. Validate the output visually and with edge cases.
3. Move stable logic into production modules.
4. Add or update automated tests.

Starter notebooks:

- `01_feature_exploration.ipynb`
- `02_model_validation.ipynb`
- `03_portfolio_simulation_analysis.ipynb`
- `04_portfolio_profile_experiments.ipynb`
- `05_ml_stock_ranking.ipynb`
- `06_portfolio_recommender_ml.ipynb`
- `07_llm_portfolio_brief.ipynb`
- `08_news_earnings_impact_model.ipynb`
- `09_anomaly_detection.ipynb`

Kernel setup:

- Use the project virtual environment if available.
- From the repo root on Windows:

```powershell
.\.venv\Scripts\python -m pip install jupyter ipykernel
.\.venv\Scripts\python -m ipykernel install --user --name ai-investment-lab --display-name "ai-investment-lab"
```

Then choose the `ai-investment-lab` kernel in Jupyter.
