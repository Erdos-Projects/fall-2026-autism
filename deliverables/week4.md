# Deliverables

## Written / Conceptual

A short document (modeling_plan.md) describing:
  - Which model families were tried and why.
  - How they align with project goals.
  - Justification for the final chosen model, tied explicitly to KPIs and evaluation strategy.
  - Notes on what didn’t work and why those approaches were discarded.

## Code / Repo Artifacts

- [] notebooks/modeling_baselines.ipynb:
  - Implements trivial and simple baselines (Dummy, Linear, Logistic, basic trees).
  - Reports CV performance with chosen KPIs.
- [] notebooks/modeling_experiments.ipynb:
  - Trains more complex models (ensembles, regularized, etc.).
  - Includes performance tables/plots across folds.
  - Documents failed experiments and iteration decisions.
- [] src/models/:
  - Python scripts defining reusable training functions and model wrappers.
    - [] src/models/tune.py:
      - Hyperparameter search logic, tied to the correct CV splitter.
    - [] results/model_comparison.csv:
      - Tabular record of model family, hyperparameters, and KPI scores.
    - [] results/interpretability/:
      - Feature importance plots, coefficient tables, or SHAP visualizations.
      - Serialized models in artifacts/ (via joblib.dump), with matching environment spec so they can be reloaded.
    - [] Tests in tests/test_models.py:
      - Ensure training scripts run end-to-end, models fit without errors, and CV returns consistent shapes.
