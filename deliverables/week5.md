# Deliverables

## Written / Conceptual

- [ ] Slide deck (presentation.pdf or .pptx/Google Slides link) covering the elements above.
- [ ] Executive summary document (summary.md or .pdf) with:
    - Problem, data sources, KPIs, results.
    - Final model choice with justification.
    - Limitations and next steps.

## Code / Repo Artifacts

- [ ] notebooks/final_results.ipynb:
    - Clean, readable notebook showing final model training, evaluation, and plots.
    - Must produce all metrics/visuals shown in the presentation.
- [ ] results/final/ folder:
    - Saved KPI tables, plots (ROC/PR, calibration, error analysis).
- [ ] artifacts/final_model.pkl (or .joblib):
    - Serialized final pipeline/model.
- [ ] presentation/ folder:
    - Slide deck source file (if not Google Slides) + exported PDF.
    - Executive summary (summary.md or .pdf).
- [ ] Reproducibility check (Makefile or run.sh):
    - One command/script to regenerate final results from raw data → features → models → figures.
