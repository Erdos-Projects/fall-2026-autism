A data audit summary: key distributional facts, missingness, correlations.

A list of dropped features with justification (low information, leakage, redundancy, irrelevance).

A list of engineered features with rationale (domain knowledge, lags, interactions, transformations).


## Deliverables
### Jess:
notebooks/eda.ipynb:
Visualizations and descriptive stats.
Missingness analysis.
Correlation heatmaps, scatterplots, outlier checks.
Exploratory lag analysis (ACF/PACF, lag vs. target plots).

notebooks/feature_selection.ipynb:
Shows feature elimination decisions (variance threshold, correlation pruning, leakage checks).
Logs results in a table (results/feature_selection.csv).

dataframe for demographics

### Doyee:
notebooks/pipeline_demo.ipynb:
Demonstrates the pipeline fitting/transforming data.

### both
src/features/transformers.py:
Custom transformers implementing engineered features (e.g., lag features, ratios, domain-driven encodings).

src/features/preprocessing.py:
Main preprocessing pipeline that drops unused features, integrates engineered features, and can be reused in modeling.

results/eda/ folder:
Key plots saved as .png or .html.

Updated schema (schema.json or schema.yaml):
Defines available features, their types, and any transformations applied.

(Optional) Tests in tests/test_pipeline.py:
Verify engineered features are generated correctly.
Ensure pipeline respects train/test splits (no future leakage).
