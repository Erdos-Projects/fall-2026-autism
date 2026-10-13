- [x] A data audit summary: key distributional facts, missingness, correlations.
    - Missing data information can be found in `results/eda/missingness.txt`
    - Distributions are all unremarkable, except for the autism quotient distributions where there are ~5 outliers for control and autism groups.
    - There is a lot of missing data for IQ tests. The only complete tests are the WASI, WAIS_III, and DAS_II_SA.
    - The mean IQ for Autism (T1) is significantly higher for all three IQ types; however, there are also significantly fewer samples. 

- [x] A list of dropped features with justification (low information, leakage, redundancy, irrelevance).
    - Remove scans of those on stimulants at time of scan. Reason: this can influence brain activity at time of scan.
    - Remove outliers for autism quotient distributions for control and autism groups. Reason: this could indicate misdiagnosis. 
    - Remove data listed as -9999. Reason: this is a placeholder which lies outside of the range specified in the legend provided. 
    - Potentially drop:
        - If we want to use IQ data, only use scans from WASI, WAIS_III, and DAS_II_SA. Reason: the other IQ test types have incomplete data.
        - Could remove scans of those with an "Unspecified" diagnosis ("DSM_IV_TR" >= 3). Reason: this could indicate misdiagnosis.

- [x] A list of engineered features with rationale (domain knowledge, lags, interactions, transformations).
    - Demographics data ranges are listed in `data/ABIDE_LEGEND_V1.02.pdf`.
    - Domains for grey and white matter listed in `results/feature_selection.csv`.


## Deliverables
> Jess:
- [x] notebooks/eda.ipynb:
    - Visualizations and descriptive stats.
    - Missingness analysis.
    - Correlation heatmaps, scatterplots, outlier checks.
    - Exploratory lag analysis (ACF/PACF, lag vs. target plots).

- [x] results/eda/ folder:
    - Key plots saved as .png or .html.

- [x] dataframe for demographics

### Doyee:
- [ ] notebooks/pipeline_demo.ipynb:
    - Demonstrates the pipeline fitting/transforming data.

### both
- [ ] notebooks/feature_selection.ipynb:
    - Shows feature elimination decisions (variance threshold, correlation pruning, leakage checks).
    - Logs results in a table (results/feature_selection.csv).

- [ ] src/features/transformers.py:
    - Custom transformers implementing engineered features (e.g., lag features, ratios, domain-driven encodings).

- [x] src/features/preprocessing.py:
    - Main preprocessing pipeline that drops unused features, integrates engineered features, and can be reused in modeling.    

- [ ] Updated schema (schema.json or schema.yaml):
    - Defines available features, their types, and any transformations applied.

- [ ] (Optional) Tests in tests/test_pipeline.py:
    - Verify engineered features are generated correctly.
    - Ensure pipeline respects train/test splits (no future leakage).
