# Checkpoint 3 Guide: Model Evaluation Plan
## Deliverables
### Jess? 
- [ ] **(overview done together during meeting)** A clear evaluation plan document (evaluation_plan.md) describing:
    - The unit of analysis and how it shapes data splits.
    - The chosen split strategy (time, group, geographic, or custom) and why it reflects deployment.
    - Planned stress tests (outlier sensitivity, distribution shift checks, adversarial leakage tests).
- [ ] Stress test scripts in src/eval/:
    - Functions to re-evaluate models under perturbed conditions (dropped features, shuffled targets, specific cohorts).
    - Produces log output or plots stored under results/stress_tests/.
### Doyee? 
- [ ] src/splits/ directory:
    - [ ] Custom splitter implementations (if needed), with docstrings explaining intended generalization scenario.
    - [ ] (Optional) Unit tests ensuring that no overlap exists between train/test sets for the specified grouping.
### Han?
- [ ] notebooks/metric_evaluation.ipynb:
    - Implements evaluation with multiple metrics (`sklearn.metrics` functions, calibration plots, cost-sensitive metrics if relevant).
    - Includes confusion matrices, classification reports, ROC/PR curves (for classification) or error distribution plots (for regression).
    - Stores results in a tabular form (e.g., CSV in results/).
### All?
- [ ] results/ folder:
    - Intermediate CSV/JSON tables of cross-validation scores, metric breakdowns, calibration curves.
    - Each file must specify the split strategy and date generated in its filename or metadata.
- **Note**: At the time of this checkpoint, you only need to have run your baseline models through your evaluation framework.  Further modeling work will be done later.
