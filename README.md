# Project Files

This project estimates education-choice models using Danish individual-level data.

## Files

- `main.py` runs the full model pipeline.
- `config.py` defines paths, variable names, and model assumptions.
- `model_utils.py` contains shared helper functions for loading data, preprocessing, OLS estimation, and summaries.
- `DataPreparations.py` prepares and cleans the raw dataset.
- `descriptive_statistics.py` prints descriptive statistics for the prepared data.
- `OLS_Print.py` runs and prints OLS wage-regression results.
- `income_block_EducationReg_no_smearing_benefits.py` estimates education-specific income models.
- `income_block_SingleReg_no_smearing_benefits.py` estimates a pooled income model.
- `study_difficulty_block_statsmodels.py` estimates the study-difficulty model using ordered probit.
- `utility_choice_block_statsmodels_alt_constants_only.py` estimates the education-choice model using conditional logit.
- `IncomePlotsDistribution_no_titles_thousandDKK_no_legend_markers.py` creates income-profile plots.
- `Simulations_fixed.py` runs simulations based on predicted education-choice probabilities.
