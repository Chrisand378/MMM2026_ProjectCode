# Project Files

This project estimates education-choice models using Danish individual-level data.

## Files

- `main.py` runs the full model pipeline.
- `config.py` defines paths, variable names, and model assumptions.
- `model_utils.py` contains shared helper functions for loading data, preprocessing, OLS estimation, and summaries.
- `DataPreparations.py` prepares and cleans the raw dataset.
- `descriptive_statistics.py` prints descriptive statistics for the prepared data.
- `OLS_Print.py` runs and prints OLS wage-regression results used for creating tables.
- `income_block_EducationReg.py` estimates education-specific income models.
- `income_block_SingleReg.py` estimates a pooled income model.
- `study_difficulty_block.py` estimates the study-difficulty model using ordered probit.
- `utility_choice_block.py` estimates the education-choice model using conditional logit.
- `IncomePlotsDistribution.py` creates income-profile plots.
- `Simulations.py` runs simulations based on predicted education-choice probabilities.
