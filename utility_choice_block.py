#%%
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

from config import MODEL_EDUCATION_COL
from model_utils import print_model_summary, safe_log, standardize


#%%
@dataclass
class ConditionalLogitResult:
    coefficients: pd.Series
    standard_errors: pd.Series
    log_likelihood: float
    aic: float
    bic: float
    nobs: int          # number of people / choice situations
    n_choices: int     # number of person-alternative rows used in estimation
    converged: bool
    iterations: int
    statsmodels_result: object


#%%
def prepare_choice_data(choice_long):
    """
    Prepare standardized variables for the conditional-logit utility model.

    Utility specification in this file:
        Utility_ig = alternative_constant_g
                   + beta_income     * standardized log expected lifetime income_ig
                   + beta_difficulty * standardized study difficulty_ig
                   + error_ig

    The dependent variable is `chosen`, coded 0/1 for each person-alternative row.
    The model can have 4 alternatives because each person has 4 rows, with exactly
    one row where chosen = 1.
    """
    data = choice_long.copy()
    data["log_expected_income"] = safe_log(data["expected_lifetime_income"])
    data["z_log_expected_income"] = standardize(data["log_expected_income"])
    data["z_study_difficulty"] = standardize(data["study_difficulty"])

    # Alternative constants
    # Do not add a global intercept in ConditionalLogit.
    alternative_dummies = pd.get_dummies(
        data[MODEL_EDUCATION_COL].astype("category"),
        prefix="alt",
        drop_first=True,
        dtype=float,
    )

    x = pd.concat(
        [
            alternative_dummies,
            data[["z_log_expected_income", "z_study_difficulty"]],
        ],
        axis=1,
    )

    return data, x


def _estimation_sample(data, x):
    model_data = pd.concat(
        [
            data[["_row_id", "chosen", MODEL_EDUCATION_COL]].copy(),
            x.copy(),
        ],
        axis=1,
    ).dropna()

    model_data["chosen"] = model_data["chosen"].astype(int)

    y = model_data["chosen"]
    groups = model_data["_row_id"]
    x_used = model_data[x.columns].astype(float)

    return model_data, y, x_used, groups


#%%
def estimate_utility_choice_model(choice_long):
    """
    Estimate the utility-choice block with statsmodels ConditionalLogit.
    """
    data, x = prepare_choice_data(choice_long)
    model_data, y, x_used, groups = _estimation_sample(data, x)

    model = ConditionalLogit(y, x_used, groups=groups)
    result = model.fit(method="bfgs", disp=False, maxiter=1000)

    coefficients = pd.Series(result.params, index=x_used.columns)
    standard_errors = pd.Series(result.bse, index=x_used.columns)

    mle_retvals = getattr(result, "mle_retvals", {}) or {}
    converged = bool(mle_retvals.get("converged", True))
    iterations = int(mle_retvals.get("iterations", mle_retvals.get("nit", -1)))

    n_params = len(coefficients)
    n_groups = int(model_data["_row_id"].nunique())
    log_likelihood = float(result.llf)

    return ConditionalLogitResult(
        coefficients=coefficients,
        standard_errors=standard_errors,
        log_likelihood=log_likelihood,
        aic=float(2 * n_params - 2 * log_likelihood),
        bic=float(np.log(max(n_groups, 1)) * n_params - 2 * log_likelihood),
        nobs=n_groups,
        n_choices=int(len(model_data)),
        converged=converged,
        iterations=iterations,
        statsmodels_result=result,
    )


#%%
def predict_choice_probabilities(choice_long, result):
    """Return predicted choice probabilities for each education alternative."""
    data, x = prepare_choice_data(choice_long)
    x = x.reindex(columns=result.coefficients.index, fill_value=0.0).astype(float)

    data["utility_index"] = x.to_numpy() @ result.coefficients.to_numpy()

    probabilities = []
    for _, group in data.groupby("_row_id", sort=False):
        exp_utility = np.exp(group["utility_index"] - group["utility_index"].max())
        probabilities.extend(exp_utility / exp_utility.sum())

    data["choice_probability"] = probabilities
    return data


#%%
# Test run for this block only.
if __name__ == "__main__":
    test_choice_long = pd.DataFrame(
        [
            {
                "_row_id": person,
                MODEL_EDUCATION_COL: education,
                "chosen": int(education == chosen),
                "expected_lifetime_income": 1_000_000 + 120_000 * education + 10_000 * person,
                "unemployment_rate": 0.07 - 0.005 * education,
                "study_difficulty": 0.4 + 0.25 * education,
            }
            for person, chosen in enumerate([1, 2, 3, 4, 2, 3, 4, 1, 2, 4, 3, 1])
            for education in [1, 2, 3, 4]
        ]
    )

    utility_result = estimate_utility_choice_model(test_choice_long)
    print_model_summary("Utility-choice block: statsmodels conditional logit, alternative constants only", utility_result)
    print("\nPredicted choice probabilities, first rows")
    print(predict_choice_probabilities(test_choice_long, utility_result).head(12))
