#%%
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from config import INDIVIDUAL_DATA_PATH, MODEL_EDUCATION_COL
from model_utils import (
    build_background_regressors,
    load_individual_csv,
    normal_cdf,
    preprocess_individual_data,
    print_model_summary,
)


#%%
@dataclass
class OrderedProbitResult:
    coefficients: pd.Series
    cutpoints: pd.Series
    standard_errors: pd.Series
    z_values: pd.Series
    p_values: pd.Series
    log_likelihood: float
    aic: float
    bic: float
    nobs: int
    converged: bool
    iterations: int
    categories: list
    statsmodels_result: object


#%%
def ordered_probit_regressors(data):
    """Build regressors for completed broad education level."""
    return build_background_regressors(data)


def _normal_p_values(z_values):
    """Two-sided large-sample p-values for maximum-likelihood z-tests."""
    z_values = np.asarray(z_values, dtype=float)
    return 2.0 * (1.0 - normal_cdf(np.abs(z_values)))


def _cutpoints_and_standard_errors(model, result, n_x, cut_names):
    """
    Return actual ordered cutpoints and delta-method standard errors.

    statsmodels estimates transformed threshold parameters. The first threshold
    is estimated directly, while later thresholds are log-increments. This turns
    them into the actual increasing cutpoints used for prediction.
    """
    params = result.params.to_numpy(dtype=float)
    transformed_thresholds = params[n_x:]

    # This returns [-inf, cut_1, cut_2, ..., cut_K, inf].
    cuts = model.transform_threshold_params(params)[1:-1]

    cov = result.cov_params().to_numpy(dtype=float)
    cov_thresholds = cov[n_x:, n_x:]

    n_cuts = len(cuts)
    jacobian = np.zeros((n_cuts, n_cuts))
    jacobian[:, 0] = 1.0

    for cut_idx in range(n_cuts):
        for raw_idx in range(1, n_cuts):
            if raw_idx <= cut_idx:
                jacobian[cut_idx, raw_idx] = np.exp(transformed_thresholds[raw_idx])

    cut_cov = jacobian @ cov_thresholds @ jacobian.T
    cut_se = np.sqrt(np.clip(np.diag(cut_cov), 0.0, np.inf))

    return (
        pd.Series(cuts, index=cut_names, name="cutpoint"),
        pd.Series(cut_se, index=cut_names, name="standard_error"),
    )


#%%
def estimate_study_difficulty_model(data):
    """
    Estimate the study-difficulty ordered probit using statsmodels.

    The model is estimated by maximum likelihood:
        education_level = ordered outcome
        regressors = background controls from build_background_regressors(data)

    Higher values of the latent index imply a higher probability of reaching a
    longer education level.
    """
    data = preprocess_individual_data(data, require_wage=False)
    x_all = ordered_probit_regressors(data)

    model_data = pd.concat([data[[MODEL_EDUCATION_COL]], x_all], axis=1).dropna().copy()
    categories = sorted(model_data[MODEL_EDUCATION_COL].unique().tolist())

    if len(categories) < 3:
        raise ValueError("Ordered probit needs at least three observed education levels.")

    y = model_data[MODEL_EDUCATION_COL]
    x = model_data[x_all.columns].astype(float)

    model = OrderedModel(y, x, distr="probit")
    result = model.fit(method="bfgs", disp=False, maxiter=1000)

    n_x = x.shape[1]
    n_cuts = len(categories) - 1

    coef_names = list(x.columns)
    cut_names = [f"cut_{categories[i]}_{categories[i + 1]}" for i in range(n_cuts)]

    coefficients = pd.Series(result.params.iloc[:n_x].to_numpy(), index=coef_names)
    coefficient_se = pd.Series(result.bse.iloc[:n_x].to_numpy(), index=coef_names)
    coefficient_z = pd.Series(result.tvalues.iloc[:n_x].to_numpy(), index=coef_names)
    coefficient_p = pd.Series(result.pvalues.iloc[:n_x].to_numpy(), index=coef_names)

    cutpoints, cutpoint_se = _cutpoints_and_standard_errors(model, result, n_x, cut_names)
    cutpoint_z = cutpoints / cutpoint_se.replace(0.0, np.nan)
    cutpoint_p = pd.Series(_normal_p_values(cutpoint_z.to_numpy()), index=cut_names)

    standard_errors = pd.concat([coefficient_se, cutpoint_se])
    z_values = pd.concat([coefficient_z, cutpoint_z])
    p_values = pd.concat([coefficient_p, cutpoint_p])

    mle_retvals = getattr(result, "mle_retvals", {})

    return OrderedProbitResult(
        coefficients=coefficients,
        cutpoints=cutpoints,
        standard_errors=standard_errors,
        z_values=z_values,
        p_values=p_values,
        log_likelihood=float(result.llf),
        aic=float(result.aic),
        bic=float(result.bic),
        nobs=int(result.nobs),
        converged=bool(mle_retvals.get("converged", True)),
        iterations=int(mle_retvals.get("iterations", mle_retvals.get("nit", -1))),
        categories=categories,
        statsmodels_result=result,
    )


#%%
def predict_completion_probabilities(data, result):
    """
    Predict P(complete/reach at least each education level) for each individual.

    For ordered education levels, this is the upper-tail probability:
        P(G_i >= g)

    This differs from P(G_i = g). The latter is the probability of ending exactly
    at level g, while this function measures whether the student can reach at
    least level g.
    """
    data = preprocess_individual_data(data, require_wage=False)
    x = build_background_regressors(data)
    x = x.reindex(columns=result.coefficients.index, fill_value=0.0).astype(float).fillna(0.0)

    index = x.to_numpy() @ result.coefficients.to_numpy()
    cuts = result.cutpoints.to_numpy()

    out = pd.DataFrame(index=data.index)

    for level_idx, level in enumerate(result.categories):
        if level_idx == 0:
            # Everyone is assumed able to reach the lowest broad education level.
            p_complete = np.ones(len(data))
        else:
            # To reach level g, latent education must exceed the cutpoint below g.
            lower_cut = cuts[level_idx - 1]
            p_complete = 1.0 - normal_cdf(lower_cut - index)

        out[f"p_complete_{level}"] = p_complete
        out[f"p_fail_{level}"] = 1.0 - p_complete

    return out


def add_study_difficulty_to_long(choice_long, data, result):
    """
    Add study difficulty to the person-alternative data.

    Study difficulty is defined as the probability that student i cannot
    complete/reach education level g:
        study_difficulty_ig = 1 - P(G_i >= g)
    """
    probabilities = predict_completion_probabilities(data, result)
    rows = []

    for _, row in choice_long.iterrows():
        level = int(row[MODEL_EDUCATION_COL])
        failure_probability = probabilities.loc[row["_row_id"], f"p_fail_{level}"]
        rows.append(float(np.clip(failure_probability, 0.0, 1.0)))

    out = choice_long.copy()
    out["study_difficulty"] = rows
    return out


#%%
if __name__ == "__main__":
    data = load_individual_csv(INDIVIDUAL_DATA_PATH)
    study_result = estimate_study_difficulty_model(data)

    print_model_summary("Study-difficulty block: ordered probit", study_result)

    print("\nCutpoints")
    print(study_result.cutpoints)

    print("\nZ-values")
    print(study_result.z_values)

    print("\nPredicted reach probabilities, first rows")
    print(predict_completion_probabilities(data, study_result).head())
