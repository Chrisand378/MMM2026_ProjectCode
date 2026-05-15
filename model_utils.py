#%%
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries

from config import (
    AGE_COL,
    CATEGORICAL_BACKGROUND_COLS,
    EDUCATION_COL,
    EXPERIENCE_COL,
    FORBIDDEN_MODEL_COLS,
    ID_COL,
    LOG_BACKGROUND_COLS,
    LOG_BACKGROUND_PREFIX,
    LOG_WAGE_COL,
    MODEL_EDUCATION_COL,
    NUMERIC_BACKGROUND_COLS,
    TRANSFORMED_EDUCATION_COL,
    WAGE_COL,
)


#%%
def normal_cdf(x):
    """Standard normal CDF using only Python/numpy."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def add_constant(x, name="const"):
    x = pd.DataFrame(x).copy()
    if name not in x.columns:
        x.insert(0, name, 1.0)
    return x


def standardize(series):
    values = pd.Series(series, dtype=float)
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return values * 0.0
    return (values - values.mean()) / std


def safe_log(values, eps=1e-8):
    return np.log(np.clip(np.asarray(values, dtype=float), eps, None))


#%%
def resolve_education_col(data):
    """Use f_udd_t when available; otherwise use the regular f_udd column."""
    if TRANSFORMED_EDUCATION_COL in data.columns:
        return TRANSFORMED_EDUCATION_COL
    if EDUCATION_COL in data.columns:
        return EDUCATION_COL
    raise ValueError(
        f"Need either '{TRANSFORMED_EDUCATION_COL}' or '{EDUCATION_COL}' in the CSV."
    )


def require_columns(data, columns, context="dataset"):
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def _coerce_education_values(values):
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int)
    return values


def preprocess_individual_data(data, require_wage=False):
    """
    Apply project-wide dataset rules.

    The raw education column is copied to MODEL_EDUCATION_COL so downstream code
    can use one name whether the source dataset uses f_udd or f_udd_t.
    """
    data = data.copy()

    if "i_udd" in data.columns:
        data = data.drop(columns=["i_udd"])

    education_col = resolve_education_col(data)
    data[MODEL_EDUCATION_COL] = _coerce_education_values(data[education_col])

    if require_wage:
        require_columns(data, [WAGE_COL], "income data")
        wage = pd.to_numeric(data[WAGE_COL], errors="coerce")
        data[LOG_WAGE_COL] = np.nan
        data.loc[wage > 0, LOG_WAGE_COL] = np.log(wage.loc[wage > 0])

    for col in [AGE_COL, EXPERIENCE_COL] + NUMERIC_BACKGROUND_COLS + LOG_BACKGROUND_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in LOG_BACKGROUND_COLS:
        if col in data.columns:
            data[f"{LOG_BACKGROUND_PREFIX}{col}"] = np.log1p(data[col].clip(lower=0))

    return data


def normalize_labor_market_entry(data):
    """Standardize the labor-market-entry education column."""
    data = data.copy()
    candidates = [MODEL_EDUCATION_COL, EDUCATION_COL, TRANSFORMED_EDUCATION_COL, "F_udd"]
    education_col = next((col for col in candidates if col in data.columns), None)
    if education_col is None:
        raise ValueError("Labor-market-entry data needs an education column.")
    if "LaborMarketEntry" not in data.columns:
        raise ValueError("Labor-market-entry data needs 'LaborMarketEntry'.")
    data[MODEL_EDUCATION_COL] = _coerce_education_values(data[education_col])
    return data[[MODEL_EDUCATION_COL, "LaborMarketEntry"]]


def build_background_regressors(data):
    """Build background controls with project-specific encoding."""
    require_columns(
        data,
        CATEGORICAL_BACKGROUND_COLS + NUMERIC_BACKGROUND_COLS + LOG_BACKGROUND_COLS,
        "background controls",
    )
    parts = []

    numeric_cols = [col for col in NUMERIC_BACKGROUND_COLS if col in data.columns]
    log_cols = [
        f"{LOG_BACKGROUND_PREFIX}{col}"
        for col in LOG_BACKGROUND_COLS
        if f"{LOG_BACKGROUND_PREFIX}{col}" in data.columns
    ]
    if numeric_cols or log_cols:
        parts.append(data[numeric_cols + log_cols].astype(float))

    for col in CATEGORICAL_BACKGROUND_COLS:
        if col in data.columns:
            dummies = pd.get_dummies(
                data[col].astype("category"),
                prefix=col,
                drop_first=True,
                dtype=float,
            )
            parts.append(dummies)

    if not parts:
        raise ValueError("No configured background columns were found in the CSV.")

    x = pd.concat(parts, axis=1)
    assert_no_forbidden_regressors(x.columns)
    return x


def assert_no_forbidden_regressors(columns):
    """Fail fast if an excluded variable slips into a model matrix."""
    forbidden = set(FORBIDDEN_MODEL_COLS)
    used = []
    for col in columns:
        base = str(col).removeprefix(LOG_BACKGROUND_PREFIX)
        if str(col) in forbidden or base in forbidden:
            used.append(str(col))
    if used:
        raise ValueError(f"Forbidden variables used as regressors: {used}")


#%%
def load_named_excel_table(path, table_name):
    """Read an Excel named table into a pandas DataFrame."""
    workbook = load_workbook(path, data_only=True)
    for worksheet in workbook.worksheets:
        if table_name not in worksheet.tables:
            continue

        table = worksheet.tables[table_name]
        min_col, min_row, max_col, max_row = range_boundaries(table.ref)
        rows = list(
            worksheet.iter_rows(
                min_row=min_row,
                max_row=max_row,
                min_col=min_col,
                max_col=max_col,
                values_only=True,
            )
        )
        return pd.DataFrame(rows[1:], columns=rows[0])

    raise ValueError(f"Could not find table '{table_name}' in {path}")


def load_individual_csv(path):
    """Read the individual-level CSV."""
    return pd.read_csv(path)


#%%
def _two_sided_p_values(t_values, df_resid):
    """
    Two-sided p-values for coefficient tests.

    Uses scipy's Student-t distribution when available. If scipy is not
    installed, falls back to the large-sample normal approximation.
    """
    t_values = np.asarray(t_values, dtype=float)
    with np.errstate(invalid="ignore"):
        abs_t = np.abs(t_values)

    try:
        from scipy import stats  # optional dependency

        return 2.0 * stats.t.sf(abs_t, df=max(int(df_resid), 1))
    except Exception:
        return 2.0 * (1.0 - normal_cdf(abs_t))


@dataclass
class OLSResult:
    coefficients: pd.Series
    standard_errors: pd.Series
    t_values: pd.Series
    p_values: pd.Series
    residual_std: float
    r_squared: float
    log_likelihood: float
    aic: float
    bic: float
    nobs: int
    n_params: int
    df_resid: int
    fitted: pd.Series
    residuals: pd.Series


def fit_ols(y, x):
    """OLS with classical standard errors, t-values, p-values, and Gaussian log-likelihood."""
    y = pd.Series(y, dtype=float).reset_index(drop=True)
    x = add_constant(x).astype(float).reset_index(drop=True)

    keep = y.notna() & x.notna().all(axis=1)
    y = y.loc[keep].reset_index(drop=True)
    x = x.loc[keep].reset_index(drop=True)

    x_mat = x.to_numpy()
    y_vec = y.to_numpy()
    nobs, n_params = x_mat.shape
    df_resid = max(nobs - n_params, 1)

    beta = np.linalg.pinv(x_mat.T @ x_mat) @ x_mat.T @ y_vec
    fitted = x_mat @ beta
    resid = y_vec - fitted

    sse = float(resid.T @ resid)
    tss = float(((y_vec - y_vec.mean()) ** 2).sum())
    sigma2 = sse / df_resid
    cov = sigma2 * np.linalg.pinv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(cov))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = beta / se
    p_values = _two_sided_p_values(t_values, df_resid)

    ll_sigma2 = max(sse / max(nobs, 1), 1e-12)
    log_likelihood = float(-0.5 * nobs * (math.log(2 * math.pi) + math.log(ll_sigma2) + 1))

    return OLSResult(
        coefficients=pd.Series(beta, index=x.columns),
        standard_errors=pd.Series(se, index=x.columns),
        t_values=pd.Series(t_values, index=x.columns),
        p_values=pd.Series(p_values, index=x.columns),
        residual_std=float(math.sqrt(sigma2)),
        r_squared=float(1 - sse / tss) if tss > 0 else np.nan,
        log_likelihood=log_likelihood,
        aic=float(2 * n_params - 2 * log_likelihood),
        bic=float(math.log(max(nobs, 1)) * n_params - 2 * log_likelihood),
        nobs=int(nobs),
        n_params=int(n_params),
        df_resid=int(df_resid),
        fitted=pd.Series(fitted),
        residuals=pd.Series(resid),
    )


#%%
def numeric_gradient(func, params, step=1e-5):
    grad = np.zeros_like(params, dtype=float)
    for idx in range(len(params)):
        delta = np.zeros_like(params, dtype=float)
        delta[idx] = step
        grad[idx] = (func(params + delta) - func(params - delta)) / (2 * step)
    return grad


def numeric_hessian(func, params, step=1e-4):
    params = np.asarray(params, dtype=float)
    n_params = len(params)
    hessian = np.zeros((n_params, n_params), dtype=float)
    for i in range(n_params):
        for j in range(n_params):
            ei = np.zeros(n_params)
            ej = np.zeros(n_params)
            ei[i] = step
            ej[j] = step
            hessian[i, j] = (
                func(params + ei + ej)
                - func(params + ei - ej)
                - func(params - ei + ej)
                + func(params - ei - ej)
            ) / (4 * step * step)
    return hessian


@dataclass
class OptimizerResult:
    params: np.ndarray
    inverse_hessian: np.ndarray
    objective: float
    converged: bool
    iterations: int


def minimize_bfgs(func, start_params, max_iter=500, tol=1e-6):
    """Small BFGS optimizer so the project code only depends on numpy/pandas."""
    params = np.asarray(start_params, dtype=float)
    n_params = len(params)
    inv_hessian = np.eye(n_params)
    value = float(func(params))

    for iteration in range(1, max_iter + 1):
        grad = numeric_gradient(func, params)
        if np.linalg.norm(grad, ord=np.inf) < tol:
            return OptimizerResult(params, inv_hessian, value, True, iteration)

        direction = -inv_hessian @ grad
        step = 1.0
        accepted = False

        for _ in range(30):
            candidate = params + step * direction
            candidate_value = float(func(candidate))
            if np.isfinite(candidate_value) and candidate_value < value:
                accepted = True
                break
            step *= 0.5

        if not accepted:
            warnings.warn("BFGS line search stopped before finding an improving step.")
            return OptimizerResult(params, inv_hessian, value, False, iteration)

        new_grad = numeric_gradient(func, candidate)
        s = candidate - params
        y = new_grad - grad
        ys = float(y @ s)

        if ys > 1e-12:
            rho = 1.0 / ys
            ident = np.eye(n_params)
            inv_hessian = (ident - rho * np.outer(s, y)) @ inv_hessian @ (
                ident - rho * np.outer(y, s)
            ) + rho * np.outer(s, s)

        params = candidate
        value = candidate_value

        if np.linalg.norm(s, ord=np.inf) < tol:
            return OptimizerResult(params, inv_hessian, value, True, iteration)

    return OptimizerResult(params, inv_hessian, value, False, max_iter)


#%%
def print_model_summary(name, result):
    """Compact console summary for interactive runs."""
    print(f"\n{name}")
    print("=" * len(name))
    if hasattr(result, "coefficients"):
        print("Coefficients")
        print(result.coefficients)
    if hasattr(result, "standard_errors"):
        print("\nStandard errors")
        print(result.standard_errors)
    if hasattr(result, "p_values"):
        print("\nP-values")
        print(result.p_values)
    for attr in ["nobs", "r_squared", "residual_std", "log_likelihood", "aic", "bic"]:
        if hasattr(result, attr):
            print(f"{attr}: {getattr(result, attr)}")
