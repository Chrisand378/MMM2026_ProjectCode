#%%
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    AGE_COL,
    ANNUAL_HOURS,
    CATEGORICAL_BACKGROUND_COLS,
    DISCOUNT_RATE,
    EXPERIENCE_COL,
    INDIVIDUAL_DATA_PATH,
    LABOR_MARKET_ENTRY_PATH,
    LABOR_MARKET_ENTRY_TABLE,
    LOG_WAGE_COL,
    MODEL_EDUCATION_COL,
    RETIREMENT_AGE,
    UNEMPLOYMENT_RATES,
)
from model_utils import (
    assert_no_forbidden_regressors,
    build_background_regressors,
    fit_ols,
    load_individual_csv,
    load_named_excel_table,
    normalize_labor_market_entry,
    normal_cdf,
    preprocess_individual_data,
    print_model_summary,
    require_columns,
)


# Unemployment benefit replacement rate used in the annual income formula.
UNEMPLOYMENT_BENEFIT_REPLACEMENT_RATE = 0.25


#%%
def _two_sided_p_values(t_values, df_resid):
    """Two-sided p-values for OLS coefficient tests."""
    t_values = np.asarray(t_values, dtype=float)
    abs_t = np.abs(t_values)

    try:
        from scipy import stats  # optional dependency

        return 2.0 * stats.t.sf(abs_t, df=max(int(df_resid), 1))
    except Exception:
        # Large-sample fallback if scipy is not available.
        return 2.0 * (1.0 - normal_cdf(abs_t))


@dataclass
class SingleIncomeResult:
    """
    Wrapper for the pooled income OLS.

    The main model estimates one pooled OLS with education-level dummies.
    This wrapper keeps the output interface close to the earlier income block:
    coefficient_table(), p_values, and education-level observation counts are
    available for result export and later blocks.
    """

    ols_result: object
    coefficients: pd.Series
    standard_errors: pd.Series
    t_values: pd.Series
    p_values: pd.Series
    residual_std: float
    r_squared: float
    log_likelihood: float
    aic: float
    bic: float
    nobs_total: int
    n_params: int
    df_resid: int
    fitted: pd.Series
    residuals: pd.Series
    nobs: pd.Series

    def coefficient_table(self):
        """Return one tidy coefficient table for the pooled OLS."""
        return pd.DataFrame(
            {
                "model": "pooled_ols",
                "variable": self.coefficients.index,
                "coefficient": self.coefficients.values,
                "standard_error": self.standard_errors.reindex(self.coefficients.index).values,
                "t_value": self.t_values.reindex(self.coefficients.index).values,
                "p_value": self.p_values.reindex(self.coefficients.index).values,
                "nobs": self.nobs_total,
                "r_squared": self.r_squared,
                "residual_std": self.residual_std,
            }
        )

    def summary_table(self):
        """Return education-level summary statistics used by export scripts."""
        levels = self.nobs.index
        return pd.DataFrame(
            {
                "education_level": levels,
                "nobs": self.nobs.reindex(levels).values,
            }
        )

    def p_value_dict(self):
        """Return p-values as a regular dictionary."""
        return self.p_values.to_dict()


#%%
def prepare_income_data(data):
    """
    Prepare the wage equation data.

    The dependent variable is log_timelon, created from timelon.
    """
    data = preprocess_individual_data(data, require_wage=True)
    require_columns(
        data,
        [MODEL_EDUCATION_COL, AGE_COL, EXPERIENCE_COL, LOG_WAGE_COL],
        "income data",
    )

    if AGE_COL in data.columns:
        data["age_sq"] = data[AGE_COL] ** 2
    if EXPERIENCE_COL in data.columns:
        data["experience_sq"] = data[EXPERIENCE_COL] ** 2

    return data


def income_regressors(data):
    """
    Build regressors for the pooled log-wage equation.

    Education level is included as a categorical variable via dummies.
    The first observed education category is the omitted reference group.
    """
    profile_cols = [AGE_COL, "age_sq", EXPERIENCE_COL, "experience_sq"]
    background = build_background_regressors(data)

    education_dummies = pd.get_dummies(
        data[MODEL_EDUCATION_COL].astype("category"),
        prefix="edu",
        drop_first=True,
        dtype=float,
    )

    x = pd.concat([data[profile_cols].astype(float), background, education_dummies], axis=1)
    assert_no_forbidden_regressors(x.columns)
    return x


#%%
def estimate_income_model(data):
    """
    Estimate the income block with one pooled OLS.

    Model: log(timelon) = education dummies + age profile
    + experience profile + background variables + error.

    The returned object keeps the earlier export interface:
    - coefficient_table()
    - coefficients / standard_errors / t_values / p_values
    - nobs by education level
    """
    data = prepare_income_data(data)
    x = income_regressors(data)
    y = data[LOG_WAGE_COL]

    # Same sample rule as fit_ols, retained here to compute education-level
    # counts from the estimation sample.
    keep = y.notna() & x.notna().all(axis=1)
    used_education = data.loc[keep, MODEL_EDUCATION_COL].reset_index(drop=True)

    ols = fit_ols(y, x)

    coefficients = ols.coefficients
    standard_errors = ols.standard_errors
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = coefficients / standard_errors.replace(0.0, np.nan)

    n_params = len(coefficients)
    df_resid = max(int(ols.nobs) - n_params, 1)
    p_values = pd.Series(
        _two_sided_p_values(t_values.to_numpy(), df_resid),
        index=coefficients.index,
    )

    nobs_by_education = used_education.value_counts().sort_index().astype(int)

    return SingleIncomeResult(
        ols_result=ols,
        coefficients=coefficients,
        standard_errors=standard_errors,
        t_values=pd.Series(t_values, index=coefficients.index),
        p_values=p_values,
        residual_std=ols.residual_std,
        r_squared=ols.r_squared,
        log_likelihood=ols.log_likelihood,
        aic=ols.aic,
        bic=ols.bic,
        nobs_total=int(ols.nobs),
        n_params=n_params,
        df_resid=df_resid,
        fitted=ols.fitted,
        residuals=ols.residuals,
        nobs=nobs_by_education,
    )


#%%
def predict_log_wage_for_alternative(row, education_level, income_result):
    """Predict log wage for one person and one broad education level."""
    values = {}

    for name in income_result.coefficients.index:
        if name == "const":
            values[name] = 1.0
        elif name.startswith("edu_"):
            dummy_level = name.replace("edu_", "", 1)
            try:
                values[name] = 1.0 if float(dummy_level) == float(education_level) else 0.0
            except ValueError:
                values[name] = 1.0 if str(dummy_level) == str(education_level) else 0.0
        elif any(name.startswith(f"{col}_") for col in CATEGORICAL_BACKGROUND_COLS):
            source_col = next(col for col in CATEGORICAL_BACKGROUND_COLS if name.startswith(f"{col}_"))
            category = name.replace(f"{source_col}_", "", 1)
            values[name] = 1.0 if str(row.get(source_col)) == category else 0.0
        elif name == "age_sq":
            values[name] = row.get(AGE_COL, np.nan) ** 2
        elif name == "experience_sq":
            values[name] = row.get(EXPERIENCE_COL, np.nan) ** 2
        else:
            values[name] = row.get(name, 0.0)

    x = pd.Series(values).reindex(income_result.coefficients.index).fillna(0.0)
    return float(x @ income_result.coefficients)


def predict_expected_hourly_wage_for_alternative(row, education_level, income_result):
    """Predict hourly wage in levels as exp(predicted log wage)."""
    log_wage = predict_log_wage_for_alternative(row, education_level, income_result)
    return float(np.exp(log_wage))


def discount_origin_age(labor_market_entry):
    """
    Return the common discount origin age.

    The discount origin is the labor-market entry age for education level 1
    if education level 1 is available. Otherwise, it is the minimum entry age
    in the labor-market-entry table.
    """
    entry = pd.Series(labor_market_entry).copy()
    try:
        entry.index = entry.index.astype(int)
    except Exception:
        pass

    if 1 in entry.index:
        return int(entry.loc[1])
    return int(entry.min())


def discount_factor_for_age(age, discount_start_age):
    """Discount factor for an annual income flow at a given age."""
    return float((1.0 + DISCOUNT_RATE) ** (int(age) - int(discount_start_age)))


def annual_income_with_unemployment_benefits(expected_hourly_wage, education_level):
    """
    Convert predicted hourly wage to annual income including unemployment benefits.

    Formula:
        Y_hat_iga = H * (1 - u_g) * w_hat_iga
                    + H * u_g * b * w_hat_iga

    where b is the unemployment benefit replacement rate.
    """
    unemployment = float(UNEMPLOYMENT_RATES.get(int(education_level), 0.0))
    return float(
        ANNUAL_HOURS
        * (
            (1.0 - unemployment) * expected_hourly_wage
            + unemployment * UNEMPLOYMENT_BENEFIT_REPLACEMENT_RATE * expected_hourly_wage
        )
    )


def _education_dummy_effect(coefficients, education_level):
    """
    Return the education-dummy effect for a given alternative.

    This is robust to dummy names such as edu_2 and edu_2.0. The omitted
    reference education level automatically gets effect zero.
    """
    effect = 0.0
    for name, value in coefficients.items():
        if not str(name).startswith("edu_"):
            continue
        dummy_level = str(name).replace("edu_", "", 1)
        try:
            is_match = float(dummy_level) == float(education_level)
        except ValueError:
            is_match = dummy_level == str(education_level)
        if is_match:
            effect += float(value)
    return float(effect)


def mean_annual_income_profile(data, income_result, labor_market_entry_df):
    """
    Build the mean annual income profile used for diagnostic plots.

    This version includes unemployment benefits but not SU or any study benefit. Income before
    the education-specific labor-market entry age is therefore zero. The x-axis
    starts at the labor-market entry age for education level 1, or the earliest
    entry age if education level 1 is not available.

    Returns one row per education level and age with:
    - mean_labor_income
    - mean_annual_income
    - mean_discounted_annual_income
    - cumulative_mean_discounted_income
    - mean_expected_hourly_wage
    - mean_discounted_expected_hourly_wage
    """
    data_p = preprocess_individual_data(data, require_wage=False)
    labor_market_entry_df = normalize_labor_market_entry(labor_market_entry_df)
    labor_market_entry = labor_market_entry_df.set_index(MODEL_EDUCATION_COL)["LaborMarketEntry"]
    alternatives = valid_income_alternatives(labor_market_entry, income_result)
    discount_start_age = discount_origin_age(labor_market_entry)

    coef = income_result.coefficients
    background = build_background_regressors(data_p)
    n = len(data_p)

    # Base part: constant + background variables. Age, experience, and education
    # are added below because they change by age and alternative.
    base_log_wage = np.zeros(n, dtype=float)
    if "const" in coef.index:
        base_log_wage += float(coef["const"])

    for col in background.columns:
        if col in coef.index:
            base_log_wage += background[col].to_numpy(dtype=float) * float(coef[col])

    rows = []
    for education_level in alternatives:
        education_level = int(education_level)
        entry_age = int(labor_market_entry.loc[education_level])
        unemployment = float(UNEMPLOYMENT_RATES.get(education_level, 0.0))
        education_effect = _education_dummy_effect(coef, education_level)
        for age in range(discount_start_age, RETIREMENT_AGE + 1):
            discount_factor = discount_factor_for_age(age, discount_start_age)

            if age >= entry_age:
                experience = max(age - entry_age, 0)
                log_wage = (
                    base_log_wage
                    + education_effect
                    + float(coef.get(AGE_COL, 0.0)) * age
                    + float(coef.get("age_sq", 0.0)) * age**2
                    + float(coef.get(EXPERIENCE_COL, 0.0)) * experience
                    + float(coef.get("experience_sq", 0.0)) * experience**2
                )
                expected_hourly_wage = np.exp(log_wage)
                mean_hourly_wage = float(np.mean(expected_hourly_wage))
                mean_labor_income = mean_hourly_wage * ANNUAL_HOURS * (1.0 - unemployment)
                mean_unemployment_benefit_income = (
                    mean_hourly_wage
                    * ANNUAL_HOURS
                    * unemployment
                    * UNEMPLOYMENT_BENEFIT_REPLACEMENT_RATE
                )
            else:
                mean_hourly_wage = np.nan
                mean_labor_income = 0.0
                mean_unemployment_benefit_income = 0.0

            mean_annual_income = float(mean_labor_income + mean_unemployment_benefit_income)
            rows.append(
                {
                    MODEL_EDUCATION_COL: education_level,
                    "age": int(age),
                    "labor_market_entry_age": int(entry_age),
                    "discount_start_age": int(discount_start_age),
                    "mean_labor_income": float(mean_labor_income),
                    "mean_unemployment_benefit_income": float(mean_unemployment_benefit_income),
                    "mean_annual_income": mean_annual_income,
                    "mean_discounted_annual_income": mean_annual_income / discount_factor,
                    "mean_expected_hourly_wage": mean_hourly_wage,
                    "mean_discounted_expected_hourly_wage": (
                        mean_hourly_wage / discount_factor if not np.isnan(mean_hourly_wage) else np.nan
                    ),
                    "unemployment_rate": unemployment,
                    "unemployment_benefit_replacement_rate": UNEMPLOYMENT_BENEFIT_REPLACEMENT_RATE,
                }
            )

    profile = pd.DataFrame(rows)
    profile = profile.sort_values([MODEL_EDUCATION_COL, "age"]).reset_index(drop=True)
    profile["cumulative_mean_discounted_income"] = (
        profile.groupby(MODEL_EDUCATION_COL)["mean_discounted_annual_income"].cumsum()
    )
    return profile


# Backwards-compatible alias for the faster vectorized version.
fast_mean_annual_income_profile = mean_annual_income_profile


def _plot_profile_line(profile, y_col, y_label, title, output_path):
    """Internal helper for line plots by education level."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for education_level, group in profile.groupby(MODEL_EDUCATION_COL):
        ax.plot(
            group["age"],
            group[y_col],
            label=f"Education level {education_level}",
        )
    ax.set_xlabel("Age")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.show()
    return output_path


def plot_mean_discounted_annual_income(profile, output_prefix="income_block"):
    """Plot mean discounted annual labor income by education level."""
    return _plot_profile_line(
        profile=profile,
        y_col="mean_discounted_annual_income",
        y_label="Mean discounted annual income, DKK",
        title="Mean discounted annual income including unemployment benefits by education level",
        output_path=f"{output_prefix}_mean_discounted_annual_income.png",
    )


def plot_mean_discounted_hourly_wage(profile, output_prefix="income_block"):
    """Plot mean discounted expected hourly wage by education level."""
    return _plot_profile_line(
        profile=profile,
        y_col="mean_discounted_expected_hourly_wage",
        y_label="Mean discounted expected hourly wage, DKK",
        title="Mean discounted expected hourly wage by education level",
        output_path=f"{output_prefix}_mean_discounted_hourly_wage.png",
    )


def plot_cumulative_mean_discounted_income(profile, output_prefix="income_block"):
    """Plot cumulative present value of mean expected income by education level."""
    profile = profile.sort_values([MODEL_EDUCATION_COL, "age"]).copy()
    if "cumulative_mean_discounted_income" not in profile.columns:
        profile["cumulative_mean_discounted_income"] = (
            profile.groupby(MODEL_EDUCATION_COL)["mean_discounted_annual_income"].cumsum()
        )

    return _plot_profile_line(
        profile=profile,
        y_col="cumulative_mean_discounted_income",
        y_label="Cumulative mean expected discounted income, DKK",
        title="Cumulative mean expected discounted income by education level",
        output_path=f"{output_prefix}_cumulative_mean_discounted_income.png",
    )


def plot_income_block_test(data, income_result, labor_market_entry_df, output_prefix="income_block"):
    """
    Create the two original income-block diagnostic plots.

    This function is kept backwards compatible with the earlier test code:
        profile, annual_plot, hourly_plot = plot_income_block_test(...)

    It includes unemployment benefits but not SU. Income before labor-market entry is zero.
    """
    profile = mean_annual_income_profile(data, income_result, labor_market_entry_df)
    annual_path = plot_mean_discounted_annual_income(profile, output_prefix)
    hourly_path = plot_mean_discounted_hourly_wage(profile, output_prefix)
    return profile, annual_path, hourly_path


def plot_income_block_test_all(data, income_result, labor_market_entry_df, output_prefix="income_block"):
    """
    Create all three income-block diagnostic plots.

    Returns:
        profile, annual_plot, hourly_plot, cumulative_plot
    """
    profile = mean_annual_income_profile(data, income_result, labor_market_entry_df)
    annual_path = plot_mean_discounted_annual_income(profile, output_prefix)
    hourly_path = plot_mean_discounted_hourly_wage(profile, output_prefix)
    cumulative_path = plot_cumulative_mean_discounted_income(profile, output_prefix)
    return profile, annual_path, hourly_path, cumulative_path




def valid_income_alternatives(labor_market_entry, income_result=None):
    """
    Return education alternatives that should enter the income and choice blocks.

    The labor-market-entry table may contain education level 0 from an older
    setup. In the corrected config, unemployment rates are only defined for the
    valid choice alternatives. Therefore we keep the intersection between the
    labor-market-entry table and UNEMPLOYMENT_RATES.

    For the education-specific regression version, we additionally require that
    an income model was actually estimated for the alternative.
    """
    entry_levels = set(pd.Index(labor_market_entry.index).astype(int))
    unemployment_levels = {int(level) for level in UNEMPLOYMENT_RATES.keys()}
    alternatives = entry_levels.intersection(unemployment_levels)

    if income_result is not None and hasattr(income_result, "results_by_education"):
        model_levels = {int(level) for level in income_result.results_by_education.keys()}
        alternatives = alternatives.intersection(model_levels)

    alternatives = sorted(alternatives)
    if not alternatives:
        raise ValueError(
            "No valid income alternatives remain after matching the labor-market-entry "
            "table to UNEMPLOYMENT_RATES. Check that education levels use the same coding."
        )
    return alternatives

def expected_discounted_income(row, education_level, income_result, labor_market_entry):
    """
    Calculate expected discounted lifetime labor income for one alternative.

    Income starts when education level g enters the labor market, but all income
    streams are discounted back to the entry age of the earliest education level.
    Usually this is education level g=1.
    """
    education_level = int(education_level)
    entry_age = int(labor_market_entry.loc[education_level])

    discount_start_age = discount_origin_age(labor_market_entry)

    unemployment = float(UNEMPLOYMENT_RATES.get(education_level, 0.0))
    present_value = 0.0

    for age in range(entry_age, RETIREMENT_AGE + 1):
        projected = row.copy()
        projected[AGE_COL] = age
        projected[EXPERIENCE_COL] = max(age - entry_age, 0)
        expected_hourly_wage = predict_expected_hourly_wage_for_alternative(
            projected,
            education_level,
            income_result,
        )
        annual_income = annual_income_with_unemployment_benefits(
            expected_hourly_wage,
            education_level,
        )
        present_value += annual_income / discount_factor_for_age(age, discount_start_age)

    return float(present_value)


def add_expected_income_by_choice(data, income_result, labor_market_entry_df):
    """Return a long person-alternative dataset with expected income."""
    data = preprocess_individual_data(data, require_wage=False)
    labor_market_entry_df = normalize_labor_market_entry(labor_market_entry_df)
    labor_market_entry = labor_market_entry_df.set_index(MODEL_EDUCATION_COL)["LaborMarketEntry"]
    alternatives = valid_income_alternatives(labor_market_entry, income_result)
    rows = []

    for obs_id, row in data.iterrows():
        for education_level in alternatives:
            rows.append(
                {
                    "_row_id": obs_id,
                    MODEL_EDUCATION_COL: education_level,
                    "chosen": int(row[MODEL_EDUCATION_COL] == education_level),
                    "expected_lifetime_income": expected_discounted_income(
                        row,
                        education_level,
                        income_result,
                        labor_market_entry,
                    ),
                    "unemployment_rate": float(UNEMPLOYMENT_RATES.get(int(education_level), 0.0)),
                    "unemployment_benefit_replacement_rate": UNEMPLOYMENT_BENEFIT_REPLACEMENT_RATE,
                }
            )

    return pd.DataFrame(rows)


#%%
# Test run for this block only.
if __name__ == "__main__":
    data = load_individual_csv(INDIVIDUAL_DATA_PATH)
    labor_market_entry = load_named_excel_table(
        LABOR_MARKET_ENTRY_PATH,
        LABOR_MARKET_ENTRY_TABLE,
    )

    income_result = estimate_income_model(data)
    print_model_summary("Income block: pooled OLS log-wage equation, no smearing, with unemployment benefits", income_result)
    print("\nIncome coefficient table, first rows")
    print(income_result.coefficient_table().head())
    print("\nIncome summary by education")
    print(income_result.summary_table())

    income_long = add_expected_income_by_choice(data, income_result, labor_market_entry)
    print("\nExpected income by alternative, first rows")
    print(income_long.head())

# %%
