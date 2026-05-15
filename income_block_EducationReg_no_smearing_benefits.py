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
    preprocess_individual_data,
    print_model_summary,
    require_columns,
)


# Unemployment benefit replacement rate used in the annual income formula.
UNEMPLOYMENT_BENEFIT_REPLACEMENT_RATE = 0.25


#%%
@dataclass
class IncomeModelResult:
    """Container for the education-specific income-block OLS results."""

    results_by_education: dict
    coefficients: pd.DataFrame
    standard_errors: pd.DataFrame
    t_values: pd.DataFrame
    p_values: pd.DataFrame
    nobs: pd.Series
    r_squared: pd.Series
    residual_std: pd.Series

    def coefficient_table(self):
        """Return one long table with estimates, standard errors, t-values, and p-values."""
        rows = []
        for education_level, result in self.results_by_education.items():
            for variable in result.coefficients.index:
                rows.append(
                    {
                        MODEL_EDUCATION_COL: education_level,
                        "variable": variable,
                        "coefficient": result.coefficients.loc[variable],
                        "standard_error": result.standard_errors.loc[variable],
                        "t_value": result.t_values.loc[variable],
                        "p_value": result.p_values.loc[variable],
                        "nobs": result.nobs,
                        "r_squared": result.r_squared,
                        "residual_std": result.residual_std,
                    }
                )
        return pd.DataFrame(rows)

    def summary_table(self):
        """Return education-level summary statistics used by export scripts."""
        levels = self.nobs.index
        return pd.DataFrame(
            {
                "education_level": levels,
                "nobs": self.nobs.reindex(levels).values,
                "r_squared": self.r_squared.reindex(levels).values,
                "residual_std": self.residual_std.reindex(levels).values,
            }
        )

    def coefficient_dict(self):
        """Return coefficients as a nested dict: {education_level: {variable: coefficient}}."""
        return {
            int(education_level): result.coefficients.to_dict()
            for education_level, result in self.results_by_education.items()
        }

    def p_value_dict(self):
        """Return p-values as a nested dict: {education_level: {variable: p_value}}."""
        return {
            int(education_level): result.p_values.to_dict()
            for education_level, result in self.results_by_education.items()
        }


#%%
def prepare_income_data(data):
    """
    Prepare the wage equation data.

    The dependent variable is log_timelon, created from timelon.
    """
    data = preprocess_individual_data(data, require_wage=True)
    require_columns(data, [MODEL_EDUCATION_COL, AGE_COL, EXPERIENCE_COL, LOG_WAGE_COL], "income data")

    if AGE_COL in data.columns:
        data["age_sq"] = data[AGE_COL] ** 2
    if EXPERIENCE_COL in data.columns:
        data["experience_sq"] = data[EXPERIENCE_COL] ** 2

    return data


def income_regressors(data):
    """
    Build regressors for the education-specific log-wage equations.

    Education dummies are intentionally excluded here, because estimate_income_model()
    estimates one separate OLS regression for each broad education level.
    """
    profile_cols = [AGE_COL, "age_sq", EXPERIENCE_COL, "experience_sq"]
    background = build_background_regressors(data)

    x = pd.concat([data[profile_cols].astype(float), background], axis=1)
    assert_no_forbidden_regressors(x.columns)
    return x


#%%
def estimate_income_model(data):
    """
    Estimate one OLS log-wage equation for each broad education level.

    For each g:
        log(timelon) = age profile + experience profile + background variables + error.

    The returned IncomeModelResult preserves the high-level structure: downstream
    code can still pass income_result into add_expected_income_by_choice().
    """
    data = prepare_income_data(data)
    x_all = income_regressors(data)
    y_all = data[LOG_WAGE_COL]

    results_by_education = {}
    for education_level in sorted(pd.Series(data[MODEL_EDUCATION_COL]).dropna().unique()):
        mask = data[MODEL_EDUCATION_COL] == education_level
        result = fit_ols(y_all.loc[mask], x_all.loc[mask])
        results_by_education[int(education_level)] = result

    return IncomeModelResult(
        results_by_education=results_by_education,
        coefficients=pd.DataFrame({g: res.coefficients for g, res in results_by_education.items()}),
        standard_errors=pd.DataFrame({g: res.standard_errors for g, res in results_by_education.items()}),
        t_values=pd.DataFrame({g: res.t_values for g, res in results_by_education.items()}),
        p_values=pd.DataFrame({g: res.p_values for g, res in results_by_education.items()}),
        nobs=pd.Series({g: res.nobs for g, res in results_by_education.items()}, name="nobs"),
        r_squared=pd.Series({g: res.r_squared for g, res in results_by_education.items()}, name="r_squared"),
        residual_std=pd.Series(
            {g: res.residual_std for g, res in results_by_education.items()},
            name="residual_std",
        ),
    )


#%%
def _education_result(education_level, income_result):
    """Select the OLS result for a broad education level."""
    education_level = int(education_level)
    if education_level not in income_result.results_by_education:
        available = sorted(income_result.results_by_education)
        raise ValueError(
            f"No income model estimated for education level {education_level}. "
            f"Available levels are {available}."
        )
    return income_result.results_by_education[education_level]


def predict_log_wage_for_alternative(row, education_level, income_result):
    """Predict log hourly wage for one person and one broad education level."""
    result = _education_result(education_level, income_result)
    values = {}

    for name in result.coefficients.index:
        if name == "const":
            values[name] = 1.0
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

    x = pd.Series(values).reindex(result.coefficients.index).fillna(0.0)
    return float(x @ result.coefficients)


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


# Backwards-compatible name used in earlier versions.
def common_discount_start_age(labor_market_entry):
    return discount_origin_age(labor_market_entry)


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

    Income includes unemployment benefits using:
        annual income = H * (1 - u_g) * w_hat + H * u_g * b * w_hat.

    The discount origin is the entry age of education level g=1 for all education
    alternatives. This gives longer educations a true delay penalty relative to the
    earliest labor-market-entry age.
    """
    education_level = int(education_level)
    entry_age = int(labor_market_entry.loc[education_level])
    discount_start_age = discount_origin_age(labor_market_entry)
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
def mean_annual_income_profile(data, income_result, labor_market_entry_df):
    """
    Build the mean annual income profile used for diagnostic plots.

    This version is adapted to the education-specific OLS model. It includes
    unemployment benefits but not SU or any study benefit. Income before the education-specific
    labor-market entry age is therefore zero. The x-axis starts at the labor-market
    entry age for education level 1, or the earliest entry age if education level 1
    is not available.

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

    background = build_background_regressors(data_p)
    n = len(data_p)
    rows = []

    for education_level in alternatives:
        education_level = int(education_level)
        result = _education_result(education_level, income_result)
        coef = result.coefficients
        entry_age = int(labor_market_entry.loc[education_level])
        unemployment = float(UNEMPLOYMENT_RATES.get(education_level, 0.0))
        # Base part for this education-specific wage equation: constant + background.
        # Age and experience are added below because they change by age.
        base_log_wage = np.zeros(n, dtype=float)
        if "const" in coef.index:
            base_log_wage += float(coef["const"])

        for col in background.columns:
            if col in coef.index:
                base_log_wage += background[col].to_numpy(dtype=float) * float(coef[col])

        for age in range(discount_start_age, RETIREMENT_AGE + 1):
            discount_factor = discount_factor_for_age(age, discount_start_age)

            if age >= entry_age:
                experience = max(age - entry_age, 0)
                log_wage = (
                    base_log_wage
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


# Backwards-compatible alias for the vectorized profile builder.
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

    This function is backwards compatible with the earlier test code:
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


#%%
# Test run for this block only.
if __name__ == "__main__":
    data = load_individual_csv(INDIVIDUAL_DATA_PATH)
    labor_market_entry = load_named_excel_table(
        LABOR_MARKET_ENTRY_PATH,
        LABOR_MARKET_ENTRY_TABLE,
    )

    income_result = estimate_income_model(data)
    print_model_summary("Income block: education-specific OLS log-wage equations, no smearing, with unemployment benefits", income_result)

    print("\nCoefficient table, first rows")
    print(income_result.coefficient_table().head())
    print("\nIncome summary by education")
    print(income_result.summary_table())

    income_long = add_expected_income_by_choice(data, income_result, labor_market_entry)
    print("\nExpected income by alternative, first rows")
    print(income_long.head())

# %%
