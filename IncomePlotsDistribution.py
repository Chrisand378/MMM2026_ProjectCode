#%% Imports
import matplotlib.pyplot as plt

from config import (
    INDIVIDUAL_DATA_PATH,
    LABOR_MARKET_ENTRY_PATH,
    LABOR_MARKET_ENTRY_TABLE,
    MODEL_EDUCATION_COL,
)
from income_block_SingleReg import (
    estimate_income_model,
    mean_annual_income_profile,
)
from model_utils import load_individual_csv, load_named_excel_table, print_model_summary


#%% Plot settings
# All money variables in the plots are shown in thousands of DKK.
DKK_TO_THOUSAND_DKK = 1_000.0

FIGSIZE = (9, 5)
DPI = 200
LINEWIDTH = 2.2
MARKER_SIZE = 6.5
MARK_EVERY = 4

# Marker shapes make the lines easier to distinguish without relying only on color.
# Education levels are assigned in sorted order.
MARKERS = ["o", "^", "s", "*", "D", "P", "X"]

AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14


def plot_profile_line_thousand_dkk(profile, y_col, y_label, output_path):
    """
    Plot one profile line per education level.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for marker_index, (education_level, group) in enumerate(
        profile.groupby(MODEL_EDUCATION_COL, sort=True)
    ):
        ax.plot(
            group["age"],
            group[y_col] / DKK_TO_THOUSAND_DKK,
            linewidth=LINEWIDTH,
            marker=MARKERS[marker_index % len(MARKERS)],
            markersize=MARKER_SIZE,
            markevery=MARK_EVERY,
        )

    ax.set_xlabel("Age", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.show()
    return output_path


#%% Load data and estimate income model
data = load_individual_csv(INDIVIDUAL_DATA_PATH)
labor_market_entry = load_named_excel_table(
    LABOR_MARKET_ENTRY_PATH,
    LABOR_MARKET_ENTRY_TABLE,
)

income_result = estimate_income_model(data)
print_model_summary(
    "Income block: pooled OLS log-wage equation with student grants",
    income_result,
)


#%% Build income profile
profile = mean_annual_income_profile(data, income_result, labor_market_entry)
profile = profile.sort_values([MODEL_EDUCATION_COL, "age"]).copy()

profile["cumulative_mean_discounted_income"] = (
    profile.groupby(MODEL_EDUCATION_COL)["mean_discounted_annual_income"].cumsum()
)

profile.to_csv("income_block_test_profile.csv", index=False)


#%% Create plots with markers, and with y-axis in thousand DKK
annual_plot = plot_profile_line_thousand_dkk(
    profile=profile,
    y_col="mean_discounted_annual_income",
    y_label="Discounted annual income",
    output_path="income_block_mean_discounted_annual_income_thousandDKK_no_title_no_legend_markers.png",
)

hourly_plot = plot_profile_line_thousand_dkk(
    profile=profile,
    y_col="mean_discounted_expected_hourly_wage",
    y_label="Discounted expected hourly wage",
    output_path="income_block_mean_discounted_hourly_wage_thousandDKK_no_title_no_legend_markers.png",
)

cumulative_plot = plot_profile_line_thousand_dkk(
    profile=profile,
    y_col="cumulative_mean_discounted_income",
    y_label="Cumulative discounted annual income",
    output_path="income_block_cumulative_mean_discounted_income_thousandDKK_no_title_no_legend_markers.png",
)

print("Saved:")
print(annual_plot)
print(hourly_plot)
print(cumulative_plot)
print("income_block_test_profile.csv")

# %%
