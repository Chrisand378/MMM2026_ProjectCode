#%%
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

#%%
BASE_FOLDER = Path(__file__).resolve().parent

RESULTS_FOLDER = BASE_FOLDER / "Results"
DATA_FILE_NAME = "choice_probabilities.csv"
SUBFOLDERS = [
    "1105_Data4_Eudd",
    "1105_Data4S_Eudd",
]

N_SIMULATIONS = 300



RANDOM_SEED = NONE

ROW_ID_COL = "_row_id"
EDUCATION_COL = "education_level"
PROB_COL = "choice_probability"
CHOSEN_COL = "chosen"

INCOME_COL = "expected_lifetime_income"
STUDY_DIFFICULTY_COL = "study_difficulty"

# Income variable in DKK -- dividing by 1,000,000 gives income in millions of DKK
INCOME_DIVISOR = 1_000_000

SAVE_INDIVIDUAL_SIMULATIONS = False

DATASET_LABELS = {
    "1105_Data4_Eudd": "Panel A: Pooled OLS",
    "1105_Data4S_Eudd": "Panel B: Education-specific OLS",
}


def log(message):
    """Small progress logger so the script does not look frozen."""
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}", flush=True)


#%%
def movement_label(movement_value):
    """Convert a numerical movement into the table label."""

    if movement_value == 0:
        return "Stay"

    direction = "up" if movement_value > 0 else "down"
    n = abs(int(movement_value))

    words = {
        1: "One level",
        2: "Two levels",
        3: "Three levels",
        4: "Four levels",
        5: "Five levels",
    }

    return f"{words.get(n, str(n) + ' levels')} {direction}"


#%%
def prepare_individual_base_data(df):
    """
    Creates one row per individual with:
    - observed education level
    - observed income
    - observed study difficulty

    Income is taken from the actually chosen education level.

    For study difficulty:
    - normally use the observed education level
    - if observed education level is 1, use study difficulty for level 2,
      because level 1 difficulty is zero.
    """

    required_cols = [
        ROW_ID_COL,
        EDUCATION_COL,
        CHOSEN_COL,
        INCOME_COL,
        STUDY_DIFFICULTY_COL,
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df[CHOSEN_COL] = pd.to_numeric(df[CHOSEN_COL], errors="coerce").fillna(0)
    df[EDUCATION_COL] = pd.to_numeric(df[EDUCATION_COL], errors="raise")
    df[INCOME_COL] = pd.to_numeric(df[INCOME_COL], errors="coerce")
    df[STUDY_DIFFICULTY_COL] = pd.to_numeric(
        df[STUDY_DIFFICULTY_COL], errors="coerce"
    )

    chosen_rows = df.loc[df[CHOSEN_COL] == 1].copy()
    chosen_counts = chosen_rows.groupby(ROW_ID_COL).size()

    if not (chosen_counts == 1).all():
        raise ValueError(
            "Each individual must have exactly one row with chosen == 1."
        )

    base = chosen_rows[[ROW_ID_COL, EDUCATION_COL, INCOME_COL]].copy()

    base = base.rename(
        columns={
            EDUCATION_COL: "observed_education_level",
            INCOME_COL: "income_million_dkk",
        }
    )

    base["income_million_dkk"] = base["income_million_dkk"] / INCOME_DIVISOR

    base["difficulty_education_level"] = np.where(
        base["observed_education_level"] == 1,
        2,
        base["observed_education_level"],
    )

    difficulty_lookup = df[[ROW_ID_COL, EDUCATION_COL, STUDY_DIFFICULTY_COL]].copy()
    difficulty_lookup = difficulty_lookup.rename(
        columns={
            EDUCATION_COL: "difficulty_education_level",
            STUDY_DIFFICULTY_COL: "study_difficulty",
        }
    )

    base = base.merge(
        difficulty_lookup,
        on=[ROW_ID_COL, "difficulty_education_level"],
        how="left",
    )

    if base["income_million_dkk"].isna().any():
        raise ValueError("Some individuals are missing income.")

    if base["study_difficulty"].isna().any():
        raise ValueError("Some individuals are missing study difficulty.")

    return base


#%%
def prepare_probability_matrix(df, base):
    """
    Creates a probability matrix:
    rows = individuals
    columns = possible education levels
    values = choice probabilities
    """

    required_cols = [ROW_ID_COL, EDUCATION_COL, PROB_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df[EDUCATION_COL] = pd.to_numeric(df[EDUCATION_COL], errors="raise")
    df[PROB_COL] = pd.to_numeric(df[PROB_COL], errors="coerce").fillna(0)

    if (df[PROB_COL] < 0).any():
        raise ValueError("Choice probabilities cannot be negative.")

    prob_wide = (
        df.pivot_table(
            index=ROW_ID_COL,
            columns=EDUCATION_COL,
            values=PROB_COL,
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index(axis=1)
    )

    # Keep individuals in exactly the same order as the base individual data.
    prob_wide = prob_wide.reindex(base[ROW_ID_COL])

    if prob_wide.isna().any().any():
        raise ValueError("Some individuals are missing choice probabilities.")

    education_levels = prob_wide.columns.to_numpy()
    prob_matrix = prob_wide.to_numpy(dtype=float)

    row_sums = prob_matrix.sum(axis=1)

    if (row_sums <= 0).any():
        raise ValueError("Some individuals have probabilities summing to zero.")

    # Normalize in case of small rounding errors.
    prob_matrix = prob_matrix / row_sums[:, None]

    return prob_matrix, education_levels, prob_wide


#%%
def simulate_predicted_education_levels(
    prob_matrix,
    education_levels,
    rng,
    n_simulations,
):
    """
    Randomly assigns each individual to one education level in each simulation.

    Each simulation uses new random draws. If RANDOM_SEED is None, the whole
    script gives new random results each time it is run. If RANDOM_SEED is an
    integer, results are reproducible.
    """

    cumulative_probs = np.cumsum(prob_matrix, axis=1)
    n_individuals = prob_matrix.shape[0]

    random_draws = rng.random(size=(n_simulations, n_individuals))

    simulated_positions = (
        random_draws[:, :, None] > cumulative_probs[None, :, :]
    ).sum(axis=2)

    simulated_education_levels = education_levels[simulated_positions]

    return simulated_education_levels


#%%
def build_individual_simulation_data(base, simulated_education_levels, dataset_name):
    """
    Creates individual-level simulation data.

    This is flexible but can be large because it has:
    n_simulations × n_individuals rows.
    """

    n_simulations, n_individuals = simulated_education_levels.shape

    observed_levels = base["observed_education_level"].to_numpy()
    predicted_levels = simulated_education_levels.reshape(-1)

    simulated_data = pd.DataFrame({
        "dataset": dataset_name,
        "model_label": DATASET_LABELS.get(dataset_name, dataset_name),
        "simulation": np.repeat(np.arange(1, n_simulations + 1), n_individuals),
        ROW_ID_COL: np.tile(base[ROW_ID_COL].to_numpy(), n_simulations),
        "observed_education_level": np.tile(observed_levels, n_simulations),
        "predicted_education_level": predicted_levels,
        "income_million_dkk": np.tile(
            base["income_million_dkk"].to_numpy(), n_simulations
        ),
        "study_difficulty": np.tile(
            base["study_difficulty"].to_numpy(), n_simulations
        ),
    })

    simulated_data["movement_value"] = (
        simulated_data["predicted_education_level"]
        - simulated_data["observed_education_level"]
    )

    movement_map = {
        value: movement_label(value)
        for value in sorted(simulated_data["movement_value"].unique())
    }

    simulated_data["predicted_movement"] = (
        simulated_data["movement_value"].map(movement_map)
    )

    return simulated_data


#%%
def create_distribution_counts_by_simulation(simulated_data):
    """Counts predicted education levels in each simulation."""

    counts = (
        simulated_data
        .groupby(["dataset", "model_label", "simulation", "predicted_education_level"])
        .size()
        .rename("N")
        .reset_index()
    )

    return counts


#%%
def create_distribution_summary(distribution_counts_by_simulation, prob_matrix, education_levels):
    """Summarizes the simulated distribution of predicted education levels."""

    expected_counts = pd.Series(
        prob_matrix.sum(axis=0),
        index=education_levels,
        name="expected_count",
    )

    n_individuals = prob_matrix.shape[0]

    summary = (
        distribution_counts_by_simulation
        .groupby(["dataset", "model_label", "predicted_education_level"], as_index=False)
        .agg(
            sim_mean_count=("N", "mean"),
            sim_std_count=("N", "std"),
            sim_p2_5_count=("N", lambda x: x.quantile(0.025)),
            sim_p97_5_count=("N", lambda x: x.quantile(0.975)),
        )
    )

    summary["expected_count"] = summary["predicted_education_level"].map(expected_counts)
    summary["expected_percent"] = 100 * summary["expected_count"] / n_individuals
    summary["sim_mean_percent"] = 100 * summary["sim_mean_count"] / n_individuals

    summary = summary[
        [
            "dataset",
            "model_label",
            "predicted_education_level",
            "expected_count",
            "expected_percent",
            "sim_mean_count",
            "sim_mean_percent",
            "sim_std_count",
            "sim_p2_5_count",
            "sim_p97_5_count",
        ]
    ].sort_values(["dataset", "predicted_education_level"])

    return summary


#%%
def create_uncertainty_stats(prob_matrix, n_simulations):
    """Creates individual-level uncertainty statistics from the probabilities."""

    n_individuals = prob_matrix.shape[0]
    n_education_levels = prob_matrix.shape[1]

    max_probs = prob_matrix.max(axis=1)

    # Normalized entropy:
    # 0 = very certain, 1 = very uncertain / probabilities evenly spread.
    eps = 1e-12
    entropy = -(prob_matrix * np.log(prob_matrix + eps)).sum(axis=1)

    if n_education_levels > 1:
        normalized_entropy = entropy / np.log(n_education_levels)
    else:
        normalized_entropy = np.zeros_like(entropy)

    uncertainty_stats = pd.DataFrame({
        "statistic": [
            "n_individuals",
            "n_education_levels",
            "n_simulations",
            "mean_max_choice_probability",
            "median_max_choice_probability",
            "share_with_top_probability_at_least_50_percent",
            "share_with_top_probability_at_least_75_percent",
            "mean_normalized_entropy",
            "median_normalized_entropy",
        ],
        "value": [
            n_individuals,
            n_education_levels,
            n_simulations,
            max_probs.mean(),
            np.median(max_probs),
            100 * np.mean(max_probs >= 0.50),
            100 * np.mean(max_probs >= 0.75),
            normalized_entropy.mean(),
            np.median(normalized_entropy),
        ],
    })

    return uncertainty_stats


#%%
def create_table_1_by_simulation(simulated_data):
    """
    Creates the data needed for your first sorting table, separately for each simulation.
    """

    table = (
        simulated_data
        .groupby(
            [
                "dataset",
                "model_label",
                "simulation",
                "observed_education_level",
                "movement_value",
                "predicted_movement",
            ],
            as_index=False,
        )
        .agg(
            N=(ROW_ID_COL, "size"),
            mean_income_million_dkk=("income_million_dkk", "mean"),
            mean_study_difficulty=("study_difficulty", "mean"),
            sum_income_million_dkk=("income_million_dkk", "sum"),
            sum_study_difficulty=("study_difficulty", "sum"),
        )
    )

    return table


#%%
def create_table_1_simulation_summary(table_1_by_simulation):
    """
    Averages your first table over the simulations.

    pooled_mean_* is the preferred mean because it weights by the number of
    people in the group across simulations.
    """

    summary = (
        table_1_by_simulation
        .groupby(
            [
                "dataset",
                "model_label",
                "observed_education_level",
                "movement_value",
                "predicted_movement",
            ],
            as_index=False,
        )
        .agg(
            n_nonempty_simulations=("simulation", "nunique"),
            mean_N_when_nonempty=("N", "mean"),
            sd_N_when_nonempty=("N", "std"),
            p2_5_N_when_nonempty=("N", lambda x: x.quantile(0.025)),
            p97_5_N_when_nonempty=("N", lambda x: x.quantile(0.975)),
            mean_of_sim_mean_income_million_dkk=("mean_income_million_dkk", "mean"),
            mean_of_sim_mean_study_difficulty=("mean_study_difficulty", "mean"),
            total_simulated_assignments=("N", "sum"),
            total_income_million_dkk=("sum_income_million_dkk", "sum"),
            total_study_difficulty=("sum_study_difficulty", "sum"),
        )
    )

    summary["mean_N_including_empty_simulations"] = (
        summary["total_simulated_assignments"] / N_SIMULATIONS
    )

    summary["pooled_mean_income_million_dkk"] = (
        summary["total_income_million_dkk"]
        / summary["total_simulated_assignments"]
    )

    summary["pooled_mean_study_difficulty"] = (
        summary["total_study_difficulty"]
        / summary["total_simulated_assignments"]
    )

    summary = summary.sort_values(
        ["dataset", "observed_education_level", "movement_value"]
    )

    return summary


#%%
def create_delta_table_by_simulation(table_1_by_simulation):
    """
    Creates absolute differences relative to the Stay group within:
    - same dataset
    - same simulation
    - same observed education level
    """

    stay_table = table_1_by_simulation.loc[
        table_1_by_simulation["movement_value"] == 0,
        [
            "dataset",
            "model_label",
            "simulation",
            "observed_education_level",
            "mean_income_million_dkk",
            "mean_study_difficulty",
        ],
    ].copy()

    stay_table = stay_table.rename(
        columns={
            "mean_income_million_dkk": "stay_mean_income_million_dkk",
            "mean_study_difficulty": "stay_mean_study_difficulty",
        }
    )

    delta_table = table_1_by_simulation.merge(
        stay_table,
        on=[
            "dataset",
            "model_label",
            "simulation",
            "observed_education_level",
        ],
        how="left",
    )

    delta_table["delta_mean_income_million_dkk"] = (
        delta_table["mean_income_million_dkk"]
        - delta_table["stay_mean_income_million_dkk"]
    )

    delta_table["delta_mean_study_difficulty"] = (
        delta_table["mean_study_difficulty"]
        - delta_table["stay_mean_study_difficulty"]
    )

    return delta_table


#%%
def create_delta_table_summary(delta_table_by_simulation):
    """Summarizes absolute differences relative to stayers."""

    summary = (
        delta_table_by_simulation
        .groupby(
            [
                "dataset",
                "model_label",
                "observed_education_level",
                "movement_value",
                "predicted_movement",
            ],
            as_index=False,
        )
        .agg(
            n_nonempty_simulations=("simulation", "nunique"),
            mean_N_when_nonempty=("N", "mean"),
            mean_delta_income_million_dkk=("delta_mean_income_million_dkk", "mean"),
            mean_delta_study_difficulty=("delta_mean_study_difficulty", "mean"),
            p2_5_delta_income_million_dkk=(
                "delta_mean_income_million_dkk",
                lambda x: x.quantile(0.025),
            ),
            p97_5_delta_income_million_dkk=(
                "delta_mean_income_million_dkk",
                lambda x: x.quantile(0.975),
            ),
            p2_5_delta_study_difficulty=(
                "delta_mean_study_difficulty",
                lambda x: x.quantile(0.025),
            ),
            p97_5_delta_study_difficulty=(
                "delta_mean_study_difficulty",
                lambda x: x.quantile(0.975),
            ),
        )
    )

    summary = summary.sort_values(
        ["dataset", "observed_education_level", "movement_value"]
    )

    return summary


#%%
def create_relative_difference_table_by_simulation(delta_table_by_simulation):
    """
    Creates relative differences from the delta table.

    Formula:
    relative difference (%) = 100 * (group mean - stay mean) / stay mean
    """

    table = delta_table_by_simulation.copy()

    total_within_observed_level = (
        table
        .groupby(
            [
                "dataset",
                "model_label",
                "simulation",
                "observed_education_level",
            ]
        )["N"]
        .transform("sum")
    )

    table["movement_share_percent"] = (
        100 * table["N"] / total_within_observed_level
    )

    table["relative_income_percent"] = np.where(
        table["stay_mean_income_million_dkk"].abs() > 1e-12,
        100
        * table["delta_mean_income_million_dkk"]
        / table["stay_mean_income_million_dkk"],
        np.nan,
    )

    table["relative_study_difficulty_percent"] = np.where(
        table["stay_mean_study_difficulty"].abs() > 1e-12,
        100
        * table["delta_mean_study_difficulty"]
        / table["stay_mean_study_difficulty"],
        np.nan,
    )

    return table


#%%
def create_relative_difference_table_summary(relative_table_by_simulation):
    """
    Summarizes relative differences across simulations.
    """

    summary = (
        relative_table_by_simulation
        .groupby(
            [
                "dataset",
                "model_label",
                "observed_education_level",
                "movement_value",
                "predicted_movement",
            ],
            as_index=False,
        )
        .agg(
            n_nonempty_simulations=("simulation", "nunique"),
            mean_N_when_nonempty=("N", "mean"),
            mean_movement_share_percent=("movement_share_percent", "mean"),
            mean_relative_income_percent=("relative_income_percent", "mean"),
            p2_5_relative_income_percent=(
                "relative_income_percent",
                lambda x: x.quantile(0.025),
            ),
            p97_5_relative_income_percent=(
                "relative_income_percent",
                lambda x: x.quantile(0.975),
            ),
            mean_relative_study_difficulty_percent=(
                "relative_study_difficulty_percent",
                "mean",
            ),
            p2_5_relative_study_difficulty_percent=(
                "relative_study_difficulty_percent",
                lambda x: x.quantile(0.025),
            ),
            p97_5_relative_study_difficulty_percent=(
                "relative_study_difficulty_percent",
                lambda x: x.quantile(0.975),
            ),
        )
    )

    summary = summary.sort_values(
        ["dataset", "observed_education_level", "movement_value"]
    )

    return summary


#%%
def create_exact_probability_weighted_table(base, prob_wide, dataset_name):
    """
    Exact probability-weighted version of the first sorting table.

    This is not simulated. It is the expectation implied directly by the
    predicted choice probabilities.
    """

    prob_long = (
        prob_wide
        .stack()
        .rename(PROB_COL)
        .reset_index()
        .rename(columns={EDUCATION_COL: "predicted_education_level"})
    )

    exact_data = base.merge(prob_long, on=ROW_ID_COL, how="left")

    exact_data["dataset"] = dataset_name
    exact_data["model_label"] = DATASET_LABELS.get(dataset_name, dataset_name)

    exact_data["movement_value"] = (
        exact_data["predicted_education_level"]
        - exact_data["observed_education_level"]
    )

    exact_data["predicted_movement"] = exact_data["movement_value"].apply(
        movement_label
    )

    exact_data["weighted_income"] = (
        exact_data[PROB_COL] * exact_data["income_million_dkk"]
    )

    exact_data["weighted_study_difficulty"] = (
        exact_data[PROB_COL] * exact_data["study_difficulty"]
    )

    exact_table = (
        exact_data
        .groupby(
            [
                "dataset",
                "model_label",
                "observed_education_level",
                "movement_value",
                "predicted_movement",
            ],
            as_index=False,
        )
        .agg(
            expected_N=(PROB_COL, "sum"),
            weighted_income_sum=("weighted_income", "sum"),
            weighted_study_difficulty_sum=("weighted_study_difficulty", "sum"),
        )
    )

    exact_table["expected_mean_income_million_dkk"] = (
        exact_table["weighted_income_sum"] / exact_table["expected_N"]
    )

    exact_table["expected_mean_study_difficulty"] = (
        exact_table["weighted_study_difficulty_sum"] / exact_table["expected_N"]
    )

    exact_table = exact_table.sort_values(
        ["dataset", "observed_education_level", "movement_value"]
    )

    return exact_table


#%%
def create_exact_relative_difference_table(exact_table):
    """
    Creates exact probability-weighted relative differences compared with stayers.

    Formula:
    relative difference (%) = 100 * (group mean - stay mean) / stay mean
    """

    table = exact_table.copy()

    stay_table = table.loc[
        table["movement_value"] == 0,
        [
            "dataset",
            "model_label",
            "observed_education_level",
            "expected_mean_income_million_dkk",
            "expected_mean_study_difficulty",
        ],
    ].copy()

    stay_table = stay_table.rename(
        columns={
            "expected_mean_income_million_dkk": "stay_expected_mean_income_million_dkk",
            "expected_mean_study_difficulty": "stay_expected_mean_study_difficulty",
        }
    )

    table = table.merge(
        stay_table,
        on=["dataset", "model_label", "observed_education_level"],
        how="left",
    )

    total_expected_within_observed_level = (
        table
        .groupby(["dataset", "model_label", "observed_education_level"])["expected_N"]
        .transform("sum")
    )

    table["expected_movement_share_percent"] = (
        100 * table["expected_N"] / total_expected_within_observed_level
    )

    table["exact_delta_income_million_dkk"] = (
        table["expected_mean_income_million_dkk"]
        - table["stay_expected_mean_income_million_dkk"]
    )

    table["exact_delta_study_difficulty"] = (
        table["expected_mean_study_difficulty"]
        - table["stay_expected_mean_study_difficulty"]
    )

    table["exact_relative_income_percent"] = np.where(
        table["stay_expected_mean_income_million_dkk"].abs() > 1e-12,
        100
        * table["exact_delta_income_million_dkk"]
        / table["stay_expected_mean_income_million_dkk"],
        np.nan,
    )

    table["exact_relative_study_difficulty_percent"] = np.where(
        table["stay_expected_mean_study_difficulty"].abs() > 1e-12,
        100
        * table["exact_delta_study_difficulty"]
        / table["stay_expected_mean_study_difficulty"],
        np.nan,
    )

    table = table.sort_values(
        ["dataset", "observed_education_level", "movement_value"]
    )

    return table


#%%
def process_one_dataset(subfolder, rng):
    """Run the full simulation and table creation for one dataset."""

    start_time = perf_counter()

    dataset_path = RESULTS_FOLDER / subfolder / DATA_FILE_NAME
    output_folder = RESULTS_FOLDER / subfolder
    output_folder.mkdir(parents=True, exist_ok=True)

    log("=" * 90)
    log(f"Dataset: {subfolder}")
    log(f"Reading: {dataset_path}")

    df = pd.read_csv(dataset_path)
    log(f"Loaded {len(df):,} rows")

    base = prepare_individual_base_data(df)
    log(f"Prepared base data for {len(base):,} individuals")

    prob_matrix, education_levels, prob_wide = prepare_probability_matrix(
        df=df,
        base=base,
    )
    log(f"Prepared probability matrix: {prob_matrix.shape[0]:,} individuals × {prob_matrix.shape[1]:,} alternatives")

    simulated_education_levels = simulate_predicted_education_levels(
        prob_matrix=prob_matrix,
        education_levels=education_levels,
        rng=rng,
        n_simulations=N_SIMULATIONS,
    )
    log("Finished random simulation draws")

    simulated_data = build_individual_simulation_data(
        base=base,
        simulated_education_levels=simulated_education_levels,
        dataset_name=subfolder,
    )
    log(f"Built individual simulation data: {len(simulated_data):,} rows")

    distribution_counts_by_simulation = create_distribution_counts_by_simulation(
        simulated_data
    )
    distribution_summary = create_distribution_summary(
        distribution_counts_by_simulation=distribution_counts_by_simulation,
        prob_matrix=prob_matrix,
        education_levels=education_levels,
    )
    uncertainty_stats = create_uncertainty_stats(
        prob_matrix=prob_matrix,
        n_simulations=N_SIMULATIONS,
    )
    log("Created distribution and uncertainty tables")

    table_1_by_simulation = create_table_1_by_simulation(simulated_data)
    table_1_summary = create_table_1_simulation_summary(table_1_by_simulation)
    log("Created Table 1 simulation tables")

    delta_table_by_simulation = create_delta_table_by_simulation(table_1_by_simulation)
    delta_table_summary = create_delta_table_summary(delta_table_by_simulation)
    log("Created absolute-difference tables")

    relative_table_by_simulation = create_relative_difference_table_by_simulation(
        delta_table_by_simulation
    )
    relative_table_summary = create_relative_difference_table_summary(
        relative_table_by_simulation
    )
    log("Created relative-difference tables")

    exact_table = create_exact_probability_weighted_table(
        base=base,
        prob_wide=prob_wide,
        dataset_name=subfolder,
    )
    exact_relative_table = create_exact_relative_difference_table(exact_table)
    log("Created exact probability-weighted tables")

    # Save dataset-specific files.
    distribution_counts_by_simulation.to_csv(
        output_folder / "simulation_distribution_counts_by_run.csv",
        index=False,
    )
    distribution_summary.to_csv(
        output_folder / "simulation_distribution_summary.csv",
        index=False,
    )
    uncertainty_stats.to_csv(
        output_folder / "simulation_uncertainty_stats.csv",
        index=False,
    )
    table_1_by_simulation.to_csv(
        output_folder / "simulation_table_1_by_run.csv",
        index=False,
    )
    table_1_summary.to_csv(
        output_folder / "simulation_table_1_summary.csv",
        index=False,
    )
    delta_table_by_simulation.to_csv(
        output_folder / "simulation_delta_table_by_run.csv",
        index=False,
    )
    delta_table_summary.to_csv(
        output_folder / "simulation_delta_table_summary.csv",
        index=False,
    )
    relative_table_by_simulation.to_csv(
        output_folder / "simulation_relative_difference_table_by_run.csv",
        index=False,
    )
    relative_table_summary.to_csv(
        output_folder / "simulation_relative_difference_table_summary.csv",
        index=False,
    )
    exact_table.to_csv(
        output_folder / "probability_weighted_table_1_exact.csv",
        index=False,
    )
    exact_relative_table.to_csv(
        output_folder / "probability_weighted_relative_difference_table_exact.csv",
        index=False,
    )

    if SAVE_INDIVIDUAL_SIMULATIONS:
        log("Saving individual simulation assignments. This can take a long time.")
        simulated_data.to_csv(
            output_folder / "simulation_individual_assignments.csv.gz",
            index=False,
            compression="gzip",
        )

    log(f"Saved all dataset-specific outputs for {subfolder}")

    print("\nAverage simulated Table 1")
    print(
        table_1_summary[
            [
                "model_label",
                "observed_education_level",
                "predicted_movement",
                "mean_N_including_empty_simulations",
                "pooled_mean_income_million_dkk",
                "pooled_mean_study_difficulty",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nRelative differences compared with stayers")
    print(
        relative_table_summary[
            [
                "model_label",
                "observed_education_level",
                "predicted_movement",
                "n_nonempty_simulations",
                "mean_movement_share_percent",
                "mean_relative_income_percent",
                "mean_relative_study_difficulty_percent",
            ]
        ].round(3).to_string(index=False)
    )

    elapsed = perf_counter() - start_time
    log(f"Finished {subfolder} in {elapsed:,.1f} seconds")

    return {
        "distribution_counts_by_simulation": distribution_counts_by_simulation,
        "distribution_summary": distribution_summary,
        "uncertainty_stats": uncertainty_stats,
        "table_1_by_simulation": table_1_by_simulation,
        "table_1_summary": table_1_summary,
        "delta_table_by_simulation": delta_table_by_simulation,
        "delta_table_summary": delta_table_summary,
        "relative_table_by_simulation": relative_table_by_simulation,
        "relative_table_summary": relative_table_summary,
        "exact_table": exact_table,
        "exact_relative_table": exact_relative_table,
    }


#%%
def main():
    """Run both datasets and save combined outputs."""

    rng = np.random.default_rng(RANDOM_SEED)

    all_results = {}
    all_distribution_counts = []
    all_distribution_summaries = []
    all_uncertainty_stats = []
    all_table_1_by_simulation = []
    all_table_1_summaries = []
    all_delta_tables = []
    all_delta_summaries = []
    all_relative_tables = []
    all_relative_summaries = []
    all_exact_tables = []
    all_exact_relative_tables = []

    for subfolder in SUBFOLDERS:
        result = process_one_dataset(subfolder=subfolder, rng=rng)
        all_results[subfolder] = result

        all_distribution_counts.append(result["distribution_counts_by_simulation"])
        all_distribution_summaries.append(result["distribution_summary"])
        all_uncertainty_stats.append(
            result["uncertainty_stats"].assign(dataset=subfolder)
        )
        all_table_1_by_simulation.append(result["table_1_by_simulation"])
        all_table_1_summaries.append(result["table_1_summary"])
        all_delta_tables.append(result["delta_table_by_simulation"])
        all_delta_summaries.append(result["delta_table_summary"])
        all_relative_tables.append(result["relative_table_by_simulation"])
        all_relative_summaries.append(result["relative_table_summary"])
        all_exact_tables.append(result["exact_table"])
        all_exact_relative_tables.append(result["exact_relative_table"])

    combined_output_folder = RESULTS_FOLDER / "simulation_outputs_combined"
    combined_output_folder.mkdir(parents=True, exist_ok=True)

    pd.concat(all_distribution_counts, ignore_index=True).to_csv(
        combined_output_folder / "simulation_distribution_counts_by_run_all_datasets.csv",
        index=False,
    )
    pd.concat(all_distribution_summaries, ignore_index=True).to_csv(
        combined_output_folder / "simulation_distribution_summary_all_datasets.csv",
        index=False,
    )
    pd.concat(all_uncertainty_stats, ignore_index=True).to_csv(
        combined_output_folder / "simulation_uncertainty_stats_all_datasets.csv",
        index=False,
    )
    pd.concat(all_table_1_by_simulation, ignore_index=True).to_csv(
        combined_output_folder / "simulation_table_1_by_run_all_datasets.csv",
        index=False,
    )
    pd.concat(all_table_1_summaries, ignore_index=True).to_csv(
        combined_output_folder / "simulation_table_1_summary_all_datasets.csv",
        index=False,
    )
    pd.concat(all_delta_tables, ignore_index=True).to_csv(
        combined_output_folder / "simulation_delta_table_by_run_all_datasets.csv",
        index=False,
    )
    pd.concat(all_delta_summaries, ignore_index=True).to_csv(
        combined_output_folder / "simulation_delta_table_summary_all_datasets.csv",
        index=False,
    )
    pd.concat(all_relative_tables, ignore_index=True).to_csv(
        combined_output_folder / "simulation_relative_difference_table_by_run_all_datasets.csv",
        index=False,
    )
    pd.concat(all_relative_summaries, ignore_index=True).to_csv(
        combined_output_folder / "simulation_relative_difference_table_summary_all_datasets.csv",
        index=False,
    )
    pd.concat(all_exact_tables, ignore_index=True).to_csv(
        combined_output_folder / "probability_weighted_table_1_exact_all_datasets.csv",
        index=False,
    )
    pd.concat(all_exact_relative_tables, ignore_index=True).to_csv(
        combined_output_folder / "probability_weighted_relative_difference_table_exact_all_datasets.csv",
        index=False,
    )

    log(f"Saved combined outputs to: {combined_output_folder}")

    return all_results


#%%
if __name__ == "__main__":
    main()

# %%
