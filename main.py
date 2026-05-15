#%%
from config import (
    INDIVIDUAL_DATA_PATH,
    LABOR_MARKET_ENTRY_PATH,
    LABOR_MARKET_ENTRY_TABLE,
)
from income_block_EducationReg_no_smearing_benefits import (
    add_expected_income_by_choice,
    estimate_income_model,
)
from model_utils import load_individual_csv, load_named_excel_table, print_model_summary
from study_difficulty_block_statsmodels import (
    add_study_difficulty_to_long,
    estimate_study_difficulty_model,
)
from utility_choice_block_statsmodels_alt_constants_only import (
    estimate_utility_choice_model,
    predict_choice_probabilities,
)


#%%
def run_all(
    individual_data_path=INDIVIDUAL_DATA_PATH,
    labor_market_entry_path=LABOR_MARKET_ENTRY_PATH,
):
    """
    Run all model blocks in chronological order.

    1. Income block: OLS wage equation and expected discounted labor income.
    2. Study-difficulty block: ordered probit for completed broad education.
    3. Utility-choice block: conditional logit over broad education levels.
    """
    data = load_individual_csv(individual_data_path)
    labor_market_entry = load_named_excel_table(
        labor_market_entry_path,
        LABOR_MARKET_ENTRY_TABLE,
    )

    income_result = estimate_income_model(data)
    choice_long = add_expected_income_by_choice(data, income_result, labor_market_entry)

    study_result = estimate_study_difficulty_model(data)
    choice_long = add_study_difficulty_to_long(choice_long, data, study_result)

    utility_result = estimate_utility_choice_model(choice_long)
    choice_probabilities = predict_choice_probabilities(choice_long, utility_result)

    return {
        "data": data,
        "labor_market_entry": labor_market_entry,
        "income_result": income_result,
        "study_result": study_result,
        "utility_result": utility_result,
        "choice_long": choice_long,
        "choice_probabilities": choice_probabilities,
    }


#%%
# Main run: use this cell to estimate the full model at once.
if __name__ == "__main__":
    results = run_all()

    print_model_summary("Income block: OLS log-wage equation", results["income_result"])

    print_model_summary("Study-difficulty block: ordered probit", results["study_result"])
    print("\nStudy-difficulty cutpoints")
    print(results["study_result"].cutpoints)

    print_model_summary("Utility-choice block: conditional logit", results["utility_result"])

    print("\nChoice probabilities, first rows")
    print(results["choice_probabilities"].head(12))
