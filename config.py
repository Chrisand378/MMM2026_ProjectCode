#%%
from pathlib import Path


#%%
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"

# Place your future individual-level CSV here.
INDIVIDUAL_DATA_PATH = DATA_DIR / "data4_25years_PT2.csv"

# Excel workbook containing the named table LaborMarketEntry_data.
LABOR_MARKET_ENTRY_PATH = DATA_DIR / "UDDAKT10_med_vaegtet_gennemsnit.xlsx"
LABOR_MARKET_ENTRY_TABLE = "LaborMarketEntry_data"


#%%
# Core variable names in the individual-level dataset.
EDUCATION_COL = "e_udd"
MODEL_EDUCATION_COL = "education_level"
WAGE_COL = "timelon"
LOG_WAGE_COL = "log_timelon"
AGE_COL = "alder"
EXPERIENCE_COL = "exp"

# Background variables used in the model.
CATEGORICAL_BACKGROUND_COLS = [
    "koen",
    "f_udd_far15",
    "f_udd_mor15",
]

NUMERIC_BACKGROUND_COLS = [
    "arledgr_far15",
    "arledgr_mor15",
]

LOG_BACKGROUND_COLS = [
    "persamlinknetrent_ny_far15",
    "persamlinknetrent_ny_mor15",
]

LOG_BACKGROUND_PREFIX = "log1p_"

BACKGROUND_COLS = (
    CATEGORICAL_BACKGROUND_COLS
    + NUMERIC_BACKGROUND_COLS
    + LOG_BACKGROUND_COLS
)


#%%
# Model assumptions
RETIREMENT_AGE = 70
DISCOUNT_RATE = 0.015
ANNUAL_HOURS = 160.33 * 12  # 160.33 hours per month * 12 months

# Unemployment risk by broad education level
UNEMPLOYMENT_RATES = {
    1: 0.0467,
    2: 0.0261,
    3: 0.0268,
    4: 0.0312,
}

# Ordered broad education levels in the choice set.
EDUCATION_LEVELS = [1, 2, 3, 4]

# %%
