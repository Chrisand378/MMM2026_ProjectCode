#%% Imports
from pathlib import Path
import numpy as np  
import pandas as pd


#%% Load Dataset
DATA_PATH = Path("../Data/data4_25years.csv")
data = pd.read_csv(DATA_PATH)
print(f"Number of rows: {len(data):,}")

print(f"Number of rows: {len(data):,}")
print(f"Number of columns: {len(data.columns):,}")


#%% Define categorical and excluded columns
# Categorical columns: koen and every column containing "udd"
categorical_cols = ["koen"] + [col for col in data.columns if "udd" in col.lower()]
categorical_cols = [col for col in categorical_cols if col in data.columns]

# Columns that should not be included in descriptive statistics
exclude_cols = ["pnr", "induagg", "aar"] + categorical_cols

# All remaining columns are treated as numeric
numeric_cols = [col for col in data.columns if col not in exclude_cols]


#%% Descriptive statistics for numeric columns
numeric_data = data[numeric_cols].apply(pd.to_numeric, errors="coerce")

descriptive_statistics = numeric_data.agg(
    ["min", "median", "mean", "max", "std"]
).T

descriptive_statistics = descriptive_statistics.rename(
    columns={
        "min": "Minimum",
        "median": "Median",
        "mean": "Mean",
        "max": "Maximum",
        "std": "Standard deviation",
    }
)

print("\n==================== DESCRIPTIVE STATISTICS ====================")
print(descriptive_statistics)


#%% Distributions for categorical columns
for col in categorical_cols:
    print(f"\n==================== DISTRIBUTION: {col} ====================")

    distribution = data[col].value_counts(dropna=False).sort_index()
    distribution = pd.DataFrame(
        {
            "Count": distribution,
            "Percent": (distribution / len(data) * 100).round(2),
        }
    )

    print(distribution)



#%% Descriptive statistics for categorical columns

categorical_tables = []

for col in categorical_cols:
    counts = data[col].value_counts(dropna=False).sort_index()
    percents = (counts / len(data) * 100).round(2)

    table = pd.DataFrame({
        "Variable": col,
        "Category": counts.index.astype(str),
        "Count": counts.values,
        "Percent": percents.values
    })

    # Rename missing values nicely
    table["Category"] = table["Category"].replace("nan", "Missing")

    categorical_tables.append(table)

categorical_statistics = pd.concat(categorical_tables, ignore_index=True)

print("\n==================== CATEGORICAL DESCRIPTIVE STATISTICS ====================")
print(categorical_statistics)