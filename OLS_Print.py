#%% Imports
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


#%% Load Dataset

DATA_PATH = Path("../Data/data4_25years.csv")
data = pd.read_csv(DATA_PATH)
print(f"Number of rows: {len(data):,}")

#%% Keep and rename variables
rename = {
    "timelon": "hourly_wage",
    "alder": "age",
    "koen": "gender",
    "f_udd": "f_edu",
    "e_udd": "e_edu",
    "f_udd_far15": "f_edu_dad15",
    "f_udd_mor15": "f_edu_mom15",
    "persamlinknetrent_ny_far15": "income_dad15",
    "persamlinknetrent_ny_mor15": "income_mom15",
    "arledgr_far15": "unemp_dad15",
    "arledgr_mor15": "unemp_mom15",
}

data = data.rename(columns=rename).copy()
data.head(10)
print(f"Number of rows: {len(data):,}")

#%% Transform variables
numeric_cols = [
    "hourly_wage",
    "age",
    "exp",
    "income_dad15",
    "income_mom15",
    "unemp_dad15",
    "unemp_mom15",
]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col])

print(f"Number of rows: {len(data):,}")
#%%
# Transform to categorical values
data["f_edu"] = data["f_edu"].astype("category")
data["f_edu_dad15"] = data["f_edu_dad15"].astype("category")
data["f_edu_mom15"] = data["f_edu_mom15"].astype("category")
data["e_edu"] = data["e_edu"].astype("category")
data["gender"] = data["gender"].astype("category")

#%%
data = data[data["hourly_wage"] > 0].copy()
data["log_hourly_wage"] = np.log(data["hourly_wage"])
data["log1p_income_dad15"] = np.log1p(data["income_dad15"].clip(lower=0))
data["log1p_income_mom15"] = np.log1p(data["income_mom15"].clip(lower=0))
data["age_sq"] = data["age"] ** 2
data["exp_sq"] = data["exp"] ** 2

print(f"Rows after wage and variable cleaning: {len(data):,}")


#%% Run OLS regression models with Education level as categorical regresor
model1_formula_singelReg = (
    "log_hourly_wage ~ age + age_sq + exp + exp_sq "
    "+ gender + e_edu"
)

model2_formula_singelReg = (
    model1_formula_singelReg
    + " + f_edu_dad15 + f_edu_mom15"
)

model3_formula_singelReg = (
    model2_formula_singelReg
    + " + unemp_dad15 + unemp_mom15 + log1p_income_dad15 + log1p_income_mom15"
)

model1_singReg = smf.ols(model1_formula_singelReg, data=data, missing="drop").fit() 
model2_singReg = smf.ols(model2_formula_singelReg, data=data, missing="drop").fit()
model3_singReg = smf.ols(model3_formula_singelReg, data=data, missing="drop").fit()


#%% Print results in the interactive window
print("\n==================== MODEL 1 ====================")
print(model1_singReg.summary())

print("\n==================== MODEL 2 ====================")
print(model2_singReg.summary())

print("\n==================== MODEL 3 ====================")
print(model3_singReg.summary())


#%% Model comparison
comparison_singleReg = pd.DataFrame(
    {
        "model": ["Model 1", "Model 2", "Model 3"],
        "nobs": [model1_singReg.nobs, model2_singReg.nobs, model3_singReg.nobs],
        "r_squared": [model1_singReg.rsquared, model2_singReg.rsquared, model3_singReg.rsquared],
        "adj_r_squared": [model1_singReg.rsquared_adj, model2_singReg.rsquared_adj, model3_singReg.rsquared_adj],
    }
)

print("\n==================== MODEL COMPARISON ====================")
print(comparison_singleReg)


#%% Prepare specifications of OLS regression models for each education level separately

# Split the data into four subsets based on education level
data_edu1 = data[data["e_edu"] == 1].copy()
data_edu2 = data[data["e_edu"] == 2].copy()
data_edu3 = data[data["e_edu"] == 3].copy()
data_edu4 = data[data["e_edu"] == 4].copy()

model1_formula = (
    "log_hourly_wage ~ age + age_sq + exp + exp_sq "
    "+ gender"
)

model2_formula = (
    model1_formula
    + " + f_edu_dad15 + f_edu_mom15"
)

model3_formula = (
    model2_formula
    + " + unemp_dad15 + unemp_mom15 + log1p_income_dad15 + log1p_income_mom15"
)

#%% Run OLS regression models for education level 1 separately
model1_edu1 = smf.ols(model1_formula, data=data_edu1, missing="drop").fit() 
model2_edu1 = smf.ols(model2_formula, data=data_edu1, missing="drop").fit()
model3_edu1 = smf.ols(model3_formula, data=data_edu1, missing="drop").fit()

#%% Run OLS regression models for education level 2 separately
model1_edu2 = smf.ols(model1_formula, data=data_edu2, missing="drop").fit() 
model2_edu2 = smf.ols(model2_formula, data=data_edu2, missing="drop").fit()
model3_edu2 = smf.ols(model3_formula, data=data_edu2, missing="drop").fit()

#%% Run OLS regression models for education level 3 separately
model1_edu3 = smf.ols(model1_formula, data=data_edu3, missing="drop").fit() 
model2_edu3 = smf.ols(model2_formula, data=data_edu3, missing="drop").fit()
model3_edu3 = smf.ols(model3_formula, data=data_edu3, missing="drop").fit()

#%% Run OLS regression models for education level 4 separately
model1_edu4 = smf.ols(model1_formula, data=data_edu4, missing="drop").fit() 
model2_edu4 = smf.ols(model2_formula, data=data_edu4, missing="drop").fit()
model3_edu4 = smf.ols(model3_formula, data=data_edu4, missing="drop").fit()

#%% Print results of model for education level 1 in the interactive window
print("\n==================== MODEL 1 - Education Level 1 ====================")
print(model1_edu1.summary())

print("\n==================== MODEL 2 - Education Level 1 ====================")
print(model2_edu1.summary())

print("\n==================== MODEL 3 - Education Level 1 ====================")
print(model3_edu1.summary())


#%% Model comparison
comparison_edu1 = pd.DataFrame(
    {
        "Education Level": [1, 1, 1],
        "model": ["Model 1", "Model 2", "Model 3"],
        "nobs": [model1_edu1.nobs, model2_edu1.nobs, model3_edu1.nobs],
        "r_squared": [model1_edu1.rsquared, model2_edu1.rsquared, model3_edu1.rsquared],
        "adj_r_squared": [model1_edu1.rsquared_adj, model2_edu1.rsquared_adj, model3_edu1.rsquared_adj],
    }
)

print("\n==================== MODEL COMPARISON ====================")
print(comparison_edu1)

#%% Print results of model for education level 2 in the interactive window
print("\n==================== MODEL 1 - Education Level 2 ====================")
print(model1_edu2.summary())

print("\n==================== MODEL 2 - Education Level 2 ====================")
print(model2_edu2.summary())

print("\n==================== MODEL 3 - Education Level 2 ====================")
print(model3_edu2.summary())


#%% Model comparison
comparison_edu2 = pd.DataFrame(
    {
        "Education Level": [2, 2, 2],
        "model": ["Model 1", "Model 2", "Model 3"],
        "nobs": [model1_edu2.nobs, model2_edu2.nobs, model3_edu2.nobs],
        "r_squared": [model1_edu2.rsquared, model2_edu2.rsquared, model3_edu2.rsquared],
        "adj_r_squared": [model1_edu2.rsquared_adj, model2_edu2.rsquared_adj, model3_edu2.rsquared_adj],
    }
)

print("\n==================== MODEL COMPARISON ====================")
print(comparison_edu2)

#%% Print results of model for education level 3 in the interactive window
print("\n==================== MODEL 1 - Education Level 3 ====================")
print(model1_edu3.summary())

print("\n==================== MODEL 2 - Education Level 3 ====================")
print(model2_edu3.summary())

print("\n==================== MODEL 3 - Education Level 3 ====================")
print(model3_edu3.summary())


#%% Model comparison
comparison_edu3 = pd.DataFrame(
    {
        "Education Level": [3, 3, 3],
        "model": ["Model 1", "Model 2", "Model 3"],
        "nobs": [model1_edu3.nobs, model2_edu3.nobs, model3_edu3.nobs],
        "r_squared": [model1_edu3.rsquared, model2_edu3.rsquared, model3_edu3.rsquared],
        "adj_r_squared": [model1_edu3.rsquared_adj, model2_edu3.rsquared_adj, model3_edu3.rsquared_adj],
    }
)

print("\n==================== MODEL COMPARISON ====================")
print(comparison_edu3)

# %%
#%% Print results of model for education level 4 in the interactive window
print("\n==================== MODEL 1 - Education Level 4 ====================")
print(model1_edu4.summary())

print("\n==================== MODEL 2 - Education Level 4 ====================")
print(model2_edu4.summary())

print("\n==================== MODEL 3 - Education Level 4 ====================")
print(model3_edu4.summary())


#%% Model comparison
comparison_edu4 = pd.DataFrame(
    {
        "Education Level": [4, 4, 4],
        "model": ["Model 1", "Model 2", "Model 3"],
        "nobs": [model1_edu4.nobs, model2_edu4.nobs, model3_edu4.nobs],
        "r_squared": [model1_edu4.rsquared, model2_edu4.rsquared, model3_edu4.rsquared],
        "adj_r_squared": [model1_edu4.rsquared_adj, model2_edu4.rsquared_adj, model3_edu4.rsquared_adj],
    }
)

print("\n==================== MODEL COMPARISON ====================")
print(comparison_edu4)