#%%
import pandas as pd
import numpy as np
#%%
# load dataset
data = pd.read_stata('/Users/christoffer/Desktop/Uni/8. Semester/Micro and Macro Models in the Labour Market/MMM Project/Data/pnr_sample_updated.dta')

# %%
# Keep relevant columns
columns_to_keep = [
    'pnr',
    'koen',
    'c_antboernf',
    'arledgr',
    'arledgr_far15',
    'arledgr_mor15',
    'f_udd_far15',
    'f_udd_mor15',
    'persamlinknetrent_ny_far15',
    'persamlinknetrent_ny_mor15',
    'exp',
    'induagg',
    'f_udd',
    'i_udd',
    'd9',
    'm9',
    'timelon',
    'aar',
    'alder'
]

data_reduced = data[columns_to_keep].copy()

data_reduced['alder'] = data_reduced['alder'].replace('', np.nan)
data_reduced['alder'] = pd.to_numeric(data_reduced['alder'], errors='coerce')

# Drop rows with missing values in all columns except i_udd, d9, and m9
cols_to_check = [col for col in columns_to_keep if col not in ('i_udd', 'd9', 'm9')]
data_reduced = data_reduced.dropna(subset=cols_to_check)

# Remove rows with time_lon <= 0
data_reduced = data_reduced[data_reduced['timelon'] > 0].copy()
data_reduced['alder'] = data_reduced['alder'].astype(int)

# Create dataset for last year (2010) and age 18 and above
data_lastyear = data_reduced[(data_reduced['aar'] == 2010) & (data_reduced['alder'] >= 18)]


#%%
# Dataset 1 --> Broad dataset assuming F_udd = 1 is vocational educated workers
data1_ColNames = [
    'pnr',
    'koen',
    'arledgr',
    'arledgr_far15',
    'arledgr_mor15',
    'f_udd_far15',
    'f_udd_mor15',
    'persamlinknetrent_ny_far15',
    'persamlinknetrent_ny_mor15',
    'exp',
    'induagg',
    'f_udd',
    'i_udd',
    'timelon',
    'aar',
    'alder'
]

# Remove all irrelevant columns
data1 = data_lastyear[data1_ColNames].copy()

# Remove rows with missing values (without removing rows with missing values in 'i_udd')
cols = [col for col in data1.columns if col != 'i_udd']
data1 = data1.dropna(subset=cols)

# Create new column "Expected education level" (e_udd) as i_udd is seen as "completed"
data1['e_udd'] = data1['i_udd'].fillna(data1['f_udd'])

print(data1['f_udd'].value_counts())
print(data1['induagg'].value_counts())

# %%
# Dataset 2 --> Restricted dataset seperating vocational educated workers by work-industry

# Create dataset 2
data2 = data1.copy()

# The industries which we remove as the share of vocational workers are below 70%
special_induagg = [7, 8, 10, 12]

# Rules for new categorized column
conditions = [
    data2["f_udd"] == 1,
    (data2["f_udd"] == 2) & (data2["induagg"].isin(special_induagg)),
    (data2["f_udd"] == 2) & (~data2["induagg"].isin(special_induagg)),
    data2["f_udd"] == 3,
    data2["f_udd"] == 4
]

# Levels for new categorized column
choices = [0, 1, 2, 3, 4]

# Create new column using above code
data2["f_udd_t"] = np.select(conditions, choices, default=np.nan)

print(data2['f_udd'].value_counts())

# %%
# Dataset 3 --> Restricted dataset seperating vocational educated workers by work-industry
## And removing workers which have education level 2 and is not vocational workers

# Create dataset 3
data3 = data1.copy()

# Remove workers with education level 2 and is not vocational workers
data3 = data3[~((data3["f_udd"] == 2) & (~data3["induagg"].isin(special_induagg)))]

print(data3['f_udd'].value_counts())

#%%
# Dataset 4 --> Unrestricted but with minimum age of 26
alder = 25
data4 = data1.copy()
data4 = data4[data4['alder'] >= alder]

print(f"Age: {alder} and above")
print(f"Number of observations: {len(data4)}")
# For education level "f_udd"
print(data4['f_udd'].value_counts())
dist_f = data4['f_udd'].value_counts().sort_index()
print(dist_f / dist_f.sum())

print(data4['e_udd'].value_counts())
dist_e = data4['e_udd'].value_counts().sort_index()
print(dist_e / dist_e.sum())

print(data4['arledgr'].value_counts())

print(data4['induagg'].value_counts())
data5 = data4[data4['arledgr'] > 0].copy()

#%%
# Create new column "Expected education level" (e_udd) as i_udd is seen as "completed"
data1['e_udd'] = data1['i_udd'].fillna(data1['f_udd'])
data2['e_udd'] = data2['i_udd'].fillna(data2['f_udd'])
data3['e_udd'] = data3['i_udd'].fillna(data3['f_udd'])
data4['e_udd'] = data4['i_udd'].fillna(data4['f_udd'])
# Count number of observations in each category of e_udd
print(data1['e_udd'].value_counts())
print(data2['e_udd'].value_counts())
print(data3['e_udd'].value_counts())
print(data4['e_udd'].value_counts())

#%%
# Save datasets as CSV-files
#data1.to_csv('/Users/christoffer/Desktop/Uni/8. Semester/Micro and Macro Models in the Labour Market/MMM Project/Data/data1_unrestricted.csv', index=False)
#data2.to_csv('/Users/christoffer/Desktop/Uni/8. Semester/Micro and Macro Models in the Labour Market/MMM Project/Data/data2_restricted.csv', index=False)
#data3.to_csv('/Users/christoffer/Desktop/Uni/8. Semester/Micro and Macro Models in the Labour Market/MMM Project/Data/data3_restricted_removed_NonVocationalWorkers.csv', index=False)
#data4.to_csv('/Users/christoffer/Desktop/Uni/8. Semester/Micro and Macro Models in the Labour Market/MMM Project/Data/data4_25years_PT2.csv', index=False)
# %%
data1.corr()

