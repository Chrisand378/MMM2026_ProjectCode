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

data_reduced = data[columns_to_keep].copy()

data_reduced['alder'] = data_reduced['alder'].replace('', np.nan)
data_reduced['alder'] = pd.to_numeric(data_reduced['alder'], errors='coerce')

# Drop rows with missing values in all columns except i_udd
cols_to_check = [col for col in columns_to_keep if col not in ('i_udd')]
data_reduced = data_reduced.dropna(subset=cols_to_check)

# Remove rows with time_lon <= 0
data_reduced = data_reduced[data_reduced['timelon'] > 0].copy()
data_reduced['alder'] = data_reduced['alder'].astype(int)

# Create dataset for the last year (2010) and age above 18
data_lastyear = data_reduced[(data_reduced['aar'] == 2010) & (data_reduced['alder'] >= 18)]

#%%
# Create Dataset --> All aged 25 or above and merging f_udd and i_udd
alder = 25

data = data_lastyear[data_lastyear['alder']>=alder].copy()

# Create new column "Expected education level" (e_udd) as i_udd is seen as "completed"
data['e_udd'] = data['i_udd'].fillna(data['f_udd'])

# Save dataset:
#data.to_csv('/Users/christoffer/Desktop/Uni/8. Semester/Micro and Macro Models in the Labour Market/MMM Project/Data/data.csv', index=False)


#%%
# Print simple results
print(f"Number of observations: {len(data)}")

# For education level "f_udd"
print(data['f_udd'].value_counts())
dist_f = data['f_udd'].value_counts().sort_index()
print(dist_f / dist_f.sum())

# For education level "e_udd"
print(data['e_udd'].value_counts())
dist_e = data['e_udd'].value_counts().sort_index()
print(dist_e / dist_e.sum())

# %%
