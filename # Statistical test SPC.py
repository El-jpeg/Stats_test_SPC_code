# Statistical test SPC 

## 



# Install packages 

%pip install quantecon
%pip install inequalipy

# import libraries 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import quantecon as qe 
import inequalipy as ineq
import seaborn as sns 

# Load data 

# Load data 

individuals = pd.read_csv('/content/individuals.csv')

households = pd.read_csv('/content/households.csv')

households.info()

merged_df = pd.merge(individuals, households, on='hid', how='inner')

merged_df.info()

 merged_df.head()

# * Number of people and number of households in the population and in the sample

# Number of people in the dataset 
num_ppl_dataset = len(individuals)

# Number of households in the dataset 
num_hh_dataset = len(households)

print(f'Number of people in the sample: {num_ppl_dataset}')
print(f'Number of households in the sample: {num_hh_dataset}')

# Number of people in the population 
num_ppl_pop = merged_df['final_weight'].sum()

# Number of households in the population
num_hh_pop = households['final_weight'].sum()


print(f'Estimated number of people in the population: {num_ppl_pop:,.2f}')
print(f'Estimated number of households in the population: {num_hh_pop:,.2f}')


# * Number of people in the population living in rural and in urban areas

rururb_pop = merged_df.groupby('rururb')['final_weight'].sum()

print("Number of people in the population living in rural and urban areas:")
for area, pop in rururb_pop.items():
    print(f"{area}: {round(pop,2)}")

# * Number of people in the population and average household size living in female-headed and male-headed
# households

# Number of people by household head sex (weighted)
people_by_hhhead_sex = merged_df.groupby('hhhead_sex')['final_weight'].sum()
print("Estimated number of people in female-headed and male-headed households:")
print(people_by_hhhead_sex)

# Average household size by household head sex (weighted)

household_weights_by_hhhead_sex = households.groupby('hhhead_sex')['final_weight'].sum()

hh_by_hhhead_sex = people_by_hhhead_sex / household_weights_by_hhhead_sex

print("\nEstimated average household size in female-headed and male-headed households:")
print(hh_by_hhhead_sex)


# * Mean per person income (not grouping by households) for the country as a whole

weighted_mean_income = (merged_df['income'] * merged_df['final_weight']).sum() / merged_df['final_weight'].sum()

print(f"Estimated mean per person income for the country as a whole: {round(weighted_mean_income,2)}")

# * Mean and median income by rural/urban

income_rururb = merged_df.groupby('rururb')['income'].agg(['mean', 'median'])
print(round(income_rururb,2))

# * Bonus question - confidence interval for the mean and median income by rural and urban status

# Calculate confidence intervals for the mean and median income by rural and urban status

from scipy import stats

# Make sure data is numeric and remove missing values
rural_income = merged_df[merged_df['rururb'] == 'Rural']['income'].dropna()
urban_income = merged_df[merged_df['rururb'] == 'Urban']['income'].dropna()

# Function to calculate confidence interval for the mean
def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = stats.sem(data)  # standard error of the mean
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m - h, m + h

# Calculate confidence intervals
rural_mean_ci = mean_confidence_interval(rural_income)
urban_mean_ci = mean_confidence_interval(urban_income)

print(f"95% Confidence Interval for Rural Income Mean: {rural_mean_ci}")
print(f"95% Confidence Interval for Urban Income Mean: {urban_mean_ci}")


# Group by age and sex
age_sex = merged_df.groupby(['age', 'sex'])['final_weight'].sum().unstack(fill_value=0)

# Plot
age_sex.plot(kind='bar', figsize=(12, 5))

plt.xlabel('Age')
plt.ylabel('Population')
plt.title('Population by Age and Sex')
plt.tight_layout()
plt.show()