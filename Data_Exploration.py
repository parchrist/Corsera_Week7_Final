#!/usr/bin/env python
# coding: utf-8

# # Boston Dataset 

# ### Headers of the data set 

# CRIM - per capita crime rate by town
# 
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# INDUS - proportion of non-retail business acres per town.
# 
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# 
# NOX - nitric oxides concentration (parts per 10 million)
# 
# RM - average number of rooms per dwelling
# 
# AGE - proportion of owner-occupied units built prior to 1940
# 
# DIS - weighted distances to five Boston employment centres
# 
# RAD - index of accessibility to radial highways
# 
# TAX - full-value property-tax rate per $10,000
# 
# PTRATIO - pupil-teacher ratio by town
# 
# LSTAT - % lower status of the population
# 
# MEDV - Median value of owner-occupied homes in $1000's

# In[60]:


import csv 
import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from scipy.stats import norm


# In[61]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
df = pd.read_csv(boston_url)
df.head()


# ## Median Value of Owner occupied homes 

# In[58]:


median_value=df['MEDV'].median()
print('The medain value of the data set is...'), print(median_value)

df.boxplot(column='MEDV')
plt.show()


# ## Provide a  bar plot for the Charles river variable

# In[27]:


counts = df['CHAS'].value_counts()
counts.plot.bar(title='Frequency of CHAS')
plt.xlabel('CHAS values')
plt.ylabel('Frequency')

plt.show()
print(counts)


# ## Provide a boxplot for the MEDV variable vs the AGE variable. (Discretize the age variable into three groups of 35 years and younger, between 35 and 70 years and 70 years and older)

# In[38]:


bins = [0, 35, 70, np.inf]
labels = ['35 years and younger', 'between 35 and 70 years', '70 years and older']
df['age_group'] = pd.cut(df['AGE'], bins=bins, labels=labels, include_lowest=True)

# Box plot Creation
plt.figure(figsize=(7,5))
sns.boxplot(x='age_group', y='MEDV', data=df)
plt.title('Boxplot of MEDV by Age Group')
plt.xlabel('Age Group')
plt.ylabel('MEDV')

plt.show()


# ## Provide a scatter plot to show the relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town. What can you say about the relationship?

# In[41]:


sns.regplot(x='INDUS', y='NOX', data=df)
plt.title('Relationship between Nitric Oxide Concentrations and Non-Retail Business Acres')
plt.xlabel('Proportion of Non-Retail Business Acres per Town')
plt.ylabel('Nitric Oxide Concentrations')
slope, intercept = np.polyfit(df['INDUS'], df['NOX'], 1)
equation = 'y = {:.2f}x + {:.2f}'.format(slope, intercept)
plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')

plt.show()


# ## Create a histogram for the pupil to teacher ratio variable

# In[42]:


plt.hist(df['PTRATIO'], bins=10)
plt.title('Distribution of Pupil to Teacher Ratio')
plt.xlabel('Pupil to Teacher Ratio')
plt.ylabel('Frequency')
plt.show()


# In[44]:


plt.hist(df['PTRATIO'], bins=10, density=True)
plt.title('Distribution of Pupil to Teacher Ratio')
plt.xlabel('Pupil to Teacher Ratio')
plt.ylabel('Density')
mu, std = df['PTRATIO'].mean(), df['PTRATIO'].std()
x = np.linspace(mu - 3*std, mu + 3*std, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

plt.show()


# # My answers to Questions

# ### Question 1 

# In[59]:


from scipy.stats import ttest_ind

# separate the MEDV values into two groups based on the CHAS values
medv_bound = df.loc[df['CHAS'] == 1, 'MEDV']
medv_not_bound = df.loc[df['CHAS'] == 0, 'MEDV']

# run the t-test
t_stat, p_val = ttest_ind(medv_bound, medv_not_bound)

# print the results
print('t-statistic: {:.2f}'.format(t_stat))
print('p-value: {:.4f}'.format(p_val))


# From this data I can see that the t-stat shows the diffrence between the two. I think that the value of the home is positively effected from the location to the river. Thus meaning, the closer the home is to the river, the higher the value the home has.  

# ### Question 2 

# In[62]:


from scipy.stats import f_oneway

age_groups = []
for i in range(3):
    age_group = df.loc[(df['AGE'] >= i*35) & (df['AGE'] < (i+1)*35), 'MEDV']
    age_groups.append(age_group)
f_stat, p_val = f_oneway(age_groups[0], age_groups[1], age_groups[2])
print('F-statistic: {:.2f}'.format(f_stat))
print('p-value: {:.4f}'.format(p_val))


# Based off of the evidence presented from running the test, I can reject the null hypothesis, and I can say with confidence that there is a relationship between homes that are older than 1940, and their value compared to the other age group of homes. 

# ### Question 3 

# In[64]:


from scipy.stats import pearsonr

sns.regplot(x='INDUS', y='NOX', data=df)
plt.title('Relationship between Nitric Oxide Concentrations and Non-Retail Business Acres')
plt.xlabel('Proportion of Non-Retail Business Acres per Town')
plt.ylabel('Nitric Oxide Concentrations')

# calculate the Pearson correlation coefficient and p-value
corr_coef, p_val = pearsonr(df['INDUS'], df['NOX'])

# add the correlation coefficient to the plot
corr_str = 'r = {:.2f}, p = {:.4f}'.format(corr_coef, p_val)
plt.annotate(corr_str, xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, ha='left', va='top')

plt.show()


# When looking at the pearson correlation, we can clearly see that the correlation shows a positive relationship, 0.76. the correlation bewtween the Nitric Oxide Concentrations and non-retail business acres. The 'P' Value shows that there is a dependency that the relationship did not happen alone, and there is something else effecting the relationship between the two. In conclusion there is a relationship between the two. 

# ### Question 4 

# In[65]:


import statsmodels.api as sm

X = df['DIS']
y = df['MEDV']
X = sm.add_constant(X) 
model = sm.OLS(y, X).fit()

print(model.summary())


# When looking at the model summary, the "dis" coef is showing and suggesting that theire is an increase in the value of the homes with the distance of the homes. In simple terms, the further the home, the more the home will cost. All of the data within the summary shows that there is a very high likely hood that the hypthosis in the Dis-coef is correct, and all of the values point that way and verify our statistical discovery. 

# In[ ]:




