import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

missing_values = [88, 77]
df = pd.read_csv('./data/ees2009.csv', sep=";", na_values = missing_values)
df.columns =['PoliticalPosition', 'OpinionAbortion', 'FemaleEmployment', 'gender', 'religion', 'LevelOfSpirituality', 'country']


df.loc[df['country']== 1372, 'country'] = 'Ireland'
df.loc[df['country']== 1470, 'country'] = 'Malta'
df.loc[df['OpinionAbortion']>5, 'OpinionAbortion'] = None
df.loc[df['FemaleEmployment']>5, 'FemaleEmployment'] = None
df.loc[df['gender']==7, 'gender'] = None

ees2009_df = df[(df['country']== 1372) | (df['country']== 1470)]
#ireland_df = df[df['country']== 1372]
#malta_df = df[df['country']== 1470]

# missing values
NaN_ireland_sum = ees2009_df[ees2009_df['country']== 'Ireland'].isnull().sum()
NaN_malta_sum = ees2009_df[ees2009_df['country']== 'Malta'].isnull().sum()

## Univariat analysis
univariat_ireland = ees2009_df[ees2009_df['country']== 'Ireland'].describe()
univariat_malta = ees2009_df[ees2009_df['country']== 'Malta'].describe()

# Boxplot Political Position
sns.set(style="whitegrid")
sns.boxplot(x='PoliticalPosition', y='country', data=ees2009_df, palette='pastel')
#plt.set(xlabel='Country', ylabel='Political Position on a Left-Right Scale', title='Political Positons')
plt.xlabel('Self Positioning on a Left-Right Scale')
plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ('Left', 1, 2, 3, 4, 5, 6, 7, 8, 9, 'Right'))
plt.ylabel('Country')
plt.title('Political Positions in Ireland and Malta')
#plt.show()

# Boxplot Opinion on Abortion
sns.set(style="whitegrid")
sns.boxplot(x='OpinionAbortion', y='country', data=ees2009_df, palette='pastel')
plt.xticks((1, 2, 3, 4, 5), ('strongly agree', 2, 'neither agree nor disagree', 4, 'strongly disagree'), rotation=45)
plt.ylabel('Country')
plt.title('Opinion on abortion in Ireland and Malta')
#plt.show()

# Boxplot Opinion on female employment
sns.set(style="whitegrid")
sns.boxplot(x='FemaleEmployment', y='country', data=ees2009_df, palette='pastel')
plt.xlabel('Women should be prepared to cut down on paid work for the sake of their family.')
plt.xticks((1, 2, 3, 4, 5), ('strongly agree', 2, 'neither agree nor disagree', 4, 'strongly disagree'))
plt.ylabel('Country')
plt.title('Opinion on female employment in Ireland and Malta')
#plt.show()

# Barplot gender
ees2009_gender = ees2009_df.groupby('country')['gender'].value_counts(normalize=True).reset_index(name='genderPercentage')
ees2009_gender['genderPercentage'] = ees2009_gender['genderPercentage'] * 100
GenderOfInterviewed = sns.catplot(x='gender', y ='genderPercentage', col='country',data=ees2009_gender, palette='pastel', kind='bar')
(GenderOfInterviewed.set_axis_labels('Gender', 'Percentage').set_xticklabels(['male', 'female']).set_titles('{col_name}').despine(top=True, right=True))
#plt.show()

# Barplot religion
ees2009_religion = ees2009_df.groupby('country')['religion'].value_counts(normalize=True).reset_index(name='religionPercentage')
ees2009_religion['religionPercentage'] = ees2009_religion['religionPercentage'] * 100
ReligionOfInterviewed = sns.catplot(x='religion', y ='religionPercentage', row='country',data=ees2009_religion, palette='pastel', kind='bar')
(ReligionOfInterviewed.set_axis_labels('Religion', 'Percentage').set_xticklabels(['No denomination', 'R.Catholic', 'Protestant', 'Orthodox', 'Jew', 'Muslim', 'Hinud', 'Buddhist', 'Church of England'], rotation=45).set_titles('{row_name}').despine(top=True, right=True))
#plt.show()

# Level of Level Of Spirituality
LevelOfSpirituality_ax = sns.boxplot(x='LevelOfSpirituality', y='country', data=ees2009_df, palette='pastel')
LevelOfSpirituality_ax.set(xlabel='Self declaring the level of religiousness', ylabel='Country', title='Self declaring the level of religiousness in Ireland and Malta')
plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ('Not at all religious', 1, 2, 3, 4, 5, 6, 7, 8, 9, 'Very religious'), rotation=90)


# Test 1st hypothesis
ees2009_ireland = ees2009_df[ees2009_df['country'] == 'Ireland']
ees2009_malta = ees2009_df[ees2009_df['country'] == 'Malta']

corrtab_Ireland = pd.crosstab(ees2009_ireland['gender'], ees2009_ireland['OpinionAbortion'])
chi2_ireland, p_ireland, dof_ireland, ex_ireland = stats.chi2_contingency(corrtab_Ireland)

corrtab_Malta = pd.crosstab(ees2009_malta['gender'], ees2009_malta['OpinionAbortion'])
chi2_malta, p_malta, dof_malta, ex_malta = stats.chi2_contingency(corrtab_Malta)

# Test 2nd hypothesis
