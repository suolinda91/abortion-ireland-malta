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

ees2009_df = df[(df['country']== 'Ireland') | (df['country']== 'Malta')]

# missing values
NaN_ireland_sum = ees2009_df[ees2009_df['country']== 'Ireland'].isnull().sum()
NaN_malta_sum = ees2009_df[ees2009_df['country']== 'Malta'].isnull().sum()

print('== Missing values in Ireland ==')
print(NaN_ireland_sum)
print('Total amount of rows: %s' % ees2009_df[ees2009_df['country']== 'Ireland'].shape[0])
print('\n')
print('== Missing values in Malta ==')
print(NaN_malta_sum)
print('Total amount of rows: %s' % ees2009_df[ees2009_df['country']== 'Malta'].shape[0])
print('\n')

## Univariat analysis
univariat_ireland = ees2009_df[ees2009_df['country']== 'Ireland'].describe()
univariat_malta = ees2009_df[ees2009_df['country']== 'Malta'].describe()

# Boxplot Political Position
sns.set(style="whitegrid")
PoliticalPosition_ax = sns.boxplot(x='PoliticalPosition', y='country', data=ees2009_df, palette='pastel')
PoliticalPosition_ax.set(xlabel='Political Position on a Left-Right Scale', ylabel='Country', title='Self positioning on the political spectrum in Ireland and Malta')
plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ('Left', 1, 2, 3, 4, 5, 6, 7, 8, 9, 'Right'))

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

# Level of Level Of Spirituality
LevelOfSpirituality_ax = sns.boxplot(x='LevelOfSpirituality', y='country', data=ees2009_df, palette='pastel')
LevelOfSpirituality_ax.set(xlabel='Self declaring the level of religiousness', ylabel='Country', title='Self declaring the level of religiousness in Ireland and Malta')
plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ('Not at all religious', 1, 2, 3, 4, 5, 6, 7, 8, 9, 'Very religious'), rotation=90)


# Test 1st hypothesis
corrtab_Ireland = pd.crosstab(ees2009_df[ees2009_df['country'] == 'Ireland']['OpinionAbortion'], ees2009_df[ees2009_df['country'] == 'Ireland']['gender'])
chi2_1hypothesis_ireland, p_1hypothesis_ireland, dof_1hypothesis_ireland, ex_1hypothesis_ireland = stats.chi2_contingency(corrtab_Ireland)

corrtab_Malta = pd.crosstab(ees2009_df[ees2009_df['country'] == 'Malta']['OpinionAbortion'], ees2009_df[ees2009_df['country'] == 'Malta']['gender'])
chi2_1hypothesis_malta, p_1hypothesis_malta, dof_1hypothesis_malta, ex_1hypothesis_malta = stats.chi2_contingency(corrtab_Malta)

# Test 2nd hypothesis
hypothesis2_ireland = ees2009_df[ees2009_df['country'] == 'Ireland'][['OpinionAbortion', 'FemaleEmployment']].dropna()
hypothesis2_malta = ees2009_df[ees2009_df['country'] == 'Malta'][['OpinionAbortion', 'FemaleEmployment']].dropna()

rho_2hypothesis_ireland, p_2hypothesis_ireland = stats.spearmanr(hypothesis2_ireland['OpinionAbortion'],hypothesis2_ireland['FemaleEmployment'])
rho_2hypothesis_malta, p_2hypothesis_malta = stats.spearmanr(hypothesis2_malta['OpinionAbortion'],hypothesis2_malta['FemaleEmployment'])
#print(ees2009_ireland['OpinionAbortion'].corr(ees2009_ireland['FemaleEmployment'], method='spearman'))

# Test 3rd hypothesis
hypothesis3_ireland = ees2009_df[ees2009_df['country'] == 'Ireland'][['OpinionAbortion', 'PoliticalPosition']].dropna()
hypothesis3_malta = ees2009_df[ees2009_df['country'] == 'Malta'][['OpinionAbortion', 'PoliticalPosition']].dropna()
rho_3hypothesis_ireland, p_3hypothesis_ireland = stats.spearmanr(hypothesis3_ireland['OpinionAbortion'],hypothesis3_ireland['PoliticalPosition'])
rho_3hypothesis_malta, p_3hypothesis_malta = stats.spearmanr(hypothesis3_malta['OpinionAbortion'],hypothesis3_malta['PoliticalPosition'])

#Test 4th hypothesis
hypothesis4_ireland = ees2009_df[ees2009_df['country'] == 'Ireland'][['OpinionAbortion', 'LevelOfSpirituality']].dropna()
hypothesis4_malta = ees2009_df[ees2009_df['country'] == 'Malta'][['OpinionAbortion', 'LevelOfSpirituality']].dropna()
rho_4hypothesis_ireland, p_4hypothesis_ireland = stats.spearmanr(hypothesis4_ireland['LevelOfSpirituality'], hypothesis4_ireland['OpinionAbortion'])
rho_4hypothesis_malta, p_4hypothesis_malta = stats.spearmanr(hypothesis4_malta['OpinionAbortion'],hypothesis4_malta['LevelOfSpirituality'])

print(p_4hypothesis_ireland)
print('Test')
