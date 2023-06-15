# 1- drop id column
# 2- convert categorical to numeric
# 3- move target variable to the end

import pandas as pd

df = pd.read_csv('breast-cancer.csv')

# 1- drop id column:
df.drop(df.columns[[0]], axis=1, inplace=True)

# 2- replacing values in diagnosis
df['diagnosis'].replace(['M', 'B'], [0, 1], inplace=True)

# 3- move target variable to the end
target = df.pop('diagnosis')
df.insert(len(df.columns), 'diagnosis', target)

# new dataframe
result = df
# create new csv file with new dataframe
result.to_csv(r'bc.csv', index = False, header=True)
