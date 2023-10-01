# importing important libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 1 importing dataset
# df = pd.read_csv("https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv")
# df.to_csv("C://Users//BIKASH//Desktop/Data.csv",index=False)
df = pd.read_csv("C://Users//BIKASH//Desktop/Data.csv")
df = pd.DataFrame(df)
df.head()
# 2. High Level Data Understanding:
# a. Find no. of rows & columns in the dataset
# b. Data types of columns.
# c. Info & describe of data in dataframe.
# a. Find no. of rows & columns in the dataset
df_1= df.copy()
print(f'No. of rows in dataset : {df_1.shape[0]}') # no. of rows
print(f'\nNo. of columns in dataset : {df_1.shape[1]}') # no. of columns
# b. Data types of columns.
print(f'Datatype of columns\n{df_1.dtypes}')
# solution2
for i in df_1.columns:
    print (f"{i} = {np.dtype(df_1[i])}\n")
# c. Info & describe of data in dataframe.
print(f'Info of the Dataset{df_1.info()}')
print(f'\nDescription of the Dataset\n{df_1.describe(include="all")}')
# 3. Low Level Data Understanding :
# a. Find count of unique values in location column.
# b. Find which continent has maximum frequency using values_counts.
# c. Find maximum & mean value in 'total_cases'.
# d. Find 25%,50% & 75% quartile value in 'total_deaths'.
# e. Find which continent has maximum 'human_development_index'.
# f. Find which continent has minimum 'gdp_per_capita'

# a. Find count of unique values in location column.
df_3 = df.copy()
print(f"count of uniquie values in location column is {len(df_3['location'].value_counts())}")

# b. Find which continent has maximum frequency using values counts.
print(f"\ncontinent having maximum frequency is {df_3['continent'].value_counts().idxmax()}")

# c. Find maximum & mean value in 'total_cases'.
print('the maximum value in total cases is',df_3['total_cases'].max())
print('\nthe mean value in total_cases is',round(df_3['total_cases'].mean(),2))

# d. Find 25%,50% & 75% quartile value in 'total_deaths'.
print(df_3['total_deaths'].quantile([0.25,0.5,0.75]))
# Solution2
print(f"25% of total_death_values : {df_3['total_deaths'].quantile(q=.25)}")
print(f"50% of total_death_values : {df_3['total_deaths'].quantile(q=.5)}")
print(f"75% of total_death_values : {df_3['total_deaths'].quantile(q=.75)}")

# e.Find which continent has maximum'human_development_index'.
print(df_3.groupby('continent')['human_development_index'].max().idxmax())
# solution2
print(df_3.groupby('continent')['human_development_index'].max().sort_values().tail(1))
# solution3
print(df_3.groupby('continent')['human_development_index'].max().sort_values(ascending=False).head(1))

# f. Find which continent has minimum 'gdp_per_capita'
print(df_3.groupby('continent')['gdp_per_capita'].min().idxmin())
# solution2
print(df_3.groupby('continent')['gdp_per_capita'].min().sort_values().head(1))
# solution3
print(df_3.groupby('continent')['gdp_per_capita'].min().sort_values(ascending=False).tail(1))

# 4. Filter the dataframe with only this columns
# ['continent','location','date','total_cases','total_deaths''gdp_per_capita','human_development_index'] 
# and update the data frame.

df_4 = df.copy()
# filtering values where continrnt is asia and total_death_cases greater than 5000
print(df_4[(df_4['continent']=='Asia')&(df_4['total_deaths']>5000)])

# filtering data where gdp_per_capita is greater than 1500 and human_development_index is maximum
print(df_4[(df_4['gdp_per_capita']>1500) & (df_4['human_development_index']==max(df_4['human_development_index']))])

# filtering data where continents are asia and africa with notnull values and location starts with C.
print (df_4.query("continent.isin(['Asia','Africa']).notnull() and location.str.startswith('C')"))

# filtering dataset where total_cases equals to total_deaths
print(df_4[df_4['total_cases']==df_4['total_deaths']])
print(df_4.sort_values(by=['continent','location','total_deaths']).nsmallest(5,['human_development_index']))

# 5. Data Cleaning
# a. Remove all duplicates observations
# b. Find missing values in all columns
# c. Remove all observations where continent column value is missing
# Tip : using subset parameter in dropna
# d. Fill all missing values with 0

# a. Remove all duplicates observations
df_5 = df.copy()
df_5=df_5.drop_duplicates()
print(df_5)

# b. Find missing values in all columns
df_missing = df_5.copy()
print(f"missing values in all columns\n{df_missing.isna().sum()}")
# solution2
print(f"missing values in all columns\n{df_missing.isnull().sum()}")

# c. Remove all observations where continent column value is missing
df_remove_dup = df_5.copy()
df_remove_dup = df_remove_dup.dropna(how='all',subset='continent')
print(f"after removing all observations where continent column value is missing\n{df_remove_dup}")

# d. Fill all missing values with 0
df_fill = df_5.copy()
df_fill=df_fill.fillna(0)
print(f"after filling null values with 0\n{df_fill}")


# 6. Date time format :
# a. Convert date column in datetime format using pandas.to_datetime
# b. Create new column month after extracting month data from date colum

# a. Convert date column in datetime format using pandas.to_datetime
df_date = df_fill.copy()
df_date['date'] = pd.to_datetime(df_date['date'])
print(df_date['date'].dtype)

# b. Create new column month after extracting month data from date column
df_date['month']=df_date['date'].dt.month
print(df_date.head())

# 7. Data Aggregation:
# a. Find max value in all columns using groupby function on 'continent' column
# Tip: use reset_index() after applying groupby
# b. Store the result in a new dataframe named 'df_groupby'.
# (Use df_groupby dataframe for all further analysis)

# a. Find max value in all columns using groupby function on 'continent'
df_agg= df_date.copy()
df_agg = df_agg.groupby(by='continent').max().reset_index()
print(df_agg)

# b. Store the result in a new dataframe named 'df_groupby'.(Use df_groupby dataframe for all further analysis)
df_groupby = df_agg.groupby(by='continent').max().reset_index()
print(df_groupby)

# 8. Feature Engineering :
# a. Create a new feature 'total_deaths_to_total_cases' by ratio of
# 'total_deaths' column to 'total_cases'

# a. Create a new feature 'total_deaths_to_total_cases' by ratio of'total_deaths' column to 'total_cases'
df_new=df_agg.copy()
df_new['total_deaths_to_total_cases']=(df_new['total_deaths']/df_new['total_cases'])
print(df_new.head())

# 9. Data Visualization :
# a. Perform Univariate analysis on 'gdp_per_capita' column by plotting histogram using seaborn dist plot.
# b. Plot a scatter plot of 'total_cases' & 'gdp_per_capita'
# c. Plot Pairplot on df_groupby dataset.
# d. Plot a bar plot of 'continent' column with 'total_cases' .
# Tip : using kind='bar' in seaborn catplot

# a. Perform Univariate analysis on 'gdp_per_capita' column by plotting histogram using seaborn dist plot.
plt.figure(figsize=(16,9))
sns.set()
sns.distplot(x=df_new['gdp_per_capita'],label="GDP_Per_Capita",color="blue", bins=25)
plt.title('Histogram of GDP_Per_Capita')
plt.legend()
plt.show()

# b. Plot a scatter plot of 'total_cases' & 'gdp_per_capita'
plt.figure(figsize=(16,9))
sns.set()
sns.scatterplot(x='total_cases', y='gdp_per_capita',s=200, c='g', data=df_new)
plt.xlabel('Total Cases',)
plt.ylabel('GDP_Per_Capita')
plt.title('Total Cases Upon GDP_Per_Capita')
plt.show()

# c. Plot Pairplot on df_groupby dataset.
sns.pairplot(df_groupby,kind='reg', 
vars=['total_deaths','gdp_per_capita','total_cases','new_cases',],)
plt.show()

# d. Plot a bar plot of 'continent' column with 'total_cases' .
sns.set()
plt.figure(figsize=(16,9))
sns.barplot(x='continent',y='total_cases',data=df_new)
plt.title('Total Cases on Different Continent')
plt.show()
# solution2
sns.set()
sns.catplot( x='continent', y='total_cases', kind='bar',data=df_new,)
plt.xlabel('Cotinent')
plt.ylabel('Total Cases')
plt.title('Bar Plot on Cotinent and Total Cases')
plt.show()

# 10.Save the df_groupby dataframe in your local drive using pandas.to_csvfunction .
df_groupby.to_csv("C://Users//BIKASH//Desktop/Data2.csv",index=False)