---
title: "Feature Engineering 1 - Missing values - Numerical"
date: 2021-04-05
tags: [data science, Feature Engineering, messy data]
header:
  image: "/images/Feature_engineering.jpeg"
excerpt: "Data Science, Feature Engineering, Messy Data"
mathjax: "true"
---

## Missing Values - Feature Engineering

Lifecycle of Data Science projects
1. Data Collection Strategy
    - from company side
    - 3rd party API's
    - Surveys
2. Featue Engineering
    - handling missing values

why are there missing values?
    - In case of surveys, people might have not filled the values
    - some input error while uploading numbers into system

* In Data Science projects -> Data should be collected from multiple sources

* Types of data that might be missings:-
    - Continuous data
        - Discrete continuous data (e.g age which will have integral value)
        - Continuous data (e.g. height)
    - categorial data



#### What are different types of Missing data?

1. #### Missing completely at random (MCAR)

    - a variable is missing completely at random (MCAR) if the probability of being missing is same for all the observations. When data is MCAR, there is absolutely no relationship between the data missing and any other values,observed or missing, within dataset. In other words, those missing data points are a random subset of the data. There is nothing systematic going on that makes some data more likely to be missing than other.
    if values for observation are missing completely at random, then disregarding those cases would not bias the inferences made.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df=pd.read_csv('Datasets/Titanic/train.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isna().sum() #Number of missing values in each columns
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



* We can sense that there might be relationship between age and Cabin missing values as the person whose age is unknown cabin seems unknown for them, or person having null values in Age and Cabin may not have survived.

* Missing values of Embarked seems MCAR.


```python
df[df.Embarked.isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>61</td>
      <td>62</td>
      <td>1</td>
      <td>1</td>
      <td>Icard, Miss. Amelie</td>
      <td>female</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>829</td>
      <td>830</td>
      <td>1</td>
      <td>1</td>
      <td>Stone, Mrs. George Nelson (Martha Evelyn)</td>
      <td>female</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



2. #### Missing data not at random (MNAR) : Systematic missing value
    - There is absolutely some relationship between the data missing and any other values, observed or missing, within the dataset.


```python
df[df.Cabin.isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>884</td>
      <td>885</td>
      <td>0</td>
      <td>3</td>
      <td>Sutehall, Mr. Henry Jr</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392076</td>
      <td>7.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>885</td>
      <td>886</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Mrs. William (Margaret Norton)</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <td>886</td>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>888</td>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>890</td>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>687 rows × 12 columns</p>
</div>




```python
#df['cabin_null']=df['Cabin'].apply(lambda x: 0 if x.isna() else 1)
df['cabin_null']=np.where(df.Cabin.isna(),1,0)
```


```python
df[['Cabin','cabin_null']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cabin</th>
      <th>cabin_null</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>C85</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>C123</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#find percentage of null values
df['cabin_null'].mean() #Alternate -> df.Cabin.isna().mean()
```




    0.7710437710437711




```python
df.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'cabin_null'],
          dtype='object')



Trying to figure out relationship of Cabin missing value with Survived column as we expect there should be less missing values for survived person


```python
df.groupby('Survived').mean().cabin_null
```




    Survived
    0    0.876138
    1    0.602339
    Name: cabin_null, dtype: float64



3. #### Missing at Random (MAR)
    - The probability of the missing values with respect to various features -> this will be almost same in the whole dataset.
    - When the msising values have no relation with any other column but have relation with the column in which the value is missing
    - e.g. Men hiding their salary, women hiding their age. (Kind of trend being analysed but not dependent to other variable)


#### All the techniques of handling missing values :-

1. Mean/Median/Mode replacement
2. Random Sample Imputation
3. Capturing NAN values with a new feature
4. End of Distribution imputation
5. Arbitrary imputation
6. Frequent categories imputation


# 1. Mean/Median/Mode imputation

When should we apply this?
- Mean/median imputation has the assumption that the data are missing completely at random (MCAR).
- Solve this by replacing the NAN with most frequent occurence of the variable.
- To overcome outlier we can use media/mode instead of mean.


```python
df1=pd.read_csv('Datasets/Titanic/train.csv',usecols=['Age','Fare','Survived'])
```


```python
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Lets go and see the percentage of missing values
df1.isna().mean()
```




    Survived    0.000000
    Age         0.198653
    Fare        0.000000
    dtype: float64




```python
def impute_nan(df,variable,method):
    if method=='median':
        value=df[variable].median()
    elif method=='mean':
        value=df[variable].mean()
    elif method=='mode':
        value=df[variable].mode()
    df[variable+"_"+method]=df[variable].fillna(value)
```


```python
impute_nan(df1,'Age','median')
```


```python
impute_nan(df1,'Age','mode')
```


```python
impute_nan(df1,'Age','mean')
```


```python
df1[df1.Age.isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Age_median</th>
      <th>Age_mode</th>
      <th>Age_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>8.4583</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <td>17</td>
      <td>1</td>
      <td>NaN</td>
      <td>13.0000</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <td>19</td>
      <td>1</td>
      <td>NaN</td>
      <td>7.2250</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0</td>
      <td>NaN</td>
      <td>7.2250</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <td>28</td>
      <td>1</td>
      <td>NaN</td>
      <td>7.8792</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>859</td>
      <td>0</td>
      <td>NaN</td>
      <td>7.2292</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <td>863</td>
      <td>0</td>
      <td>NaN</td>
      <td>69.5500</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <td>868</td>
      <td>0</td>
      <td>NaN</td>
      <td>9.5000</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <td>878</td>
      <td>0</td>
      <td>NaN</td>
      <td>7.8958</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>NaN</td>
      <td>23.4500</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>29.699118</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 6 columns</p>
</div>



* Now we can check whether the standard deviation of age column changed if we compare the same with age_median


```python
print('Age StdDev :-',df1.Age.std())
print('Age_median StdDev :-',df1.Age_median.std())
```

    Age StdDev :- 14.526497332334044
    Age_median StdDev :- 13.019696550973194


* Now we can check distributiojn of Age_median w.r.t. Age column using matplotlib


```python
fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111)
df1['Age'].plot(kind='kde',ax=ax)
df1.Age_median.plot(kind='kde',ax=ax,color='red')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')
```




    <matplotlib.legend.Legend at 0x7f98f26f7f10>



<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_32_1.png" alt="linearly separable data">



### Advantages and disadvantages of Mean/Median imputations

#### Advantages :-
    1. Easy to implement.
    2. Robust to outliers (Median only).
    3. Faster way to obtain complete datasets.
#### Disadvantages :-
    1. Change or distortion in the original variance/standard deviation.
    2. It impacts correlation

# 2. Random Sample Imputation

Aim: Random sample imputation consists of taking random observation from the dataset and we use this observation to replace NAN value.

When shout it be used?

It assumes that data are missing completely at random (MCAR).


```python
df2=pd.read_csv('Datasets/Titanic/train.csv',usecols=['Survived','Age','Fare'])
```


```python
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.isnull().sum()
```




    Survived      0
    Age         177
    Fare          0
    dtype: int64




```python
df2.isnull().mean()
```




    Survived    0.000000
    Age         0.198653
    Fare        0.000000
    dtype: float64




```python
df['Age'].dropna().sample(df.Age.isna().sum(),random_state=0)
```




    423    28.00
    177    50.00
    305     0.92
    292    36.00
    889    26.00
           ...  
    539    22.00
    267    25.00
    352    15.00
    99     34.00
    689    15.00
    Name: Age, Length: 177, dtype: float64




```python
def impute_nan_rand(df,variable,method):
    if method!='random':
        if method=='median':
            value=df[variable].median()
        elif method=='mean':
            value=df[variable].mean()
        elif method=='mode':
            value=df[variable].mode()
        df[variable+"_"+method]=df[variable].fillna(value)
    else:
        ## Just to itroduce variable_random column which will be updated later
        df[variable+"_random"]=df[variable]
        ## it will have the random sample to fill NAN value
        random_sample=df['Age'].dropna().sample(df.Age.isna().sum(),random_state=0)
        ## pandas need to have same index in order to merge the datasets
        random_sample.index=df[df[variable].isna()].index
        #updating variable_random column with random sample for NAN values
        df.loc[df[variable].isna(),variable+'_random']=random_sample

```


```python
impute_nan_rand(df2,'Age','random')
```


```python
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Age_random</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>38.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>26.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>35.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>35.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>0</td>
      <td>27.0</td>
      <td>13.0000</td>
      <td>27.0</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>19.0</td>
      <td>30.0000</td>
      <td>19.0</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>NaN</td>
      <td>23.4500</td>
      <td>15.0</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>26.0</td>
      <td>30.0000</td>
      <td>26.0</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>32.0</td>
      <td>7.7500</td>
      <td>32.0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 4 columns</p>
</div>




```python
print('Age StdDev :-',df2.Age.std())
print('Age_random StdDev :-',df2.Age_random.std())
```

    Age StdDev :- 14.526497332334044
    Age_random StdDev :- 14.5636540895687



```python
fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111)
df2['Age'].plot(kind='kde',ax=ax)
df2.Age_random.plot(kind='kde',ax=ax,color='red')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')
```




    <matplotlib.legend.Legend at 0x7f98f3f8da10>



<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_44_1.png" alt="linearly separable data">




```python
impute_nan_rand(df2,'Age','median')
```


```python
fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111)
df2['Age'].plot(kind='kde',ax=ax)
df2.Age_random.plot(kind='kde',ax=ax,color='red')
df2.Age_median.plot(kind='kde',ax=ax,color='green')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')
```




    <matplotlib.legend.Legend at 0x7f98f410bd50>



<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_46_1.png" alt="linearly separable data">




```python
impute_nan_rand(df2,'Age','mean')
```


```python
fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111)
df2['Age'].plot(kind='kde',ax=ax)
df2.Age_random.plot(kind='kde',ax=ax,color='red')
df2.Age_median.plot(kind='kde',ax=ax,color='green')
df2.Age_mean.plot(kind='kde',ax=ax,color='orange')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')
```




    <matplotlib.legend.Legend at 0x7f98f40e2310>




<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_48_1.png" alt="linearly separable data">


#### Advantages
1. Easy to implement
2. Less distortion in variance or standard deviation

#### Disadvantages
1. In every situation randomness won't work

# 3. Capturing NAN values with a new feature

It works well if the data are missing completely not at random (MNAR).


```python
df3=pd.read_csv('Datasets/Titanic/train.csv',usecols=['Survived','Age','Fare'])
```


```python
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Just to flag the missing value, from this we can identify which row has been imputed from missing value
df3['Age_NAN']=np.where(df3.Age.isna(),1,0)
```


```python
df3.Age_NAN
```




    0      0
    1      0
    2      0
    3      0
    4      0
          ..
    886    0
    887    0
    888    1
    889    0
    890    0
    Name: Age_NAN, Length: 891, dtype: int64




```python
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Age_NAN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>0</td>
      <td>27.0</td>
      <td>13.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>19.0</td>
      <td>30.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>NaN</td>
      <td>23.4500</td>
      <td>1</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>26.0</td>
      <td>30.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>32.0</td>
      <td>7.7500</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 4 columns</p>
</div>




```python
df3['Age'].fillna(df3.Age.median(),inplace=True)
```


```python
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Age_NAN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>0</td>
      <td>27.0</td>
      <td>13.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>19.0</td>
      <td>30.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>28.0</td>
      <td>23.4500</td>
      <td>1</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>26.0</td>
      <td>30.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>32.0</td>
      <td>7.7500</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 4 columns</p>
</div>



### Advantages
1. Easy to implement
2. Captures the importance of missing values. (Age_NAN column giving importance to missing value as it is 1 for NAN cases)

### Disadvantages
1. Creating additional features, as number of features which have missing values increases we need to create same number of additional features. (Curse of Dimensionality)

# 4. End of Distribution imputation

If there is suspicion that the missing value is not at random then capturing that information is important. In this scenario, one would want to replace missing data with values that are at the tails of the distribution of the variable.


```python
df4=pd.read_csv('Datasets/Titanic/train.csv',usecols=['Survived','Age','Fare'])
df4.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4.Age.hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f98f3887750>




<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_61_1.png" alt="linearly separable data">



```python
#Mean
df4.Age.mean()
```




    29.69911764705882




```python
#Standard deviation
df4.Age.std()
```




    14.526497332334044




```python
sns.boxplot('Age',data=df4)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f98f4e01ed0>




<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_64_1.png" alt="linearly separable data">


+ In this dataset we have only right side outliers, therefore we will impute the missing values from roght extremem values


```python
#Picking values from right end of distribution -> mean+3*standard_deviation
df4.Age.mean()+3*df4.Age.std()
```




    73.27860964406095




```python
def impute_nan_extreme(df,variable,method):
    if method=='random':
        ## Just to itroduce variable_random column which will be updated later
        df[variable+"_random"]=df[variable]
        ## it will have the random sample to fill NAN value
        random_sample=df['Age'].dropna().sample(df.Age.isna().sum(),random_state=0)
        ## pandas need to have same index in order to merge the datasets
        random_sample.index=df[df[variable].isna()].index
        #updating variable_random column with random sample for NAN values
        df.loc[df[variable].isna(),variable+'_random']=random_sample
    elif method=='extreme':
        #getting the right extreme value (mean+3*std_dev)
        value=df[variable].mean()+3*df[variable].std()
        df[variable+"_end_distribution"]=df[variable].fillna(value)
    else:
        if method=='median':
            value=df[variable].median()
        elif method=='mean':
            value=df[variable].mean()
        elif method=='mode':
            value=df[variable].mode()
        df[variable+"_"+method]=df[variable].fillna(value)


```


```python
impute_nan_extreme(df4,'Age','extreme')
```


```python
df4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Age_end_distribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>22.00000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>38.00000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>26.00000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>35.00000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>35.00000</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>0</td>
      <td>27.0</td>
      <td>13.0000</td>
      <td>27.00000</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>19.0</td>
      <td>30.0000</td>
      <td>19.00000</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>NaN</td>
      <td>23.4500</td>
      <td>73.27861</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>26.0</td>
      <td>30.0000</td>
      <td>26.00000</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>32.0</td>
      <td>7.7500</td>
      <td>32.00000</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 4 columns</p>
</div>




```python
impute_nan_extreme(df4,'Age','median')
```


```python
df4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Age_end_distribution</th>
      <th>Age_median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>22.00000</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>38.00000</td>
      <td>38.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>26.00000</td>
      <td>26.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>35.00000</td>
      <td>35.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>35.00000</td>
      <td>35.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>0</td>
      <td>27.0</td>
      <td>13.0000</td>
      <td>27.00000</td>
      <td>27.0</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>19.0</td>
      <td>30.0000</td>
      <td>19.00000</td>
      <td>19.0</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>NaN</td>
      <td>23.4500</td>
      <td>73.27861</td>
      <td>28.0</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>26.0</td>
      <td>30.0000</td>
      <td>26.00000</td>
      <td>26.0</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>32.0</td>
      <td>7.7500</td>
      <td>32.00000</td>
      <td>32.0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 5 columns</p>
</div>




```python
fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111)
df4['Age'].plot(kind='kde',ax=ax)
df4.Age_end_distribution.plot(kind='kde',ax=ax,color='red')
df4.Age_median.plot(kind='kde',ax=ax,color='green')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')
```




    <matplotlib.legend.Legend at 0x7f98f4ee6350>




<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_72_1.png" alt="linearly separable data">



```python
df4.Age.hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f98f5ebfc10>




<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_73_1.png" alt="linearly separable data">



```python
df4.Age_median.hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f98f58e5f50>




<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_74_1.png" alt="linearly separable data">



```python
df4.Age_end_distribution.hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f98f5a84f10>




<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_75_1.png" alt="linearly separable data">



```python
sns.boxplot('Age_end_distribution',data=df4)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f98f6042190>




<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_76_1.png" alt="linearly separable data">


* we can see that there is no more outliers after end of distribution imputation.
* We can observe skewness in the distribution after imputation.

### Advantages
1. Easy to implement
2. Captures the importance of missing values if there is one

### Disadvantages
1. Distorts the original distribution of the variable
2. If missingness is not important, it may mask the predictive power of the original variable by distorting its distribution.
3. If number of NAN is big, it will mask true outliers in the distribution
4. If the number of NAN is small, the replaced NAN may be considered as outlier and pre-processed in subsequent steps of feature engineering

##### Note
- if we are working in real world environment only one type of feature engineering will not be sufficient, we need to show all the observation/scenarios to the stakeholders.

# 5. Arbitrary imputation

* It consists of replacing NAN by an arbitrary value
* This technique was derived from kaggle competetions



```python
df5=pd.read_csv('Datasets/Titanic/train.csv',usecols=['Survived','Age','Fare'])
df5.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>




```python
df5.Age.hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f98f63dc250>




<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 1 - Missing Values - Numerical_files/Feature Engineering 1 - Missing Values - Numerical_82_1.png" alt="linearly separable data">


##### Arbitrary values :-
- It should not be the more frequently present in data
- we can either use right end (or > right end) or left end (or < left end)


```python
def impute_arbitrary(df,variable):
    df[variable+"_rightend"]=df[variable].fillna(df[variable].max()+20)
    df[variable+"_leftend"]=df[variable].fillna(df[variable].min())
```


```python
impute_arbitrary(df5,'Age')
```


```python
df5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Age_rightend</th>
      <th>Age_leftend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>22.0</td>
      <td>22.00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>38.0</td>
      <td>38.00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>26.0</td>
      <td>26.00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>35.0</td>
      <td>35.00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>35.0</td>
      <td>35.00</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>0</td>
      <td>27.0</td>
      <td>13.0000</td>
      <td>27.0</td>
      <td>27.00</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>19.0</td>
      <td>30.0000</td>
      <td>19.0</td>
      <td>19.00</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>NaN</td>
      <td>23.4500</td>
      <td>100.0</td>
      <td>0.42</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>26.0</td>
      <td>30.0000</td>
      <td>26.0</td>
      <td>26.00</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>32.0</td>
      <td>7.7500</td>
      <td>32.0</td>
      <td>32.00</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 5 columns</p>
</div>



### Advantages
1. Easy to implement
2. Captures the importance of missing values if there is one

### Disadvantages
1. Distorts the original distribution of the variable
2. If missingness is not important, it may mask the predictive power of the original variable by distorting its distribution.
3. Hard to decide which value to use


```python

```
