```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

# How to handle categorial missing values


```python
df=pd.read_csv('Datasets/House_prices/train.csv')
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
df.columns
```




    Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
           'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
           'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
           'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
           'SaleCondition', 'SalePrice'],
          dtype='object')




```python
df[df.columns[df.isna().sum()>0]].isna().mean().sort_values()
```




    Electrical      0.000685
    MasVnrType      0.005479
    MasVnrArea      0.005479
    BsmtQual        0.025342
    BsmtCond        0.025342
    BsmtFinType1    0.025342
    BsmtExposure    0.026027
    BsmtFinType2    0.026027
    GarageCond      0.055479
    GarageQual      0.055479
    GarageFinish    0.055479
    GarageType      0.055479
    GarageYrBlt     0.055479
    LotFrontage     0.177397
    FireplaceQu     0.472603
    Fence           0.807534
    Alley           0.937671
    MiscFeature     0.963014
    PoolQC          0.995205
    dtype: float64



## 1. Frequent categories imputation


```python
df1=pd.read_csv('Datasets/House_prices/train.csv',usecols=['BsmtQual','FireplaceQu','GarageType','SalePrice'])
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
      <th>BsmtQual</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Gd</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>208500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>181500</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>223500</td>
    </tr>
    <tr>
      <td>3</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>140000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.isna().sum()
```




    BsmtQual        37
    FireplaceQu    690
    GarageType      81
    SalePrice        0
    dtype: int64




```python
df1.isna().mean().sort_values(ascending=True)
```




    SalePrice      0.000000
    BsmtQual       0.025342
    GarageType     0.055479
    FireplaceQu    0.472603
    dtype: float64




```python
df1.shape
```




    (1460, 4)



* since there are less number of missing values in BsmtQual and GarageType we can relace them with most frequent category as it will not distort its relationship with another variable or the distribution


```python
# Computing the frequency with every feature
```


```python
df1.BsmtQual.value_counts()
#df.groupby(['BsmtQual'])['BsmtQual'].count()
```




    TA    649
    Gd    618
    Ex    121
    Fa     35
    Name: BsmtQual, dtype: int64




```python
df1.BsmtQual.value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe8df97a750>




![png](Feature%20Engineering%202%20-%20Missing%20Values%20-%20Categorial_files/Feature%20Engineering%202%20-%20Missing%20Values%20-%20Categorial_14_1.png)



```python
df1.GarageType.value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe8dfaa4310>




![png](Feature%20Engineering%202%20-%20Missing%20Values%20-%20Categorial_files/Feature%20Engineering%202%20-%20Missing%20Values%20-%20Categorial_15_1.png)



```python
df1.FireplaceQu.value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe8de336810>




![png](Feature%20Engineering%202%20-%20Missing%20Values%20-%20Categorial_files/Feature%20Engineering%202%20-%20Missing%20Values%20-%20Categorial_16_1.png)



```python
df1.GarageType.mode()[0]
```




    'Attchd'




```python
df1.GarageType.value_counts().index[0]
```




    'Attchd'




```python
def impute_nan(df,variable):
    most_freq_cat=df[variable].value_counts().index[0] #most_freq_cat=df[variable].mode()[0]
    df[variable].fillna(most_freq_cat,inplace=True)
    
```


```python
for features in ['BsmtQual','GarageType','FireplaceQu']:
    impute_nan(df1,features)
```


```python
df1
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
      <th>BsmtQual</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>208500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>181500</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>223500</td>
    </tr>
    <tr>
      <td>3</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>140000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>250000</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1455</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>175000</td>
    </tr>
    <tr>
      <td>1456</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>210000</td>
    </tr>
    <tr>
      <td>1457</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>266500</td>
    </tr>
    <tr>
      <td>1458</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>142125</td>
    </tr>
    <tr>
      <td>1459</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>147500</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 4 columns</p>
</div>




```python
df1.isna().sum()
```




    BsmtQual       0
    FireplaceQu    0
    GarageType     0
    SalePrice      0
    dtype: int64




```python
df1.FireplaceQu.value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe8de7459d0>




![png](Feature%20Engineering%202%20-%20Missing%20Values%20-%20Categorial_files/Feature%20Engineering%202%20-%20Missing%20Values%20-%20Categorial_23_1.png)


* As FireplaceQu has large missing values (~47%), imputing the same with mode might distort the relationship of FireplaceQu with other variable. So it is not a good practice to impute with mode in case where there is higher percentage of missing values

### Advantages
1. Easy and faster way to implement

### Disadvantages
1. Since we are using more frequent label, it may use them in an over-represented way if there are many NANs.
2. It distorts the relation of the most frequent label.

## 2. Adding a variable to capture NAN


```python
df2=pd.read_csv('Datasets/House_prices/train.csv',usecols=['BsmtQual','FireplaceQu','GarageType','SalePrice'])
```


```python
df2['BsmtQual_NAN']=np.where(df2.BsmtQual.isna(),1,0)
```


```python
impute_nan(df2,'BsmtQual')
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
      <th>BsmtQual</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>SalePrice</th>
      <th>BsmtQual_NAN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Gd</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>208500</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>181500</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>223500</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>140000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>250000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



* we are capturing the importance of missing value (here by using a variable+_NAN column) and also imputing the column with mode (most frequnent category)
* Since we are capturing the importance of missing values, we can also apply this to the column having higher percentage of missing value (i.e. FireplaceQu)


```python
df2['FireplaceQu_NAN']=np.where(df2.FireplaceQu.isna(),1,0)
```


```python
impute_nan(df2,'FireplaceQu')
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
      <th>BsmtQual</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>SalePrice</th>
      <th>BsmtQual_NAN</th>
      <th>FireplaceQu_NAN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>208500</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>181500</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>223500</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>140000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>250000</td>
      <td>0</td>
      <td>0</td>
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
      <td>1455</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>175000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1456</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>210000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1457</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>266500</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1458</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>142125</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1459</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>147500</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 6 columns</p>
</div>



## 3. Replace NAN with a new category

we use it if we have more frequent categories


```python

```


```python
df3=pd.read_csv('Datasets/House_prices/train.csv',usecols=['BsmtQual','FireplaceQu','GarageType','SalePrice'])
```


```python
def impute_nan_new_cat(df,variable):
    df[variable+"_newvar"]=df[variable].fillna('Missing_'+variable)

```


```python
for features in ['BsmtQual','GarageType','FireplaceQu']:
    impute_nan_new_cat(df3,features)
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
      <th>BsmtQual</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>SalePrice</th>
      <th>BsmtQual_newvar</th>
      <th>GarageType_newvar</th>
      <th>FireplaceQu_newvar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Gd</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>208500</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>Missing_FireplaceQu</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>181500</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>TA</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>223500</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>TA</td>
    </tr>
    <tr>
      <td>3</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>140000</td>
      <td>TA</td>
      <td>Detchd</td>
      <td>Gd</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>250000</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>TA</td>
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
    </tr>
    <tr>
      <td>1455</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>175000</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>TA</td>
    </tr>
    <tr>
      <td>1456</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>210000</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>TA</td>
    </tr>
    <tr>
      <td>1457</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>266500</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>Gd</td>
    </tr>
    <tr>
      <td>1458</td>
      <td>TA</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>142125</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>Missing_FireplaceQu</td>
    </tr>
    <tr>
      <td>1459</td>
      <td>TA</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>147500</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>Missing_FireplaceQu</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 7 columns</p>
</div>




```python

```
