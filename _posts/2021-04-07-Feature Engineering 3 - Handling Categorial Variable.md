# Handle Categorial Features

## 1. One Hot Encoding


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df1=pd.read_csv('Datasets/Titanic/train.csv',usecols=['Embarked','Sex'])
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
      <th>Sex</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>male</td>
      <td>S</td>
    </tr>
    <tr>
      <td>1</td>
      <td>female</td>
      <td>C</td>
    </tr>
    <tr>
      <td>2</td>
      <td>female</td>
      <td>S</td>
    </tr>
    <tr>
      <td>3</td>
      <td>female</td>
      <td>S</td>
    </tr>
    <tr>
      <td>4</td>
      <td>male</td>
      <td>S</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>male</td>
      <td>S</td>
    </tr>
    <tr>
      <td>887</td>
      <td>female</td>
      <td>S</td>
    </tr>
    <tr>
      <td>888</td>
      <td>female</td>
      <td>S</td>
    </tr>
    <tr>
      <td>889</td>
      <td>male</td>
      <td>C</td>
    </tr>
    <tr>
      <td>890</td>
      <td>male</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 2 columns</p>
</div>




```python
pd.get_dummies(df1).head()
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
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



* Dummy Variable Trap :- We can explain a variable having n category using n-1 column we actually need not to convert into n columns. Above situation where variable has been split into n columns we can drop a column and still be able to get the same information.


```python
pd.get_dummies(df1,drop_first=True)
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
      <th>Sex_male</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>887</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 3 columns</p>
</div>



### Disadvantages
- If we have many categories in a variable, numbers of column required will be large (we should avoid this case)

## One hot Encoding with many categories in a feature

- Take only top n features into account and perform one-hot encoding on the same only


```python
df2=pd.read_csv('Datasets/Mercedes/train.csv',usecols=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6'])
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
      <th>X0</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>k</td>
      <td>v</td>
      <td>at</td>
      <td>a</td>
      <td>d</td>
      <td>u</td>
      <td>j</td>
    </tr>
    <tr>
      <td>1</td>
      <td>k</td>
      <td>t</td>
      <td>av</td>
      <td>e</td>
      <td>d</td>
      <td>y</td>
      <td>l</td>
    </tr>
    <tr>
      <td>2</td>
      <td>az</td>
      <td>w</td>
      <td>n</td>
      <td>c</td>
      <td>d</td>
      <td>x</td>
      <td>j</td>
    </tr>
    <tr>
      <td>3</td>
      <td>az</td>
      <td>t</td>
      <td>n</td>
      <td>f</td>
      <td>d</td>
      <td>x</td>
      <td>l</td>
    </tr>
    <tr>
      <td>4</td>
      <td>az</td>
      <td>v</td>
      <td>n</td>
      <td>f</td>
      <td>d</td>
      <td>h</td>
      <td>d</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.nunique()
```




    X0    47
    X1    27
    X2    44
    X3     7
    X4     4
    X5    29
    X6    12
    dtype: int64




```python
#We are just taking top 10 categories of a variable
df2.X1.value_counts()[:10]
```




    aa    833
    s     598
    b     592
    l     590
    v     408
    r     251
    i     203
    a     143
    c     121
    o      82
    Name: X1, dtype: int64




```python
lst_10=df2.X1.value_counts()[:10].index.tolist()
```


```python
lst_10
```




    ['aa', 's', 'b', 'l', 'v', 'r', 'i', 'a', 'c', 'o']




```python

for columns in df2.columns:
    lst_10=df2[columns].value_counts()[:10].index.tolist()
    for categories in lst_10:
        df2[columns+"_"+categories]=np.where(df2[columns]==categories,1,0)
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
      <th>X0</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X0_z</th>
      <th>X0_ak</th>
      <th>X0_y</th>
      <th>...</th>
      <th>X6_g</th>
      <th>X6_j</th>
      <th>X6_d</th>
      <th>X6_i</th>
      <th>X6_l</th>
      <th>X6_a</th>
      <th>X6_h</th>
      <th>X6_k</th>
      <th>X6_c</th>
      <th>X6_b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>k</td>
      <td>v</td>
      <td>at</td>
      <td>a</td>
      <td>d</td>
      <td>u</td>
      <td>j</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>k</td>
      <td>t</td>
      <td>av</td>
      <td>e</td>
      <td>d</td>
      <td>y</td>
      <td>l</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>az</td>
      <td>w</td>
      <td>n</td>
      <td>c</td>
      <td>d</td>
      <td>x</td>
      <td>j</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>az</td>
      <td>t</td>
      <td>n</td>
      <td>f</td>
      <td>d</td>
      <td>x</td>
      <td>l</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>az</td>
      <td>v</td>
      <td>n</td>
      <td>f</td>
      <td>d</td>
      <td>h</td>
      <td>d</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>4204</td>
      <td>ak</td>
      <td>s</td>
      <td>as</td>
      <td>c</td>
      <td>d</td>
      <td>aa</td>
      <td>d</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4205</td>
      <td>j</td>
      <td>o</td>
      <td>t</td>
      <td>d</td>
      <td>d</td>
      <td>aa</td>
      <td>h</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4206</td>
      <td>ak</td>
      <td>v</td>
      <td>r</td>
      <td>a</td>
      <td>d</td>
      <td>aa</td>
      <td>g</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4207</td>
      <td>al</td>
      <td>r</td>
      <td>e</td>
      <td>f</td>
      <td>d</td>
      <td>aa</td>
      <td>l</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4208</td>
      <td>z</td>
      <td>r</td>
      <td>ae</td>
      <td>c</td>
      <td>d</td>
      <td>aa</td>
      <td>g</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4209 rows × 68 columns</p>
</div>




```python
df2.columns
```




    Index(['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X0_z', 'X0_ak', 'X0_y',
           'X0_ay', 'X0_t', 'X0_x', 'X0_o', 'X0_f', 'X0_n', 'X0_w', 'X1_aa',
           'X1_s', 'X1_b', 'X1_l', 'X1_v', 'X1_r', 'X1_i', 'X1_a', 'X1_c', 'X1_o',
           'X2_as', 'X2_ae', 'X2_ai', 'X2_m', 'X2_ak', 'X2_r', 'X2_n', 'X2_s',
           'X2_f', 'X2_e', 'X3_c', 'X3_f', 'X3_a', 'X3_d', 'X3_g', 'X3_e', 'X3_b',
           'X4_d', 'X4_a', 'X4_b', 'X4_c', 'X5_w', 'X5_v', 'X5_q', 'X5_r', 'X5_d',
           'X5_s', 'X5_n', 'X5_m', 'X5_p', 'X5_i', 'X6_g', 'X6_j', 'X6_d', 'X6_i',
           'X6_l', 'X6_a', 'X6_h', 'X6_k', 'X6_c', 'X6_b'],
          dtype='object')




```python
df2[['X1','X1_aa','X1_s', 'X1_b', 'X1_l', 'X1_v', 'X1_r', 'X1_i', 'X1_a', 'X1_c', 'X1_o']]
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
      <th>X1</th>
      <th>X1_aa</th>
      <th>X1_s</th>
      <th>X1_b</th>
      <th>X1_l</th>
      <th>X1_v</th>
      <th>X1_r</th>
      <th>X1_i</th>
      <th>X1_a</th>
      <th>X1_c</th>
      <th>X1_o</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>v</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>t</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>w</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>t</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>v</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>4204</td>
      <td>s</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4205</td>
      <td>o</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4206</td>
      <td>v</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4207</td>
      <td>r</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4208</td>
      <td>r</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4209 rows × 11 columns</p>
</div>



## 2. Ordinal Number encoding

When the category present in the variable can be ordered based on their character/behaviour.

Example :-
1. Grading system (A,B,C,D,E)
2. User Experience (Excellent,good,average,below average,poor)
3. Weekdays in a week (weekdays-0,weekend-1)


```python
import datetime as dt
```


```python
today_date=dt.datetime.today()
```


```python
#way to get date of n prior days
today_date-dt.timedelta(15)
```




    datetime.datetime(2021, 3, 19, 13, 40, 5, 705343)




```python
#List comprehension
days=[today_date-dt.timedelta(x) for x in range(15)]
```


```python
df3=pd.DataFrame(days)
df3.columns=['Day']
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
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2021-04-03 13:40:05.705343</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2021-04-02 13:40:05.705343</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2021-04-01 13:40:05.705343</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2021-03-31 13:40:05.705343</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2021-03-30 13:40:05.705343</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2021-03-29 13:40:05.705343</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2021-03-28 13:40:05.705343</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2021-03-27 13:40:05.705343</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2021-03-26 13:40:05.705343</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2021-03-25 13:40:05.705343</td>
    </tr>
    <tr>
      <td>10</td>
      <td>2021-03-24 13:40:05.705343</td>
    </tr>
    <tr>
      <td>11</td>
      <td>2021-03-23 13:40:05.705343</td>
    </tr>
    <tr>
      <td>12</td>
      <td>2021-03-22 13:40:05.705343</td>
    </tr>
    <tr>
      <td>13</td>
      <td>2021-03-21 13:40:05.705343</td>
    </tr>
    <tr>
      <td>14</td>
      <td>2021-03-20 13:40:05.705343</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3['weekday']=df3['Day'].dt.weekday_name
```


```python
dictionary={
    'Monday':1,
    'Tuesday':2,
    'Wednesday':3,
    'Thursday':4,
    'Friday':5,
    'Saturday':6,
    'Sunday':7
}
```


```python
df3['weekday_ordinal']=df3['weekday'].map(dictionary)
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
      <th>Day</th>
      <th>weekday</th>
      <th>weekday_ordinal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2021-04-03 13:40:05.705343</td>
      <td>Saturday</td>
      <td>6</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2021-04-02 13:40:05.705343</td>
      <td>Friday</td>
      <td>5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2021-04-01 13:40:05.705343</td>
      <td>Thursday</td>
      <td>4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2021-03-31 13:40:05.705343</td>
      <td>Wednesday</td>
      <td>3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2021-03-30 13:40:05.705343</td>
      <td>Tuesday</td>
      <td>2</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2021-03-29 13:40:05.705343</td>
      <td>Monday</td>
      <td>1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2021-03-28 13:40:05.705343</td>
      <td>Sunday</td>
      <td>7</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2021-03-27 13:40:05.705343</td>
      <td>Saturday</td>
      <td>6</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2021-03-26 13:40:05.705343</td>
      <td>Friday</td>
      <td>5</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2021-03-25 13:40:05.705343</td>
      <td>Thursday</td>
      <td>4</td>
    </tr>
    <tr>
      <td>10</td>
      <td>2021-03-24 13:40:05.705343</td>
      <td>Wednesday</td>
      <td>3</td>
    </tr>
    <tr>
      <td>11</td>
      <td>2021-03-23 13:40:05.705343</td>
      <td>Tuesday</td>
      <td>2</td>
    </tr>
    <tr>
      <td>12</td>
      <td>2021-03-22 13:40:05.705343</td>
      <td>Monday</td>
      <td>1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>2021-03-21 13:40:05.705343</td>
      <td>Sunday</td>
      <td>7</td>
    </tr>
    <tr>
      <td>14</td>
      <td>2021-03-20 13:40:05.705343</td>
      <td>Saturday</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Count or Frequency encoding


```python
train_set=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',header=None,index_col=None)
```


```python
train_set
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <td>1</td>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <td>2</td>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <td>3</td>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <td>4</td>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>32556</td>
      <td>27</td>
      <td>Private</td>
      <td>257302</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <td>32557</td>
      <td>40</td>
      <td>Private</td>
      <td>154374</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <td>32558</td>
      <td>58</td>
      <td>Private</td>
      <td>151910</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <td>32559</td>
      <td>22</td>
      <td>Private</td>
      <td>201490</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <td>32560</td>
      <td>52</td>
      <td>Self-emp-inc</td>
      <td>287927</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
<p>32561 rows × 15 columns</p>
</div>




```python
columns=[1,3,5,6,7,8,9,13]
```


```python
train_set=train_set[columns]
train_set.head()
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
      <th>1</th>
      <th>3</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Self-emp-not-inc</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Private</td>
      <td>11th</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>Cuba</td>
    </tr>
  </tbody>
</table>
</div>




```python
#train_set[1].value_counts()
train_set.columns=['Employment','Degree','Status','Designation','Family_job','Race','Sex','Country']
```


```python
train_set
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
      <th>Employment</th>
      <th>Degree</th>
      <th>Status</th>
      <th>Designation</th>
      <th>Family_job</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Self-emp-not-inc</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Private</td>
      <td>11th</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>Cuba</td>
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
    </tr>
    <tr>
      <td>32556</td>
      <td>Private</td>
      <td>Assoc-acdm</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>32557</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>32558</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>32559</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
    </tr>
    <tr>
      <td>32560</td>
      <td>Self-emp-inc</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
<p>32561 rows × 8 columns</p>
</div>




```python
train_set.nunique()
```




    Employment      9
    Degree         16
    Status          7
    Designation    15
    Family_job      6
    Race            5
    Sex             2
    Country        42
    dtype: int64




```python
train_set['Country'].value_counts().to_dict()
```




    {' United-States': 29170,
     ' Mexico': 643,
     ' ?': 583,
     ' Philippines': 198,
     ' Germany': 137,
     ' Canada': 121,
     ' Puerto-Rico': 114,
     ' El-Salvador': 106,
     ' India': 100,
     ' Cuba': 95,
     ' England': 90,
     ' Jamaica': 81,
     ' South': 80,
     ' China': 75,
     ' Italy': 73,
     ' Dominican-Republic': 70,
     ' Vietnam': 67,
     ' Guatemala': 64,
     ' Japan': 62,
     ' Poland': 60,
     ' Columbia': 59,
     ' Taiwan': 51,
     ' Haiti': 44,
     ' Iran': 43,
     ' Portugal': 37,
     ' Nicaragua': 34,
     ' Peru': 31,
     ' France': 29,
     ' Greece': 29,
     ' Ecuador': 28,
     ' Ireland': 24,
     ' Hong': 20,
     ' Cambodia': 19,
     ' Trinadad&Tobago': 19,
     ' Thailand': 18,
     ' Laos': 18,
     ' Yugoslavia': 16,
     ' Outlying-US(Guam-USVI-etc)': 14,
     ' Honduras': 13,
     ' Hungary': 13,
     ' Scotland': 12,
     ' Holand-Netherlands': 1}




```python
for columns in train_set:
    dic=train_set[columns].value_counts().to_dict()
    train_set[columns+"_encoded"]=train_set[columns].map(dic)
```


```python
train_set
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
      <th>Employment</th>
      <th>Degree</th>
      <th>Status</th>
      <th>Designation</th>
      <th>Family_job</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Country</th>
      <th>Employment_encoded</th>
      <th>Degree_encoded</th>
      <th>Status_encoded</th>
      <th>Designation_encoded</th>
      <th>Family_job_encoded</th>
      <th>Race_encoded</th>
      <th>Sex_encoded</th>
      <th>Country_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>1298</td>
      <td>5355</td>
      <td>10683</td>
      <td>3770</td>
      <td>8305</td>
      <td>27816</td>
      <td>21790</td>
      <td>29170</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Self-emp-not-inc</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>2541</td>
      <td>5355</td>
      <td>14976</td>
      <td>4066</td>
      <td>13193</td>
      <td>27816</td>
      <td>21790</td>
      <td>29170</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>22696</td>
      <td>10501</td>
      <td>4443</td>
      <td>1370</td>
      <td>8305</td>
      <td>27816</td>
      <td>21790</td>
      <td>29170</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Private</td>
      <td>11th</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>United-States</td>
      <td>22696</td>
      <td>1175</td>
      <td>14976</td>
      <td>1370</td>
      <td>13193</td>
      <td>3124</td>
      <td>21790</td>
      <td>29170</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>Cuba</td>
      <td>22696</td>
      <td>5355</td>
      <td>14976</td>
      <td>4140</td>
      <td>1568</td>
      <td>3124</td>
      <td>10771</td>
      <td>95</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>32556</td>
      <td>Private</td>
      <td>Assoc-acdm</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
      <td>22696</td>
      <td>1067</td>
      <td>14976</td>
      <td>928</td>
      <td>1568</td>
      <td>27816</td>
      <td>10771</td>
      <td>29170</td>
    </tr>
    <tr>
      <td>32557</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>22696</td>
      <td>10501</td>
      <td>14976</td>
      <td>2002</td>
      <td>13193</td>
      <td>27816</td>
      <td>21790</td>
      <td>29170</td>
    </tr>
    <tr>
      <td>32558</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
      <td>22696</td>
      <td>10501</td>
      <td>993</td>
      <td>3770</td>
      <td>3446</td>
      <td>27816</td>
      <td>10771</td>
      <td>29170</td>
    </tr>
    <tr>
      <td>32559</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>22696</td>
      <td>10501</td>
      <td>10683</td>
      <td>3770</td>
      <td>5068</td>
      <td>27816</td>
      <td>21790</td>
      <td>29170</td>
    </tr>
    <tr>
      <td>32560</td>
      <td>Self-emp-inc</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
      <td>1116</td>
      <td>10501</td>
      <td>14976</td>
      <td>4066</td>
      <td>1568</td>
      <td>27816</td>
      <td>10771</td>
      <td>29170</td>
    </tr>
  </tbody>
</table>
<p>32561 rows × 16 columns</p>
</div>



#### Advantages
1. Easy to use
2. Not increasing feature space

#### Disadvantages
1. It will provide the same weight if frequency of two categories are same, model will not be able to distinguish between them after encoding.

## 4. Target Guided Ordinal Encoding

1. Ordering the labels according to the target variable
2. Replace the labels by the joint probability of being 1 or 0


```python
df4=pd.read_csv('Datasets/Titanic/train.csv',usecols=['Cabin','Survived'])
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
      <th>Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>C85</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>C123</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>B42</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>C148</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 2 columns</p>
</div>




```python
df4.Cabin.fillna('Missing',inplace=True)
```


```python
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
      <th>Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>Missing</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>C85</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>Missing</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>C123</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>Missing</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Just trying to get cabing group ignoring seat number of the corresponding cabin
df4['Cabin']=df4.Cabin.str[:1]
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
      <th>Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>M</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>C</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>M</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>C</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>M</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>0</td>
      <td>M</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>M</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>C</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 2 columns</p>
</div>




```python
df4.Cabin.unique()
```




    array(['M', 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T'], dtype=object)




```python
#percentage of people survived in each cabin group
df4.groupby(['Cabin']).Survived.mean().sort_values()
```




    Cabin
    T    0.000000
    M    0.299854
    A    0.466667
    G    0.500000
    C    0.593220
    F    0.615385
    B    0.744681
    E    0.750000
    D    0.757576
    Name: Survived, dtype: float64




```python
df4.groupby(['Cabin']).Survived.mean().sort_values().index
```




    Index(['T', 'M', 'A', 'G', 'C', 'F', 'B', 'E', 'D'], dtype='object', name='Cabin')




```python
ordinal_labels=df4.groupby(['Cabin']).Survived.mean().sort_values().index
```


```python
#Dictionary Comprehension
ordinal_labels2={k:i for i,k in enumerate(ordinal_labels,0)}
```


```python
ordinal_labels2
```




    {'T': 0, 'M': 1, 'A': 2, 'G': 3, 'C': 4, 'F': 5, 'B': 6, 'E': 7, 'D': 8}




```python
df4['Cabin_Ordinal']=df4.Cabin.map(ordinal_labels2)
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
      <th>Cabin</th>
      <th>Cabin_Ordinal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>M</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>C</td>
      <td>4</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>M</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>C</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>M</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>0</td>
      <td>M</td>
      <td>1</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>B</td>
      <td>6</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>M</td>
      <td>1</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>C</td>
      <td>4</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>M</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 3 columns</p>
</div>



## 5. Mean Encoding

- Replacing the category by the mean value corresponding to target variable


```python
df5=df4.copy()
```


```python
mean_ordinal=df5.groupby(['Cabin']).Survived.mean().sort_values().to_dict()
```


```python
mean_ordinal
```




    {'T': 0.0,
     'M': 0.29985443959243085,
     'A': 0.4666666666666667,
     'G': 0.5,
     'C': 0.5932203389830508,
     'F': 0.6153846153846154,
     'B': 0.7446808510638298,
     'E': 0.75,
     'D': 0.7575757575757576}




```python
df5['Cabin_mean_ordinal']=df5.Cabin.map(mean_ordinal)
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
      <th>Cabin</th>
      <th>Cabin_Ordinal</th>
      <th>Cabin_mean_ordinal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>M</td>
      <td>1</td>
      <td>0.299854</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>C</td>
      <td>4</td>
      <td>0.593220</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>M</td>
      <td>1</td>
      <td>0.299854</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>C</td>
      <td>4</td>
      <td>0.593220</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>M</td>
      <td>1</td>
      <td>0.299854</td>
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
      <td>M</td>
      <td>1</td>
      <td>0.299854</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>B</td>
      <td>6</td>
      <td>0.744681</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>M</td>
      <td>1</td>
      <td>0.299854</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>C</td>
      <td>4</td>
      <td>0.593220</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>M</td>
      <td>1</td>
      <td>0.299854</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 4 columns</p>
</div>



#### Advantages
1. It captures information within the label therefore rendering more predictive features
2. It creates a monotonic relationship between variable and target

#### Disadvantages
1. Sometimes it leads to overfitting

## 6. Probability Ratio Encoding
- replacing category by odds w.r.t target variable
- Odds = p/(1-p)  where p is probability of target variable in a category


```python
df6=pd.read_csv('Datasets/Titanic/train.csv',usecols=['Cabin','Survived'])
```


```python
df6.head()
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
      <th>Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>C85</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>C123</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Replacing NAN with new category - Missing
df6.Cabin.fillna('Missing',inplace=True)
```


```python
df6.head()
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
      <th>Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>Missing</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>C85</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>Missing</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>C123</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>Missing</td>
    </tr>
  </tbody>
</table>
</div>




```python
df6.Cabin=df6.Cabin.str[:1]
```


```python
df6
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
      <th>Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>M</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>C</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>M</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>C</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>M</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>0</td>
      <td>M</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>M</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>C</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 2 columns</p>
</div>




```python
df6.Cabin.unique()
```




    array(['M', 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T'], dtype=object)




```python
# Probabily of surviving a person in the cabin
prob_df=pd.DataFrame(df6.groupby('Cabin')['Survived'].mean())
```


```python
prob_df
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
    </tr>
    <tr>
      <th>Cabin</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>A</td>
      <td>0.466667</td>
    </tr>
    <tr>
      <td>B</td>
      <td>0.744681</td>
    </tr>
    <tr>
      <td>C</td>
      <td>0.593220</td>
    </tr>
    <tr>
      <td>D</td>
      <td>0.757576</td>
    </tr>
    <tr>
      <td>E</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <td>F</td>
      <td>0.615385</td>
    </tr>
    <tr>
      <td>G</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <td>M</td>
      <td>0.299854</td>
    </tr>
    <tr>
      <td>T</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Probability of person died in the cabin
prob_df['Died']=1-prob_df.Survived
```


```python
prob_df
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
      <th>Died</th>
    </tr>
    <tr>
      <th>Cabin</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>A</td>
      <td>0.466667</td>
      <td>0.533333</td>
    </tr>
    <tr>
      <td>B</td>
      <td>0.744681</td>
      <td>0.255319</td>
    </tr>
    <tr>
      <td>C</td>
      <td>0.593220</td>
      <td>0.406780</td>
    </tr>
    <tr>
      <td>D</td>
      <td>0.757576</td>
      <td>0.242424</td>
    </tr>
    <tr>
      <td>E</td>
      <td>0.750000</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <td>F</td>
      <td>0.615385</td>
      <td>0.384615</td>
    </tr>
    <tr>
      <td>G</td>
      <td>0.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <td>M</td>
      <td>0.299854</td>
      <td>0.700146</td>
    </tr>
    <tr>
      <td>T</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now probability ratio = prob(Survived) / prob(Died)

prob_df['Probability_ratio']=prob_df['Survived']/prob_df['Died']
```


```python
prob_df
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
      <th>Died</th>
      <th>Probability_ratio</th>
    </tr>
    <tr>
      <th>Cabin</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>A</td>
      <td>0.466667</td>
      <td>0.533333</td>
      <td>0.875000</td>
    </tr>
    <tr>
      <td>B</td>
      <td>0.744681</td>
      <td>0.255319</td>
      <td>2.916667</td>
    </tr>
    <tr>
      <td>C</td>
      <td>0.593220</td>
      <td>0.406780</td>
      <td>1.458333</td>
    </tr>
    <tr>
      <td>D</td>
      <td>0.757576</td>
      <td>0.242424</td>
      <td>3.125000</td>
    </tr>
    <tr>
      <td>E</td>
      <td>0.750000</td>
      <td>0.250000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>F</td>
      <td>0.615385</td>
      <td>0.384615</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <td>G</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>M</td>
      <td>0.299854</td>
      <td>0.700146</td>
      <td>0.428274</td>
    </tr>
    <tr>
      <td>T</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We can use this probability_ratio to encode Cabin feature
df6['Cabin_prob_encoded']=df6.Cabin.map(prob_df['Probability_ratio'].to_dict())
```


```python
df6
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
      <th>Cabin</th>
      <th>Cabin_prob_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>M</td>
      <td>0.428274</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>C</td>
      <td>1.458333</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>M</td>
      <td>0.428274</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>C</td>
      <td>1.458333</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>M</td>
      <td>0.428274</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>886</td>
      <td>0</td>
      <td>M</td>
      <td>0.428274</td>
    </tr>
    <tr>
      <td>887</td>
      <td>1</td>
      <td>B</td>
      <td>2.916667</td>
    </tr>
    <tr>
      <td>888</td>
      <td>0</td>
      <td>M</td>
      <td>0.428274</td>
    </tr>
    <tr>
      <td>889</td>
      <td>1</td>
      <td>C</td>
      <td>1.458333</td>
    </tr>
    <tr>
      <td>890</td>
      <td>0</td>
      <td>M</td>
      <td>0.428274</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 3 columns</p>
</div>




```python

```
