---
title: "Python Pandas - DataFrame"
date: 2021-05-17
tags: [Data Science, Pandas, Python Learning]
header:
  image: "/images/Feature_engineering.jpeg"
excerpt: "Data Science, Pandas, Python Learning"
mathjax: "true"
---

# Exercise - Python Pandas DataFrame


```python
import pandas as pd
import numpy as np
```

1. Write a Pandas program to create and display a DataFrame from a specified dictionary data.


```python
data1= {'X':[78,85,96,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]}
```


```python
df1=pd.DataFrame(data1)
```


```python
print(df1)
```

        X   Y   Z
    0  78  84  86
    1  85  94  97
    2  96  89  96
    3  80  83  72
    4  86  86  83


2. Write a Pandas program to create and display a DataFrame from a specified dictionary data which has the index labels.


```python
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
```


```python
df2=pd.DataFrame(exam_data,index=labels,columns=['attempts','name','qualify','score'])
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



3. Write a Pandas program to display a summary of the basic information about a specified DataFrame and its data.


```python
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 10 entries, a to j
    Data columns (total 4 columns):
    attempts    10 non-null int64
    name        10 non-null object
    qualify     10 non-null object
    score       8 non-null float64
    dtypes: float64(1), int64(1), object(2)
    memory usage: 400.0+ bytes



```python
df2.describe()
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
      <th>attempts</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>10.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1.900000</td>
      <td>13.562500</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.875595</td>
      <td>4.693746</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>2.000000</td>
      <td>13.500000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>2.750000</td>
      <td>17.125000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>3.000000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
</div>



4. Write a Pandas program to get the first 3 rows of a given DataFrame.


```python
df2.head(3)
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
  </tbody>
</table>
</div>



5. 5. Write a Pandas program to select the 'name' and 'score' columns from the above DataFrame.


```python
df2[['name','score']]
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
      <th>name</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>Anastasia</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>Dima</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>Katherine</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>James</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>Emily</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>Michael</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>Matthew</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>Laura</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>Kevin</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>Jonas</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



6. Write a Pandas program to select the specified columns and rows from a given data frame.


```python
df2[['name','score']].iloc[[1,3,5,6],:]
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
      <th>name</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>b</td>
      <td>Dima</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>d</td>
      <td>James</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>f</td>
      <td>Michael</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>Matthew</td>
      <td>14.5</td>
    </tr>
  </tbody>
</table>
</div>



7. Write a Pandas program to select the rows where the number of attempts in the examination is greater than 2.


```python
df2[df2.attempts>2]
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>



8. Write a Pandas program to count the number of rows and columns of a DataFrame.


```python
df2.columns
```




    Index(['attempts', 'name', 'qualify', 'score'], dtype='object')




```python
print("Number of rows : {} \nNumber of columns : {}".format(df2.shape[0],df2.shape[1]))
```

    Number of rows : 10
    Number of columns : 4



```python
print("Number of rows : {} \nNumber of columns : {}".format(len(df2),len(df2.columns)))
```

    Number of rows : 10
    Number of columns : 4


9. Write a Pandas program to select the rows where the score is missing, i.e. is NaN.


```python
df2[df2.score.isna()]
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



10. Write a Pandas program to select the rows the score is between 15 and 20 (inclusive).


```python
df2[(df2.score>=15) & (df2.score<=20)]
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



11. Write a Pandas program to select the rows where number of attempts in the examination is less than 2 and score greater than 15.


```python
df2[(df2.attempts<2) & (df2.score>15)]
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



12. Write a Pandas program to change the score in row 'd' to 11.5.


```python
df12=df2.copy()
```


```python
df12.loc['d','score']=11.5
```


```python
df12.loc['d']
```




    attempts        3
    name        James
    qualify        no
    score        11.5
    Name: d, dtype: object



13. Write a Pandas program to calculate the sum of the examination attempts by the students.


```python
df2.attempts.sum()
```




    19



14. Write a Pandas program to calculate the mean score for each different student in DataFrame.


```python
df2.score.mean()
```




    13.5625



15. Write a Pandas program to append a new row 'k' to data frame with given values for each column. Now delete the new row and return the original DataFrame.


```python
df15=df2.copy()
```


```python
df15.loc['k']=[1,'Suresh','yes',15.5]
```


```python
df15
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
    <tr>
      <td>k</td>
      <td>1</td>
      <td>Suresh</td>
      <td>yes</td>
      <td>15.5</td>
    </tr>
  </tbody>
</table>
</div>



16. Write a Pandas program to sort the DataFrame first by 'name' in descending order, then by 'score' in ascending order.


```python
df2.sort_values(['name','score'],ascending=[False,True])
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
  </tbody>
</table>
</div>



17. Write a Pandas program to replace the 'qualify' column contains the values 'yes' and 'no' with True and False.


```python
df17=df2.copy()
```


```python
df17.qualify=df17.qualify.apply(lambda x:True if x=='yes' else False)
```


```python
df17
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>True</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>False</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>True</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>False</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>True</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>True</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>False</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>True</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



18. Write a Pandas program to change the name 'James' to 'Suresh' in name column of the DataFrame.


```python
df18=df2.copy()
df18.name=df18.name.replace('James','Suresh')
```


```python
df18
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>Suresh</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



19. Write a Pandas program to delete the 'attempts' column from the DataFrame.


```python
df19=df2.copy()
```


```python
df19.drop('attempts',axis=1)
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
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



20. Write a Pandas program to insert a new column in existing DataFrame.


```python
df20=df2.copy()
```


```python
color = ['Red','Blue','Orange','Red','White','White','Blue','Green','Green','Red']
```


```python
df20['color']=color
```


```python
df20
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
      <td>Red</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
      <td>Blue</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
      <td>Orange</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
      <td>Red</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
      <td>White</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
      <td>White</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
      <td>Blue</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
      <td>Green</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
      <td>Green</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
      <td>Red</td>
    </tr>
  </tbody>
</table>
</div>



21. Write a Pandas program to iterate over rows in a DataFrame.


```python
exam_data = [{'name':'Anastasia', 'score':12.5}, {'name':'Dima','score':9}, {'name':'Katherine','score':16.5}]
```


```python
df21=pd.DataFrame(exam_data)
```


```python
for index,row in df21.iterrows():
    print(row['name'],row['score'])
```

    Anastasia 12.5
    Dima 9.0
    Katherine 16.5


22. Write a Pandas program to get list from DataFrame column headers.


```python
df2.columns.tolist()
```




    ['attempts', 'name', 'qualify', 'score']



23. Write a Pandas program to rename columns of a given DataFrame.


```python
df23= pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]})
```


```python
df23
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
df23.columns=['Column1','Column2','Column3']
```


```python
df23
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
      <th>Column1</th>
      <th>Column2</th>
      <th>Column3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
df23=df23.rename(columns={'Column1':'Columns1'})
```


```python
df23
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
      <th>Columns1</th>
      <th>Column2</th>
      <th>Column3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



24. Write a Pandas program to select rows from a given DataFrame based on values in some columns.


```python
df24= pd.DataFrame({'col1': [1, 4,3,4,5], 'col2': [4, 5, 6,7,8], 'col3': [7, 8, 9,0,1]})
```


```python
df24[df24.col1==4]
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



25. Write a Pandas program to change the order of a DataFrame columns.


```python
df25=df24[['col3','col2','col1']]
```


```python
df25
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
      <th>col3</th>
      <th>col2</th>
      <th>col1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <td>2</td>
      <td>9</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>7</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



26. Write a Pandas program to add one row in an existing DataFrame.


```python
df26=df24.copy()
```


```python
df26.loc[len(df26)]=[10,11,12]
```


```python
df26
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



27. Write a Pandas program to write a DataFrame to CSV file using tab separator.


```python
df24.to_csv('27.csv',sep='\t',index=False)
```


```python
df27=pd.read_csv('27.csv')
```


```python
df27
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
      <th>col1\tcol2\tcol3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1\t4\t7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4\t5\t8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3\t6\t9</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4\t7\t0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5\t8\t1</td>
    </tr>
  </tbody>
</table>
</div>



28. Write a Pandas program to count city wise number of people from a given of data set (city, name of the person).


```python
df28 = pd.DataFrame({'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'city': ['California', 'Los Angeles', 'California', 'California', 'California', 'Los Angeles', 'Los Angeles', 'Georgia', 'Georgia', 'Los Angeles']})

```


```python
df28
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
      <th>name</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Anastasia</td>
      <td>California</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Dima</td>
      <td>Los Angeles</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Katherine</td>
      <td>California</td>
    </tr>
    <tr>
      <td>3</td>
      <td>James</td>
      <td>California</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Emily</td>
      <td>California</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Michael</td>
      <td>Los Angeles</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Matthew</td>
      <td>Los Angeles</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Laura</td>
      <td>Georgia</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Kevin</td>
      <td>Georgia</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Jonas</td>
      <td>Los Angeles</td>
    </tr>
  </tbody>
</table>
</div>




```python
df28.groupby('city').size().reset_index(name='Number of people')
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
      <th>city</th>
      <th>Number of people</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>California</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Georgia</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Los Angeles</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



29.  Write a Pandas program to delete DataFrame row(s) based on given column value.


```python
df29=df24.copy()
```


```python
df29=df29[df29.col2!=5]
```


```python
df29
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



30. Write a Pandas program to widen output display to see more columns.


```python
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
```


```python
print(df24)
```

       col1  col2  col3
    0     1     4     7
    1     4     5     8
    2     3     6     9
    3     4     7     0
    4     5     8     1


31. Write a Pandas program to select a row of series/dataframe by given integer index.


```python
df24.loc[[2]]
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



32. Write a Pandas program to replace all the NaN values with Zero's in a column of a dataframe.


```python
df32=df2.copy()
```


```python
df32.fillna(0)
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



33. Write a Pandas program to convert index in a column of the given dataframe.


```python
df33=df2.copy()
```


```python
df33.reset_index()
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
      <th>index</th>
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>7</td>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>8</td>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df33.reset_index().to_string(index=False))
```

    index  attempts       name qualify  score
        a         1  Anastasia     yes   12.5
        b         3       Dima      no    9.0
        c         2  Katherine     yes   16.5
        d         3      James      no    NaN
        e         2      Emily      no    9.0
        f         3    Michael     yes   20.0
        g         1    Matthew     yes   14.5
        h         1      Laura      no    NaN
        i         2      Kevin      no    8.0
        j         1      Jonas     yes   19.0



```python
#Back to original be setting index once again
df33.reset_index().set_index('index')
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



34. Write a Pandas program to set a given value for particular cell in  DataFrame using index value.


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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df34=df2.copy()
```


```python
df34.loc['i',:].score=10.2
```

    C:\Users\a112471\AppData\Local\Continuum\anaconda3\lib\site-packages\pandas\core\generic.py:5208: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self[name] = value



```python
df34
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df34.set_value('i','score',10.2)
```

    C:\Users\a112471\AppData\Local\Continuum\anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
      """Entry point for launching an IPython kernel.





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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>10.2</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df34.at['i','score']=10.2
```


```python
df34
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>10.2</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df34.iat[8,3]
```




    10.2



35. Write a Pandas program to count the NaN values in one or more columns in DataFrame.


```python
df2.isna().sum().sum()
```




    2



36. Write a Pandas program to drop a list of rows from a specified DataFrame.


```python
df36=df24.copy()
```


```python
df36.drop([2,4],axis=0)
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



37. Write a Pandas program to reset index in a given DataFrame.


```python
df37=df2.copy()
```


```python
df37.drop(['a','b']).reset_index()
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
      <th>index</th>
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>5</td>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>6</td>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df37
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



38. Write a Pandas program to divide a DataFrame in a given ratio.


```python
df38=pd.DataFrame(np.random.randn(10,2))
```


```python
df38
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.248443</td>
      <td>-1.241437</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.350421</td>
      <td>-1.539320</td>
    </tr>
    <tr>
      <td>2</td>
      <td>-0.118075</td>
      <td>1.586380</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.126995</td>
      <td>1.419173</td>
    </tr>
    <tr>
      <td>4</td>
      <td>-0.062786</td>
      <td>0.070672</td>
    </tr>
    <tr>
      <td>5</td>
      <td>-0.077202</td>
      <td>0.442252</td>
    </tr>
    <tr>
      <td>6</td>
      <td>-0.547482</td>
      <td>0.472672</td>
    </tr>
    <tr>
      <td>7</td>
      <td>-0.154437</td>
      <td>-0.138206</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.634787</td>
      <td>1.480123</td>
    </tr>
    <tr>
      <td>9</td>
      <td>-0.029771</td>
      <td>0.834099</td>
    </tr>
  </tbody>
</table>
</div>




```python
df38_70per=df38.sample(frac=.7,random_state=0)
```


```python
df38_30per=df38.drop(df38_70per.index)
```


```python
df38_70per
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>-0.118075</td>
      <td>1.586380</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.634787</td>
      <td>1.480123</td>
    </tr>
    <tr>
      <td>4</td>
      <td>-0.062786</td>
      <td>0.070672</td>
    </tr>
    <tr>
      <td>9</td>
      <td>-0.029771</td>
      <td>0.834099</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.350421</td>
      <td>-1.539320</td>
    </tr>
    <tr>
      <td>6</td>
      <td>-0.547482</td>
      <td>0.472672</td>
    </tr>
    <tr>
      <td>7</td>
      <td>-0.154437</td>
      <td>-0.138206</td>
    </tr>
  </tbody>
</table>
</div>




```python
df38_30per
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.248443</td>
      <td>-1.241437</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.126995</td>
      <td>1.419173</td>
    </tr>
    <tr>
      <td>5</td>
      <td>-0.077202</td>
      <td>0.442252</td>
    </tr>
  </tbody>
</table>
</div>



39. Write a Pandas program to combining two series into a DataFrame.


```python
s39_1 = pd.Series(['100', '200', 'python', '300.12', '400'])
s39_2 = pd.Series(['10', '20', 'php', '30.12', '40'])
```


```python
pd.concat([s39_1,s39_2],axis=1)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>100</td>
      <td>10</td>
    </tr>
    <tr>
      <td>1</td>
      <td>200</td>
      <td>20</td>
    </tr>
    <tr>
      <td>2</td>
      <td>python</td>
      <td>php</td>
    </tr>
    <tr>
      <td>3</td>
      <td>300.12</td>
      <td>30.12</td>
    </tr>
    <tr>
      <td>4</td>
      <td>400</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



40. Write a Pandas program to shuffle a given DataFrame rows.


```python
df40=df2.copy()
```


```python
df40.sample(frac=1)
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df40
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



41. Write a Pandas program to convert DataFrame column type from string to datetime.


```python
s41 = pd.Series(['3/11/2000', '3/12/2000', '3/13/2000'])
```


```python
s41
```




    0    3/11/2000
    1    3/12/2000
    2    3/13/2000
    dtype: object




```python
pd.to_datetime(s41)
```




    0   2000-03-11
    1   2000-03-12
    2   2000-03-13
    dtype: datetime64[ns]



42. Write a Pandas program to rename a specific column name in a given DataFrame.


```python
df42= pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]})
```


```python
df42.rename(columns={'col2':'column2'})
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
      <th>col1</th>
      <th>column2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



43. Write a Pandas program to get a list of a specified column of a DataFrame.


```python
df42.col2.tolist()
```




    [4, 5, 6]



44. Write a Pandas program to create a DataFrame from a Numpy array and specify the index column and column headers.


```python
pd.DataFrame(np.random.randn(30).reshape(10,3),index=np.linspace(0,9,10,dtype='int'),columns=['column1','columns2','columns3'])
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
      <th>column1</th>
      <th>columns2</th>
      <th>columns3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>-0.345403</td>
      <td>2.167887</td>
      <td>0.495698</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.146200</td>
      <td>-1.078212</td>
      <td>0.231268</td>
    </tr>
    <tr>
      <td>2</td>
      <td>-0.619157</td>
      <td>1.287958</td>
      <td>0.216466</td>
    </tr>
    <tr>
      <td>3</td>
      <td>-0.859700</td>
      <td>-0.735332</td>
      <td>0.437529</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.395431</td>
      <td>1.002545</td>
      <td>-0.999019</td>
    </tr>
    <tr>
      <td>5</td>
      <td>-0.474449</td>
      <td>-0.106919</td>
      <td>-2.089148</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.375528</td>
      <td>0.173526</td>
      <td>-1.103849</td>
    </tr>
    <tr>
      <td>7</td>
      <td>-1.468913</td>
      <td>1.206882</td>
      <td>-0.983365</td>
    </tr>
    <tr>
      <td>8</td>
      <td>-0.751145</td>
      <td>-0.289252</td>
      <td>-0.346848</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.923659</td>
      <td>1.038505</td>
      <td>-1.394882</td>
    </tr>
  </tbody>
</table>
</div>



45. Write a Pandas program to find the row for where the value of a given column is maximum.


```python
df45 = pd.DataFrame({'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]})

```


```python
df45
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Solution 1 (not efficient)
df45[df45.col1==df45.col1.max()].index[0]
```




    4




```python
df45['col1'].idxmax()
```




    4




```python
df45['col2'].idxmax()
```




    3




```python
df45['col3'].idxmax()
```




    2



46. Write a Pandas program to check whether a given column is present in a DataFrame or not.


```python
col='Col4'
if col in df45.columns.tolist():
    print(col,'present in DataFrame')
else:
    print(col,'not present in DataFrame')
```

    Col4 not present in DataFrame


47. Write a Pandas program to get the specified row value of a given DataFrame.


```python
df45.iloc[0]
```




    col1    1
    col2    4
    col3    7
    Name: 0, dtype: int64



48. Write a Pandas program to get the datatypes of columns of a DataFrame.


```python
df2.dtypes
```




    attempts      int64
    name         object
    qualify      object
    score       float64
    dtype: object



49. Write a Pandas program to append data to an empty DataFrame.


```python
df49=pd.DataFrame()
```


```python
d49=pd.DataFrame({'col1':range(3),'col2':range(3)})
```


```python
df49.append(d49)
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



50. Write a Pandas program to sort a given DataFrame by two or more columns.


```python
df50=df2.copy()
```


```python
df50.sort_values(['attempts','name'],ascending=[True,False])
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
      <th>attempts</th>
      <th>name</th>
      <th>qualify</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>g</td>
      <td>1</td>
      <td>Matthew</td>
      <td>yes</td>
      <td>14.5</td>
    </tr>
    <tr>
      <td>h</td>
      <td>1</td>
      <td>Laura</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>j</td>
      <td>1</td>
      <td>Jonas</td>
      <td>yes</td>
      <td>19.0</td>
    </tr>
    <tr>
      <td>a</td>
      <td>1</td>
      <td>Anastasia</td>
      <td>yes</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>i</td>
      <td>2</td>
      <td>Kevin</td>
      <td>no</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>c</td>
      <td>2</td>
      <td>Katherine</td>
      <td>yes</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>e</td>
      <td>2</td>
      <td>Emily</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>f</td>
      <td>3</td>
      <td>Michael</td>
      <td>yes</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>d</td>
      <td>3</td>
      <td>James</td>
      <td>no</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>b</td>
      <td>3</td>
      <td>Dima</td>
      <td>no</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>



51. Write a Pandas program to convert the datatype of a given column (floats to ints).


```python
df51=df2.copy()
```


```python
df51.dtypes
```




    attempts      int64
    name         object
    qualify      object
    score       float64
    dtype: object




```python
df51.fillna(0).astype({'score':int}).dtypes
```




    attempts     int64
    name        object
    qualify     object
    score        int32
    dtype: object



52. Write a Pandas program to remove infinite values from a given DataFrame.


```python
df52 = pd.DataFrame([1000, 2000, 3000, -4000, np.inf, -np.inf])
```


```python
df52
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2000.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3000.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>-4000.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>5</td>
      <td>-inf</td>
    </tr>
  </tbody>
</table>
</div>




```python
df52.replace([np.inf,-np.inf],np.NaN)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2000.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3000.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>-4000.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



53. Write a Pandas program to insert a given column at a specific column index in a DataFrame.


```python
df53 = pd.DataFrame({'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]})

```


```python
df53
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
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>6</td>
      <td>12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
col1= [1, 2, 3, 4, 7]
```


```python
idx=0
```


```python
df53.insert(loc=idx,column='col1',value=col1)
```


```python
df53
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



54. Write a Pandas program to convert a given list of lists into a Dataframe.


```python
d54=[[2, 4], [1, 3]]
```


```python
pd.DataFrame(d54,columns=['col1','col2'])
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



55. Write a Pandas program to group by the first column and get second column as lists in rows.


```python
df55=pd.DataFrame({'col1':['C1','C1','C2','C2','C2','C3','C2'],'col2':[1,2,3,3,4,6,5]})
```


```python
g55=df55.groupby('col1')['col2']
```


```python
#Wrong Solution (or not gettring desired result)
for ele in g55:
    print(ele[1])
```

    0    1
    1    2
    Name: col2, dtype: int64
    2    3
    3    3
    4    4
    6    5
    Name: col2, dtype: int64
    5    6
    Name: col2, dtype: int64



```python
#Correct Solution
df55.groupby('col1')['col2'].apply(list)
```




    col1
    C1          [1, 2]
    C2    [3, 3, 4, 5]
    C3             [6]
    Name: col2, dtype: object



56. Write a Pandas program to get column index from column name of a given DataFrame.


```python
df45.columns.tolist().index('col2')
```




    1




```python
df45.columns.get_loc('col2')
```




    1



57. Write a Pandas program to count number of columns of a DataFrame.


```python
df45.shape[1]
```




    3




```python
len(df45.columns)
```




    3



58. Write a Pandas program to select all columns, except one given column in a DataFrame.


```python
#Solution 1
df45.drop('col3',axis=1)
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Solution 2
df45.loc[:,df45.columns!='col3']
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



59. Write a Pandas program to get first n records of a DataFrame.


```python
df45.head(3)
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



60. Write a Pandas program to get last n records of a DataFrame.


```python
df45.iloc[-3:,:]
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
df45.tail(3)
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



61. Write a Pandas program to get topmost n records within each group of a DataFrame.


```python
df61 = pd.DataFrame({'col1': [1, 2, 3, 4, 7, 11], 'col2': [4, 5, 6, 9, 5, 0], 'col3': [7, 5, 8, 12, 1,11]})

```


```python
df61
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
      <td>12</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>11</td>
      <td>0</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
df61.nlargest(3,'col1')
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5</td>
      <td>11</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
df61.nlargest(3,'col2')
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
      <td>12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df61.nlargest(3,'col3')
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
      <td>12</td>
    </tr>
    <tr>
      <td>5</td>
      <td>11</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



62. Write a Pandas program to remove first n rows of a given DataFrame.


```python
df62=df61.copy()
```


```python
df62
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
      <td>12</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>11</td>
      <td>0</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
df62=df62.tail(-3)
```


```python
df62
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>9</td>
      <td>12</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>11</td>
      <td>0</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



63. Write a Pandas program to remove last n rows of a given DataFrame.


```python
df63=df61.copy()
```


```python
df63=df63.iloc[:-3,:]
```


```python
df63
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



64. Write a Pandas program to add a prefix or suffix to all columns of a given DataFrame.


```python
df64 = pd.DataFrame({'W':[68,75,86,80,66],'X':[78,85,96,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]});

```


```python
df64_1=df64.copy()
```


```python
df64_1.columns='A_'+df64.columns
```


```python
df64_1
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
      <th>A_W</th>
      <th>A_X</th>
      <th>A_Y</th>
      <th>A_Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>68</td>
      <td>78</td>
      <td>84</td>
      <td>86</td>
    </tr>
    <tr>
      <td>1</td>
      <td>75</td>
      <td>85</td>
      <td>94</td>
      <td>97</td>
    </tr>
    <tr>
      <td>2</td>
      <td>86</td>
      <td>96</td>
      <td>89</td>
      <td>96</td>
    </tr>
    <tr>
      <td>3</td>
      <td>80</td>
      <td>80</td>
      <td>83</td>
      <td>72</td>
    </tr>
    <tr>
      <td>4</td>
      <td>66</td>
      <td>86</td>
      <td>86</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
df64_2=df64.copy()
```


```python
df64_2.columns=df64.columns+'_1'
```


```python
df64_2
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
      <th>W_1</th>
      <th>X_1</th>
      <th>Y_1</th>
      <th>Z_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>68</td>
      <td>78</td>
      <td>84</td>
      <td>86</td>
    </tr>
    <tr>
      <td>1</td>
      <td>75</td>
      <td>85</td>
      <td>94</td>
      <td>97</td>
    </tr>
    <tr>
      <td>2</td>
      <td>86</td>
      <td>96</td>
      <td>89</td>
      <td>96</td>
    </tr>
    <tr>
      <td>3</td>
      <td>80</td>
      <td>80</td>
      <td>83</td>
      <td>72</td>
    </tr>
    <tr>
      <td>4</td>
      <td>66</td>
      <td>86</td>
      <td>86</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
#sol2
df64.add_prefix('A_')
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
      <th>A_W</th>
      <th>A_X</th>
      <th>A_Y</th>
      <th>A_Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>68</td>
      <td>78</td>
      <td>84</td>
      <td>86</td>
    </tr>
    <tr>
      <td>1</td>
      <td>75</td>
      <td>85</td>
      <td>94</td>
      <td>97</td>
    </tr>
    <tr>
      <td>2</td>
      <td>86</td>
      <td>96</td>
      <td>89</td>
      <td>96</td>
    </tr>
    <tr>
      <td>3</td>
      <td>80</td>
      <td>80</td>
      <td>83</td>
      <td>72</td>
    </tr>
    <tr>
      <td>4</td>
      <td>66</td>
      <td>86</td>
      <td>86</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
df64.add_suffix('_1')
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
      <th>W_1</th>
      <th>X_1</th>
      <th>Y_1</th>
      <th>Z_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>68</td>
      <td>78</td>
      <td>84</td>
      <td>86</td>
    </tr>
    <tr>
      <td>1</td>
      <td>75</td>
      <td>85</td>
      <td>94</td>
      <td>97</td>
    </tr>
    <tr>
      <td>2</td>
      <td>86</td>
      <td>96</td>
      <td>89</td>
      <td>96</td>
    </tr>
    <tr>
      <td>3</td>
      <td>80</td>
      <td>80</td>
      <td>83</td>
      <td>72</td>
    </tr>
    <tr>
      <td>4</td>
      <td>66</td>
      <td>86</td>
      <td>86</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>



65. Write a Pandas program to reverse order (rows, columns) of a given DataFrame.


```python
df64[df64.columns[::-1]]
#df64.iloc[:,::-1] same answer
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
      <th>Z</th>
      <th>Y</th>
      <th>X</th>
      <th>W</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>86</td>
      <td>84</td>
      <td>78</td>
      <td>68</td>
    </tr>
    <tr>
      <td>1</td>
      <td>97</td>
      <td>94</td>
      <td>85</td>
      <td>75</td>
    </tr>
    <tr>
      <td>2</td>
      <td>96</td>
      <td>89</td>
      <td>96</td>
      <td>86</td>
    </tr>
    <tr>
      <td>3</td>
      <td>72</td>
      <td>83</td>
      <td>80</td>
      <td>80</td>
    </tr>
    <tr>
      <td>4</td>
      <td>83</td>
      <td>86</td>
      <td>86</td>
      <td>66</td>
    </tr>
  </tbody>
</table>
</div>




```python
df64.iloc[::-1]
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4</td>
      <td>66</td>
      <td>86</td>
      <td>86</td>
      <td>83</td>
    </tr>
    <tr>
      <td>3</td>
      <td>80</td>
      <td>80</td>
      <td>83</td>
      <td>72</td>
    </tr>
    <tr>
      <td>2</td>
      <td>86</td>
      <td>96</td>
      <td>89</td>
      <td>96</td>
    </tr>
    <tr>
      <td>1</td>
      <td>75</td>
      <td>85</td>
      <td>94</td>
      <td>97</td>
    </tr>
    <tr>
      <td>0</td>
      <td>68</td>
      <td>78</td>
      <td>84</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Reverse row order and reset index:")
df64.iloc[::-1].reset_index(drop=True)
```

    Reverse row order and reset index:





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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>66</td>
      <td>86</td>
      <td>86</td>
      <td>83</td>
    </tr>
    <tr>
      <td>1</td>
      <td>80</td>
      <td>80</td>
      <td>83</td>
      <td>72</td>
    </tr>
    <tr>
      <td>2</td>
      <td>86</td>
      <td>96</td>
      <td>89</td>
      <td>96</td>
    </tr>
    <tr>
      <td>3</td>
      <td>75</td>
      <td>85</td>
      <td>94</td>
      <td>97</td>
    </tr>
    <tr>
      <td>4</td>
      <td>68</td>
      <td>78</td>
      <td>84</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
</div>



66. Write a Pandas program to select columns by data type of a given DataFrame.


```python
df66 = pd.DataFrame({
    'name': ['Alberto Franco','Gino Mcneill','Ryan Parkes', 'Eesha Hinton', 'Syed Wharton'],
    'date_of_birth': ['17/05/2002','16/02/1999','25/09/1998','11/05/2002','15/09/1997'],
    'age': [18.5, 21.2, 22.5, 22, 23]
})
```


```python
df66
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
      <th>name</th>
      <th>date_of_birth</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Alberto Franco</td>
      <td>17/05/2002</td>
      <td>18.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Mcneill</td>
      <td>16/02/1999</td>
      <td>21.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Ryan Parkes</td>
      <td>25/09/1998</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Syed Wharton</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df66.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
    name             5 non-null object
    date_of_birth    5 non-null object
    age              5 non-null float64
    dtypes: float64(1), object(2)
    memory usage: 248.0+ bytes



```python
df66.select_dtypes(include='number')
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
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>18.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>21.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df66.select_dtypes(include='object')
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
      <th>name</th>
      <th>date_of_birth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Alberto Franco</td>
      <td>17/05/2002</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Mcneill</td>
      <td>16/02/1999</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Ryan Parkes</td>
      <td>25/09/1998</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Syed Wharton</td>
      <td>15/09/1997</td>
    </tr>
  </tbody>
</table>
</div>



67. Write a Pandas program to split a given DataFrame into two random subsets.


```python
df67=df66.copy()
```


```python
df67_1=df67.sample(frac=0.7,random_state=0)
```


```python
df67_2=df67.drop(df67_1.index)
```


```python
df67_1
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
      <th>name</th>
      <th>date_of_birth</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>Ryan Parkes</td>
      <td>25/09/1998</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Alberto Franco</td>
      <td>17/05/2002</td>
      <td>18.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Mcneill</td>
      <td>16/02/1999</td>
      <td>21.2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df67_2
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
      <th>name</th>
      <th>date_of_birth</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4</td>
      <td>Syed Wharton</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>



68. Write a Pandas program to rename all columns with the same pattern of a given DataFrame.


```python
df68 = pd.DataFrame({
    'Name': ['Alberto Franco','Gino Mcneill','Ryan Parkes', 'Eesha Hinton', 'Syed Wharton'],
    'Date_of_Birth': ['17/05/2002','16/02/1999','25/09/1998','11/05/2002','15/09/1997'],
    'Age': [18.5, 21.2, 22.5, 22, 23]
})
```


```python
df68
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
      <th>Name</th>
      <th>Date_of_Birth</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Alberto Franco</td>
      <td>17/05/2002</td>
      <td>18.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Mcneill</td>
      <td>16/02/1999</td>
      <td>21.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Ryan Parkes</td>
      <td>25/09/1998</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Syed Wharton</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df68.columns=df68.columns.str.lower().str.strip()
```


```python
df68
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
      <th>name</th>
      <th>date_of_birth</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Alberto Franco</td>
      <td>17/05/2002</td>
      <td>18.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Mcneill</td>
      <td>16/02/1999</td>
      <td>21.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Ryan Parkes</td>
      <td>25/09/1998</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Syed Wharton</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>



69. Write a Pandas program to merge datasets and check uniqueness.


```python
df69 = pd.DataFrame({
    'Name': ['Alberto Franco','Gino Mcneill','Ryan Parkes', 'Eesha Hinton', 'Syed Wharton'],
    'Date_Of_Birth ': ['17/05/2002','16/02/1999','25/09/1998','11/05/2002','15/09/1997'],
    'Age': [18.5, 21.2, 22.5, 22, 23]
})
```


```python
df69_1=df69.copy()
```


```python
df69_1=df69_1.drop([0,1])
```


```python
df69_1
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
      <th>Name</th>
      <th>Date_Of_Birth</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>Ryan Parkes</td>
      <td>25/09/1998</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Syed Wharton</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df69_2=df69.copy()
```


```python
df69_2=df69_2.drop([2])
```


```python
df69_2
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
      <th>Name</th>
      <th>Date_Of_Birth</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Alberto Franco</td>
      <td>17/05/2002</td>
      <td>18.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Mcneill</td>
      <td>16/02/1999</td>
      <td>21.2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Syed Wharton</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#one_to_one
pd.merge(df69_1,df69_2,validate='one_to_one')
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
      <th>Name</th>
      <th>Date_Of_Birth</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Syed Wharton</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#one_to_many
pd.merge(df69_1,df69_2,validate='one_to_many')
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
      <th>Name</th>
      <th>Date_Of_Birth</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Syed Wharton</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#many_to_one
pd.merge(df69_1,df69_2,validate='many_to_one')
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
      <th>Name</th>
      <th>Date_Of_Birth</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Syed Wharton</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#many_to_many
pd.merge(df69_1,df69_2,validate='many_to_many')
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
      <th>Name</th>
      <th>Date_Of_Birth</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Syed Wharton</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>



70. Write a Pandas program to convert continuous values of a column in a given DataFrame to categorical.


```python
df70 = pd.DataFrame({
    'name': ['Alberto Franco','Gino Mcneill','Ryan Parkes', 'Eesha Hinton', 'Syed Wharton', 'Kierra Gentry'],
      'age': [18, 22, 85, 50, 80, 5]
})
```


```python
df70
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
      <th>name</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Alberto Franco</td>
      <td>18</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Mcneill</td>
      <td>22</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Ryan Parkes</td>
      <td>85</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>50</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Syed Wharton</td>
      <td>80</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Kierra Gentry</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df70['age_group']=pd.cut(df70.age,bins=[0,18,65,95],labels=['kids','adult','elderly'])
```


```python
df70
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
      <th>name</th>
      <th>age</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Alberto Franco</td>
      <td>18</td>
      <td>kids</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Mcneill</td>
      <td>22</td>
      <td>adult</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Ryan Parkes</td>
      <td>85</td>
      <td>elderly</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>50</td>
      <td>adult</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Syed Wharton</td>
      <td>80</td>
      <td>elderly</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Kierra Gentry</td>
      <td>5</td>
      <td>kids</td>
    </tr>
  </tbody>
</table>
</div>



71. Write a Pandas program to display memory usage of a given DataFrame and every column of the DataFrame.


```python
df70.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6 entries, 0 to 5
    Data columns (total 3 columns):
    name         6 non-null object
    age          6 non-null int64
    age_group    6 non-null category
    dtypes: category(1), int64(1), object(1)
    memory usage: 334.0+ bytes



```python
df70.info(memory_usage='deep')
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6 entries, 0 to 5
    Data columns (total 3 columns):
    name         6 non-null object
    age          6 non-null int64
    age_group    6 non-null category
    dtypes: category(1), int64(1), object(1)
    memory usage: 903.0 bytes



```python
df70.info(memory_usage='deep')
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6 entries, 0 to 5
    Data columns (total 3 columns):
    name         6 non-null object
    age          6 non-null int64
    age_group    6 non-null category
    dtypes: category(1), int64(1), object(1)
    memory usage: 903.0 bytes



```python
df70.memory_usage(deep=True)
```




    Index        128
    name         416
    age           48
    age_group    311
    dtype: int64




```python
df70.memory_usage(deep=False)
```




    Index        128
    name          48
    age           48
    age_group    110
    dtype: int64



72. Write a Pandas program to combine many given series to create a DataFrame.


```python
s72_1=pd.Series('php python jawa c# c++'.split())
```


```python
s72_2=pd.Series([1,2,3,4,5])
```


```python
pd.concat([s72_1,s72_2],axis=1)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>php</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>python</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>jawa</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c#</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c++</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame({'col1':s72_1,'col2':s72_2})
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>php</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>python</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>jawa</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c#</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c++</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



73. Write a Pandas program to create DataFrames that contains random values, contains missing values, contains datetime values and contains mixed values.


```python
pd.DataFrame(np.random.randn(40).reshape(10,4),columns='A B C D'.split())
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.976226</td>
      <td>-1.231195</td>
      <td>-1.927390</td>
      <td>1.687527</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.265455</td>
      <td>-0.872881</td>
      <td>-1.234580</td>
      <td>0.025491</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.309553</td>
      <td>-0.368772</td>
      <td>1.281644</td>
      <td>0.903505</td>
    </tr>
    <tr>
      <td>3</td>
      <td>-0.490682</td>
      <td>-0.934150</td>
      <td>0.395868</td>
      <td>-0.614944</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.270408</td>
      <td>0.105810</td>
      <td>-0.934098</td>
      <td>-0.193174</td>
    </tr>
    <tr>
      <td>5</td>
      <td>-0.412797</td>
      <td>0.123316</td>
      <td>0.656035</td>
      <td>-0.801055</td>
    </tr>
    <tr>
      <td>6</td>
      <td>-0.627806</td>
      <td>-0.182581</td>
      <td>0.200891</td>
      <td>-0.471121</td>
    </tr>
    <tr>
      <td>7</td>
      <td>-0.705074</td>
      <td>0.416434</td>
      <td>0.424676</td>
      <td>0.495411</td>
    </tr>
    <tr>
      <td>8</td>
      <td>-0.157094</td>
      <td>-1.339826</td>
      <td>1.379757</td>
      <td>1.843449</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.265529</td>
      <td>-0.211037</td>
      <td>0.542715</td>
      <td>1.862077</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.util.testing.makeDataFrame()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mIrUvYIpau</td>
      <td>-0.132329</td>
      <td>0.369235</td>
      <td>-2.518745</td>
      <td>0.463750</td>
    </tr>
    <tr>
      <td>IIuIlUG3TZ</td>
      <td>0.224135</td>
      <td>1.173692</td>
      <td>-1.402806</td>
      <td>-0.634084</td>
    </tr>
    <tr>
      <td>lhz7LziqT6</td>
      <td>-1.747848</td>
      <td>1.112314</td>
      <td>-1.173867</td>
      <td>-1.275242</td>
    </tr>
    <tr>
      <td>bDRGRAVZ5L</td>
      <td>-1.125598</td>
      <td>1.167309</td>
      <td>0.119561</td>
      <td>-0.315563</td>
    </tr>
    <tr>
      <td>tltm1jawR2</td>
      <td>-0.170738</td>
      <td>-0.020117</td>
      <td>0.807885</td>
      <td>-1.399174</td>
    </tr>
    <tr>
      <td>gyHU2hW6dO</td>
      <td>0.228735</td>
      <td>-0.685193</td>
      <td>-0.241533</td>
      <td>-1.230713</td>
    </tr>
    <tr>
      <td>GH6l1o4kp8</td>
      <td>-0.683374</td>
      <td>-1.373173</td>
      <td>1.563031</td>
      <td>-1.344920</td>
    </tr>
    <tr>
      <td>Ee15AMPrrZ</td>
      <td>-0.229806</td>
      <td>-1.190730</td>
      <td>0.691445</td>
      <td>-0.614836</td>
    </tr>
    <tr>
      <td>CVtTzdhYOH</td>
      <td>-1.806060</td>
      <td>0.126193</td>
      <td>1.921333</td>
      <td>0.655736</td>
    </tr>
    <tr>
      <td>u7yGpt7ojr</td>
      <td>0.418048</td>
      <td>-0.560950</td>
      <td>-2.247065</td>
      <td>2.067856</td>
    </tr>
    <tr>
      <td>JQRGGsJRHG</td>
      <td>-1.116554</td>
      <td>0.614687</td>
      <td>-0.367641</td>
      <td>0.160849</td>
    </tr>
    <tr>
      <td>6NTUAVqlrk</td>
      <td>0.622822</td>
      <td>0.119354</td>
      <td>-1.052081</td>
      <td>-0.347517</td>
    </tr>
    <tr>
      <td>8XKVbhaoCO</td>
      <td>1.380594</td>
      <td>-0.907565</td>
      <td>-1.298444</td>
      <td>0.627115</td>
    </tr>
    <tr>
      <td>HSlwYeeFDf</td>
      <td>0.566009</td>
      <td>-0.185294</td>
      <td>0.437134</td>
      <td>-1.943550</td>
    </tr>
    <tr>
      <td>441fk4pWhD</td>
      <td>0.214447</td>
      <td>-1.183068</td>
      <td>1.761380</td>
      <td>-0.652720</td>
    </tr>
    <tr>
      <td>VdiWlWttWB</td>
      <td>0.743793</td>
      <td>1.722755</td>
      <td>0.757516</td>
      <td>0.615187</td>
    </tr>
    <tr>
      <td>ls77KaEvOq</td>
      <td>-1.344870</td>
      <td>0.364236</td>
      <td>0.093453</td>
      <td>-0.609153</td>
    </tr>
    <tr>
      <td>5K0bPzfMFi</td>
      <td>-1.065251</td>
      <td>0.841662</td>
      <td>1.479194</td>
      <td>-0.731203</td>
    </tr>
    <tr>
      <td>AXsuHNuwHo</td>
      <td>-0.827250</td>
      <td>1.965530</td>
      <td>0.773863</td>
      <td>0.665516</td>
    </tr>
    <tr>
      <td>T0GlToPXZi</td>
      <td>-0.320390</td>
      <td>-0.728882</td>
      <td>2.018289</td>
      <td>-0.578019</td>
    </tr>
    <tr>
      <td>22vFAApOA5</td>
      <td>-0.494500</td>
      <td>0.550730</td>
      <td>-1.067188</td>
      <td>0.044000</td>
    </tr>
    <tr>
      <td>gPiu7GA3im</td>
      <td>-0.521648</td>
      <td>-0.856955</td>
      <td>0.709654</td>
      <td>-0.932031</td>
    </tr>
    <tr>
      <td>OZnBwP9gm9</td>
      <td>-0.181220</td>
      <td>-1.985956</td>
      <td>0.794047</td>
      <td>-0.335244</td>
    </tr>
    <tr>
      <td>9U23Nv30l1</td>
      <td>0.126521</td>
      <td>-0.269881</td>
      <td>-0.475025</td>
      <td>0.873421</td>
    </tr>
    <tr>
      <td>NLLYWOAAPo</td>
      <td>-1.399080</td>
      <td>-0.614755</td>
      <td>-0.431522</td>
      <td>-1.314192</td>
    </tr>
    <tr>
      <td>0snfJwr8Zi</td>
      <td>1.104287</td>
      <td>-0.747049</td>
      <td>-0.494870</td>
      <td>-0.280476</td>
    </tr>
    <tr>
      <td>hyIRHRTG6c</td>
      <td>0.447759</td>
      <td>0.375540</td>
      <td>-1.348494</td>
      <td>-1.576017</td>
    </tr>
    <tr>
      <td>nO4Ybp7ia5</td>
      <td>-0.665945</td>
      <td>1.525348</td>
      <td>0.291121</td>
      <td>0.763574</td>
    </tr>
    <tr>
      <td>OU81k9dvGD</td>
      <td>0.897210</td>
      <td>0.587101</td>
      <td>1.375282</td>
      <td>-0.842839</td>
    </tr>
    <tr>
      <td>ktS4vr3Pqy</td>
      <td>-0.270603</td>
      <td>-0.081540</td>
      <td>-1.243459</td>
      <td>0.965349</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.util.testing.makeMissingDataframe()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>K1jtgA2gtF</td>
      <td>-1.142478</td>
      <td>1.478004</td>
      <td>0.777470</td>
      <td>0.517304</td>
    </tr>
    <tr>
      <td>Fz0OYcxMQn</td>
      <td>-0.096315</td>
      <td>0.070175</td>
      <td>0.176539</td>
      <td>-0.511371</td>
    </tr>
    <tr>
      <td>7gLTJCMX3w</td>
      <td>-0.236882</td>
      <td>0.453078</td>
      <td>-0.588427</td>
      <td>1.412213</td>
    </tr>
    <tr>
      <td>Id0qucUCeB</td>
      <td>NaN</td>
      <td>-1.088763</td>
      <td>-0.637960</td>
      <td>1.000012</td>
    </tr>
    <tr>
      <td>MlJfZSTHJj</td>
      <td>-0.732373</td>
      <td>1.112463</td>
      <td>0.441055</td>
      <td>0.148622</td>
    </tr>
    <tr>
      <td>p74iwf5CWk</td>
      <td>0.880165</td>
      <td>NaN</td>
      <td>0.319050</td>
      <td>-0.100719</td>
    </tr>
    <tr>
      <td>k4wEIqvtKd</td>
      <td>-1.076164</td>
      <td>-1.052221</td>
      <td>NaN</td>
      <td>0.817084</td>
    </tr>
    <tr>
      <td>6gdeN8A4gg</td>
      <td>1.160957</td>
      <td>0.109959</td>
      <td>0.587556</td>
      <td>1.699185</td>
    </tr>
    <tr>
      <td>c0GHnKP4vD</td>
      <td>-1.127555</td>
      <td>0.933663</td>
      <td>0.527032</td>
      <td>-0.450704</td>
    </tr>
    <tr>
      <td>0yltZq25JS</td>
      <td>-0.177030</td>
      <td>0.225647</td>
      <td>0.413793</td>
      <td>1.491172</td>
    </tr>
    <tr>
      <td>BAGqtJef0S</td>
      <td>0.356970</td>
      <td>0.621645</td>
      <td>-1.182119</td>
      <td>-0.008877</td>
    </tr>
    <tr>
      <td>iOriKpo3am</td>
      <td>0.734046</td>
      <td>-0.004511</td>
      <td>0.921544</td>
      <td>0.719030</td>
    </tr>
    <tr>
      <td>Ogd3TaGzC0</td>
      <td>-0.076480</td>
      <td>-1.460346</td>
      <td>0.026264</td>
      <td>-0.775999</td>
    </tr>
    <tr>
      <td>XZp591LpQ7</td>
      <td>-0.998655</td>
      <td>NaN</td>
      <td>1.518520</td>
      <td>0.453103</td>
    </tr>
    <tr>
      <td>nvd2jdiBX5</td>
      <td>1.592946</td>
      <td>-1.306997</td>
      <td>-0.168547</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>ICJxEvHfx4</td>
      <td>-0.577977</td>
      <td>0.387412</td>
      <td>-0.594324</td>
      <td>0.294138</td>
    </tr>
    <tr>
      <td>mLyqvY1KVk</td>
      <td>0.901325</td>
      <td>-1.560872</td>
      <td>0.160145</td>
      <td>-1.170798</td>
    </tr>
    <tr>
      <td>23TmyM4Vjt</td>
      <td>0.291432</td>
      <td>-0.810652</td>
      <td>-0.184815</td>
      <td>0.211070</td>
    </tr>
    <tr>
      <td>J2JBz9nc2N</td>
      <td>1.818592</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.257770</td>
    </tr>
    <tr>
      <td>77utwjDcEg</td>
      <td>NaN</td>
      <td>0.183574</td>
      <td>0.111528</td>
      <td>1.381484</td>
    </tr>
    <tr>
      <td>yAg15dn3KU</td>
      <td>-0.259229</td>
      <td>-0.201913</td>
      <td>-1.912119</td>
      <td>-0.271751</td>
    </tr>
    <tr>
      <td>yRbk019rAz</td>
      <td>-0.913102</td>
      <td>-0.120397</td>
      <td>-0.721780</td>
      <td>0.725017</td>
    </tr>
    <tr>
      <td>U5oGxD3WnW</td>
      <td>-0.565617</td>
      <td>0.579980</td>
      <td>NaN</td>
      <td>0.031062</td>
    </tr>
    <tr>
      <td>sNGtSOERD7</td>
      <td>NaN</td>
      <td>0.074556</td>
      <td>1.525357</td>
      <td>-1.744255</td>
    </tr>
    <tr>
      <td>ubpFxokUim</td>
      <td>1.045186</td>
      <td>1.165908</td>
      <td>-1.138931</td>
      <td>-0.082937</td>
    </tr>
    <tr>
      <td>b6ZCgA1dZD</td>
      <td>1.620180</td>
      <td>0.962883</td>
      <td>0.966021</td>
      <td>1.709217</td>
    </tr>
    <tr>
      <td>3lYiadTEpV</td>
      <td>-0.147411</td>
      <td>0.950925</td>
      <td>-0.122691</td>
      <td>-0.973400</td>
    </tr>
    <tr>
      <td>qvn2AUqCgF</td>
      <td>1.207141</td>
      <td>0.611017</td>
      <td>-0.733806</td>
      <td>0.072065</td>
    </tr>
    <tr>
      <td>dqO35ESFKx</td>
      <td>2.076349</td>
      <td>0.519654</td>
      <td>NaN</td>
      <td>0.217811</td>
    </tr>
    <tr>
      <td>am4HPhiP5I</td>
      <td>-1.403172</td>
      <td>NaN</td>
      <td>0.797241</td>
      <td>-0.641350</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.util.testing.makeTimeDataFrame()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2000-01-03</td>
      <td>1.003781</td>
      <td>-0.356442</td>
      <td>-0.690759</td>
      <td>-2.543469</td>
    </tr>
    <tr>
      <td>2000-01-04</td>
      <td>0.307821</td>
      <td>-0.425427</td>
      <td>-0.134576</td>
      <td>0.824121</td>
    </tr>
    <tr>
      <td>2000-01-05</td>
      <td>-0.976996</td>
      <td>-1.113531</td>
      <td>-0.522290</td>
      <td>-0.567436</td>
    </tr>
    <tr>
      <td>2000-01-06</td>
      <td>0.438595</td>
      <td>0.129363</td>
      <td>-0.433556</td>
      <td>1.080008</td>
    </tr>
    <tr>
      <td>2000-01-07</td>
      <td>-0.001516</td>
      <td>-0.291726</td>
      <td>-1.008141</td>
      <td>-0.605705</td>
    </tr>
    <tr>
      <td>2000-01-10</td>
      <td>0.015357</td>
      <td>-1.016492</td>
      <td>-0.162590</td>
      <td>-2.282291</td>
    </tr>
    <tr>
      <td>2000-01-11</td>
      <td>0.523147</td>
      <td>0.587595</td>
      <td>0.451137</td>
      <td>-0.031271</td>
    </tr>
    <tr>
      <td>2000-01-12</td>
      <td>0.917431</td>
      <td>1.109996</td>
      <td>-0.561045</td>
      <td>0.305243</td>
    </tr>
    <tr>
      <td>2000-01-13</td>
      <td>1.150708</td>
      <td>-0.069645</td>
      <td>-0.561491</td>
      <td>-0.165408</td>
    </tr>
    <tr>
      <td>2000-01-14</td>
      <td>1.122621</td>
      <td>-1.035067</td>
      <td>1.201935</td>
      <td>-0.690251</td>
    </tr>
    <tr>
      <td>2000-01-17</td>
      <td>0.813662</td>
      <td>0.924708</td>
      <td>0.545710</td>
      <td>-0.002078</td>
    </tr>
    <tr>
      <td>2000-01-18</td>
      <td>1.379715</td>
      <td>-0.229399</td>
      <td>-0.726494</td>
      <td>0.493481</td>
    </tr>
    <tr>
      <td>2000-01-19</td>
      <td>0.622288</td>
      <td>-0.757303</td>
      <td>0.689828</td>
      <td>0.229719</td>
    </tr>
    <tr>
      <td>2000-01-20</td>
      <td>-0.680374</td>
      <td>1.011196</td>
      <td>-0.551390</td>
      <td>-1.713440</td>
    </tr>
    <tr>
      <td>2000-01-21</td>
      <td>-1.524046</td>
      <td>-1.010875</td>
      <td>-0.997332</td>
      <td>0.666093</td>
    </tr>
    <tr>
      <td>2000-01-24</td>
      <td>-0.941409</td>
      <td>1.025098</td>
      <td>-0.905963</td>
      <td>1.082745</td>
    </tr>
    <tr>
      <td>2000-01-25</td>
      <td>0.042881</td>
      <td>1.382499</td>
      <td>1.049391</td>
      <td>1.074299</td>
    </tr>
    <tr>
      <td>2000-01-26</td>
      <td>0.967413</td>
      <td>-0.923113</td>
      <td>2.007157</td>
      <td>0.628600</td>
    </tr>
    <tr>
      <td>2000-01-27</td>
      <td>-0.051459</td>
      <td>1.172377</td>
      <td>0.464885</td>
      <td>1.129185</td>
    </tr>
    <tr>
      <td>2000-01-28</td>
      <td>0.533879</td>
      <td>-1.320932</td>
      <td>0.503104</td>
      <td>-0.695140</td>
    </tr>
    <tr>
      <td>2000-01-31</td>
      <td>-1.133007</td>
      <td>0.437374</td>
      <td>0.505606</td>
      <td>-1.571305</td>
    </tr>
    <tr>
      <td>2000-02-01</td>
      <td>0.979238</td>
      <td>-0.721377</td>
      <td>0.234655</td>
      <td>-0.061103</td>
    </tr>
    <tr>
      <td>2000-02-02</td>
      <td>0.231642</td>
      <td>-0.627750</td>
      <td>-0.133623</td>
      <td>0.748030</td>
    </tr>
    <tr>
      <td>2000-02-03</td>
      <td>-1.431866</td>
      <td>1.060292</td>
      <td>-0.040169</td>
      <td>0.229246</td>
    </tr>
    <tr>
      <td>2000-02-04</td>
      <td>-0.350627</td>
      <td>-1.165422</td>
      <td>0.436717</td>
      <td>0.446511</td>
    </tr>
    <tr>
      <td>2000-02-07</td>
      <td>-1.886733</td>
      <td>0.926730</td>
      <td>0.604955</td>
      <td>-1.034671</td>
    </tr>
    <tr>
      <td>2000-02-08</td>
      <td>1.205402</td>
      <td>1.982943</td>
      <td>-0.653149</td>
      <td>-0.675256</td>
    </tr>
    <tr>
      <td>2000-02-09</td>
      <td>0.212245</td>
      <td>-0.374800</td>
      <td>-0.472396</td>
      <td>1.577047</td>
    </tr>
    <tr>
      <td>2000-02-10</td>
      <td>-0.732806</td>
      <td>-1.332025</td>
      <td>1.173203</td>
      <td>0.602202</td>
    </tr>
    <tr>
      <td>2000-02-11</td>
      <td>-0.681866</td>
      <td>-0.170969</td>
      <td>-1.366112</td>
      <td>0.779899</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.util.testing.makeMixedDataFrame()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>foo1</td>
      <td>2009-01-01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>foo2</td>
      <td>2009-01-02</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>foo3</td>
      <td>2009-01-05</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>foo4</td>
      <td>2009-01-06</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>foo5</td>
      <td>2009-01-07</td>
    </tr>
  </tbody>
</table>
</div>



74. Write a Pandas program to fill missing values in time series data.


```python
s74 = {"c1":[120, 130 ,140, 150, np.nan, 170], "c2":[7, np.nan, 10, np.nan, 5.5, 16.5]}
```


```python
pd.util.testing.makeDateIndex()[:6]
```




    DatetimeIndex(['2000-01-03', '2000-01-04', '2000-01-05', '2000-01-06', '2000-01-07', '2000-01-10'], dtype='datetime64[ns]', freq='B')




```python
df74=pd.DataFrame(s74,index=pd.util.testing.makeDateIndex()[:6])
```


```python
df74
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
      <th>c1</th>
      <th>c2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2000-01-03</td>
      <td>120.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <td>2000-01-04</td>
      <td>130.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2000-01-05</td>
      <td>140.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <td>2000-01-06</td>
      <td>150.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2000-01-07</td>
      <td>NaN</td>
      <td>5.5</td>
    </tr>
    <tr>
      <td>2000-01-10</td>
      <td>170.0</td>
      <td>16.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df74.interpolate()
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
      <th>c1</th>
      <th>c2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2000-01-03</td>
      <td>120.0</td>
      <td>7.00</td>
    </tr>
    <tr>
      <td>2000-01-04</td>
      <td>130.0</td>
      <td>8.50</td>
    </tr>
    <tr>
      <td>2000-01-05</td>
      <td>140.0</td>
      <td>10.00</td>
    </tr>
    <tr>
      <td>2000-01-06</td>
      <td>150.0</td>
      <td>7.75</td>
    </tr>
    <tr>
      <td>2000-01-07</td>
      <td>160.0</td>
      <td>5.50</td>
    </tr>
    <tr>
      <td>2000-01-10</td>
      <td>170.0</td>
      <td>16.50</td>
    </tr>
  </tbody>
</table>
</div>



75. Write a Pandas program to use a local variable within a query.


```python
df75 = pd.DataFrame({'W':[68,75,86,80,66],'X':[78,85,96,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]});

```


```python
df75
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>68</td>
      <td>78</td>
      <td>84</td>
      <td>86</td>
    </tr>
    <tr>
      <td>1</td>
      <td>75</td>
      <td>85</td>
      <td>94</td>
      <td>97</td>
    </tr>
    <tr>
      <td>2</td>
      <td>86</td>
      <td>96</td>
      <td>89</td>
      <td>96</td>
    </tr>
    <tr>
      <td>3</td>
      <td>80</td>
      <td>80</td>
      <td>83</td>
      <td>72</td>
    </tr>
    <tr>
      <td>4</td>
      <td>66</td>
      <td>86</td>
      <td>86</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
maxx = df75["W"].max()
```


```python
df75.query("W < @maxx")
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>68</td>
      <td>78</td>
      <td>84</td>
      <td>86</td>
    </tr>
    <tr>
      <td>1</td>
      <td>75</td>
      <td>85</td>
      <td>94</td>
      <td>97</td>
    </tr>
    <tr>
      <td>3</td>
      <td>80</td>
      <td>80</td>
      <td>83</td>
      <td>72</td>
    </tr>
    <tr>
      <td>4</td>
      <td>66</td>
      <td>86</td>
      <td>86</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Solution 2 (without Query)
df75[df75.W<df75.W.max()]
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>68</td>
      <td>78</td>
      <td>84</td>
      <td>86</td>
    </tr>
    <tr>
      <td>1</td>
      <td>75</td>
      <td>85</td>
      <td>94</td>
      <td>97</td>
    </tr>
    <tr>
      <td>3</td>
      <td>80</td>
      <td>80</td>
      <td>83</td>
      <td>72</td>
    </tr>
    <tr>
      <td>4</td>
      <td>66</td>
      <td>86</td>
      <td>86</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>



76. Write a Pandas program to clean object column with mixed data of a given DataFrame using regular expression.


```python
df76 = pd.DataFrame({"agent": ["a001", "a002", "a003", "a003", "a004"], "purchase":[4500.00, 7500.00, "$3000.25", "$1250.35", "9000.00"]})

```


```python
df76
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
      <th>agent</th>
      <th>purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>a001</td>
      <td>4500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>a002</td>
      <td>7500</td>
    </tr>
    <tr>
      <td>2</td>
      <td>a003</td>
      <td>$3000.25</td>
    </tr>
    <tr>
      <td>3</td>
      <td>a003</td>
      <td>$1250.35</td>
    </tr>
    <tr>
      <td>4</td>
      <td>a004</td>
      <td>9000.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
df76.purchase.apply(type)
```




    0    <class 'float'>
    1    <class 'float'>
    2      <class 'str'>
    3      <class 'str'>
    4      <class 'str'>
    Name: purchase, dtype: object




```python
df76.purchase.replace("[$]","",regex=True).astype('float').apply(type)
```




    0    <class 'float'>
    1    <class 'float'>
    2    <class 'float'>
    3    <class 'float'>
    4    <class 'float'>
    Name: purchase, dtype: object



77. Write a Pandas program to get the numeric representation of an array by identifying distinct values of a given column of a dataframe.


```python
df77 = pd.DataFrame({
    'Name': ['Alberto Franco','Gino Mcneill','Ryan Parkes', 'Eesha Hinton', 'Gino Mcneill'],
    'Date_Of_Birth ': ['17/05/2002','16/02/1999','25/09/1998','11/05/2002','15/09/1997'],
    'Age': [18.5, 21.2, 22.5, 22, 23]
})
```


```python
df77
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
      <th>Name</th>
      <th>Date_Of_Birth</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Alberto Franco</td>
      <td>17/05/2002</td>
      <td>18.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Mcneill</td>
      <td>16/02/1999</td>
      <td>21.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Ryan Parkes</td>
      <td>25/09/1998</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gino Mcneill</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
a77_1=df77.Name.unique()
```


```python
d77={}
for ind,name in df77.iterrows():
    print(ind,name[0])
    if name[0] not in d77.keys():
        d77[name[0]]=ind
```

    0 Alberto Franco
    1 Gino Mcneill
    2 Ryan Parkes
    3 Eesha Hinton
    4 Gino Mcneill



```python
d77
```




    {'Alberto Franco': 0, 'Gino Mcneill': 1, 'Ryan Parkes': 2, 'Eesha Hinton': 3}




```python
df77.replace({'Name':d77})
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
      <th>Name</th>
      <th>Date_Of_Birth</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>17/05/2002</td>
      <td>18.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>16/02/1999</td>
      <td>21.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>25/09/1998</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#solution 1
df77['Name'].map(d77).tolist()
```




    [0, 1, 2, 3, 1]




```python
#Solution 2
df77.replace({'Name':d77}).Name.tolist()
```




    [0, 1, 2, 3, 1]




```python
#one step Solution
pd.factorize(df77.Name)
```




    (array([0, 1, 2, 3, 1], dtype=int64),
     Index(['Alberto Franco', 'Gino Mcneill', 'Ryan Parkes', 'Eesha Hinton'], dtype='object'))



78. Write a Pandas program to replace the current value in a dataframe column based on last largest value. If the current value is less than last largest value replaces the value with 0.


```python
df78=pd.DataFrame({'rnum':[23, 21, 27, 22, 34, 33, 34, 31, 25, 22, 34, 19, 31, 32, 19]})
```


```python
df78
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
      <th>rnum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <td>1</td>
      <td>21</td>
    </tr>
    <tr>
      <td>2</td>
      <td>27</td>
    </tr>
    <tr>
      <td>3</td>
      <td>22</td>
    </tr>
    <tr>
      <td>4</td>
      <td>34</td>
    </tr>
    <tr>
      <td>5</td>
      <td>33</td>
    </tr>
    <tr>
      <td>6</td>
      <td>34</td>
    </tr>
    <tr>
      <td>7</td>
      <td>31</td>
    </tr>
    <tr>
      <td>8</td>
      <td>25</td>
    </tr>
    <tr>
      <td>9</td>
      <td>22</td>
    </tr>
    <tr>
      <td>10</td>
      <td>34</td>
    </tr>
    <tr>
      <td>11</td>
      <td>19</td>
    </tr>
    <tr>
      <td>12</td>
      <td>31</td>
    </tr>
    <tr>
      <td>13</td>
      <td>32</td>
    </tr>
    <tr>
      <td>14</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
a78_1=[-np.inf]
for ele in df78.rnum:
    if ele<np.amax(a78_1):
        a78_1.append(0)
    else:
        a78_1.append(ele)
    #print(ele)
print(a78_1[1:])
```

    [23, 0, 27, 0, 34, 0, 34, 0, 0, 0, 34, 0, 0, 0, 0]



```python
#Solution 2
df78.rnum.where(df78.rnum.eq(df78.rnum.cummax()),0)
```




    0     23
    1      0
    2     27
    3      0
    4     34
    5      0
    6     34
    7      0
    8      0
    9      0
    10    34
    11     0
    12     0
    13     0
    14     0
    Name: rnum, dtype: int64



79. Write a Pandas program to create a DataFrame from the clipboard (data from an Excel spreadsheet or a Google Sheet).


```python
df79=pd.read_clipboard()
```


```python
df79
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
      <th>Python</th>
      <th>Pandas</th>
      <th>DataFrame</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



80. Write a Pandas program to check for inequality of two given DataFrames.


```python
df80_1 = pd.DataFrame({'W':[68,75,86,80,None],'X':[78,85,None,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]});
df80_2 = pd.DataFrame({'W':[78,75,86,80,None],'X':[78,85,96,80,76], 'Y':[84,84,89,83,86],'Z':[86,97,96,72,83]});

```


```python
df80_1!=df80_2
#~(df80_1==df80_2)
#df80_1.ne(df80_2)
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



81. Write a Pandas program to get lowest n records within each group of a given DataFrame.


```python
df81 = pd.DataFrame({'col1': [1, 2, 3, 4, 7, 11], 'col2': [4, 5, 6, 9, 5, 0], 'col3': [7, 5, 8, 12, 1,11]})

```


```python
df81.nsmallest(3,'col1')
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df81.nsmallest(3,'col2')
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5</td>
      <td>11</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df81.nsmallest(3,'col3')
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df81.nsmallest(3,columns=['col2','col1']) # It is taking first passed column into account even if we pass multiple columns
```
