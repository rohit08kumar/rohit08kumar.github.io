---
title: "Python Pandas - Joining and merging DataFrame"
date: 2021-05-17
tags: [Data Science, Pandas, Python Learning]
header:
  image: "/images/Feature_engineering.jpeg"
excerpt: "Data Science, Pandas, Python Learning"
mathjax: "true"
---


# Exercise - Pandas Joining and merging DataFrame


```python
import pandas as pd
import numpy as np
```

1. Write a Pandas program to join the two given dataframes along rows and assign all data.


```python
df1_1 = pd.DataFrame({
        'student_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
         'name': ['Danniella Fenton', 'Ryder Storey', 'Bryce Jensen', 'Ed Bernal', 'Kwame Morin'],
        'marks': [200, 210, 190, 222, 199]})

df1_2 = pd.DataFrame({
        'student_id': ['S4', 'S5', 'S6', 'S7', 'S8'],
        'name': ['Scarlette Fisher', 'Carla Williamson', 'Dante Morse', 'Kaiser William', 'Madeeha Preston'],
        'marks': [201, 200, 198, 219, 201]})
```


```python
pd.concat([df1_1,df1_2],axis=0)
#df1_1.append(df1_2,ignore_index=False)
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
      <th>student_id</th>
      <th>name</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>S1</td>
      <td>Danniella Fenton</td>
      <td>200</td>
    </tr>
    <tr>
      <td>1</td>
      <td>S2</td>
      <td>Ryder Storey</td>
      <td>210</td>
    </tr>
    <tr>
      <td>2</td>
      <td>S3</td>
      <td>Bryce Jensen</td>
      <td>190</td>
    </tr>
    <tr>
      <td>3</td>
      <td>S4</td>
      <td>Ed Bernal</td>
      <td>222</td>
    </tr>
    <tr>
      <td>4</td>
      <td>S5</td>
      <td>Kwame Morin</td>
      <td>199</td>
    </tr>
    <tr>
      <td>0</td>
      <td>S4</td>
      <td>Scarlette Fisher</td>
      <td>201</td>
    </tr>
    <tr>
      <td>1</td>
      <td>S5</td>
      <td>Carla Williamson</td>
      <td>200</td>
    </tr>
    <tr>
      <td>2</td>
      <td>S6</td>
      <td>Dante Morse</td>
      <td>198</td>
    </tr>
    <tr>
      <td>3</td>
      <td>S7</td>
      <td>Kaiser William</td>
      <td>219</td>
    </tr>
    <tr>
      <td>4</td>
      <td>S8</td>
      <td>Madeeha Preston</td>
      <td>201</td>
    </tr>
  </tbody>
</table>
</div>



2. Write a Pandas program to join the two given dataframes along columns and assign all data.


```python
pd.concat([df1_1,df1_2],axis=1)
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
      <th>student_id</th>
      <th>name</th>
      <th>marks</th>
      <th>student_id</th>
      <th>name</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>S1</td>
      <td>Danniella Fenton</td>
      <td>200</td>
      <td>S4</td>
      <td>Scarlette Fisher</td>
      <td>201</td>
    </tr>
    <tr>
      <td>1</td>
      <td>S2</td>
      <td>Ryder Storey</td>
      <td>210</td>
      <td>S5</td>
      <td>Carla Williamson</td>
      <td>200</td>
    </tr>
    <tr>
      <td>2</td>
      <td>S3</td>
      <td>Bryce Jensen</td>
      <td>190</td>
      <td>S6</td>
      <td>Dante Morse</td>
      <td>198</td>
    </tr>
    <tr>
      <td>3</td>
      <td>S4</td>
      <td>Ed Bernal</td>
      <td>222</td>
      <td>S7</td>
      <td>Kaiser William</td>
      <td>219</td>
    </tr>
    <tr>
      <td>4</td>
      <td>S5</td>
      <td>Kwame Morin</td>
      <td>199</td>
      <td>S8</td>
      <td>Madeeha Preston</td>
      <td>201</td>
    </tr>
  </tbody>
</table>
</div>



3. Write a Pandas program to append rows to an existing DataFrame and display the combined data.


```python
s3 = pd.Series(['S6', 'Scarlette Fisher', 205], index=['student_id', 'name', 'marks'])
s3
```




    student_id                  S6
    name          Scarlette Fisher
    marks                      205
    dtype: object




```python
df1_1.append(s3,ignore_index=True)
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
      <th>student_id</th>
      <th>name</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>S1</td>
      <td>Danniella Fenton</td>
      <td>200</td>
    </tr>
    <tr>
      <td>1</td>
      <td>S2</td>
      <td>Ryder Storey</td>
      <td>210</td>
    </tr>
    <tr>
      <td>2</td>
      <td>S3</td>
      <td>Bryce Jensen</td>
      <td>190</td>
    </tr>
    <tr>
      <td>3</td>
      <td>S4</td>
      <td>Ed Bernal</td>
      <td>222</td>
    </tr>
    <tr>
      <td>4</td>
      <td>S5</td>
      <td>Kwame Morin</td>
      <td>199</td>
    </tr>
    <tr>
      <td>5</td>
      <td>S6</td>
      <td>Scarlette Fisher</td>
      <td>205</td>
    </tr>
  </tbody>
</table>
</div>



4. Write a Pandas program to append a list of dictioneries or series to a existing DataFrame and display the combined data.


```python
d4 = [{'student_id': 'S6', 'name': 'Scarlette Fisher', 'marks': 203},
         {'student_id': 'S7', 'name': 'Bryce Jensen', 'marks': 207}]
d4
```




    [{'student_id': 'S6', 'name': 'Scarlette Fisher', 'marks': 203},
     {'student_id': 'S7', 'name': 'Bryce Jensen', 'marks': 207}]




```python
df1_1.append(d4,ignore_index=True)
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
      <th>student_id</th>
      <th>name</th>
      <th>marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>S1</td>
      <td>Danniella Fenton</td>
      <td>200</td>
    </tr>
    <tr>
      <td>1</td>
      <td>S2</td>
      <td>Ryder Storey</td>
      <td>210</td>
    </tr>
    <tr>
      <td>2</td>
      <td>S3</td>
      <td>Bryce Jensen</td>
      <td>190</td>
    </tr>
    <tr>
      <td>3</td>
      <td>S4</td>
      <td>Ed Bernal</td>
      <td>222</td>
    </tr>
    <tr>
      <td>4</td>
      <td>S5</td>
      <td>Kwame Morin</td>
      <td>199</td>
    </tr>
    <tr>
      <td>5</td>
      <td>S6</td>
      <td>Scarlette Fisher</td>
      <td>203</td>
    </tr>
    <tr>
      <td>6</td>
      <td>S7</td>
      <td>Bryce Jensen</td>
      <td>207</td>
    </tr>
  </tbody>
</table>
</div>



5. Write a Pandas program to join the two given dataframes along rows and merge with another dataframe along the common column id.


```python
df5_1 = pd.DataFrame({
        'student_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
         'name': ['Danniella Fenton', 'Ryder Storey', 'Bryce Jensen', 'Ed Bernal', 'Kwame Morin'],
        'marks': [200, 210, 190, 222, 199]})
df5_2 = pd.DataFrame({
        'student_id': ['S4', 'S5', 'S6', 'S7', 'S8'],
        'name': ['Scarlette Fisher', 'Carla Williamson', 'Dante Morse', 'Kaiser William', 'Madeeha Preston'],
        'marks': [201, 200, 198, 219, 201]})

df5_3 = pd.DataFrame({
        'student_id': ['S1', 'S2', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'],
        'exam_id': [23, 45, 12, 67, 21, 55, 33, 14, 56, 83, 88, 12]})
```


```python
pd.merge(pd.concat([df5_1,df5_2],ignore_index=True),df5_3,how='inner',on=['student_id'])
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
      <th>student_id</th>
      <th>name</th>
      <th>marks</th>
      <th>exam_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>S1</td>
      <td>Danniella Fenton</td>
      <td>200</td>
      <td>23</td>
    </tr>
    <tr>
      <td>1</td>
      <td>S2</td>
      <td>Ryder Storey</td>
      <td>210</td>
      <td>45</td>
    </tr>
    <tr>
      <td>2</td>
      <td>S3</td>
      <td>Bryce Jensen</td>
      <td>190</td>
      <td>12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>S4</td>
      <td>Ed Bernal</td>
      <td>222</td>
      <td>67</td>
    </tr>
    <tr>
      <td>4</td>
      <td>S4</td>
      <td>Scarlette Fisher</td>
      <td>201</td>
      <td>67</td>
    </tr>
    <tr>
      <td>5</td>
      <td>S5</td>
      <td>Kwame Morin</td>
      <td>199</td>
      <td>21</td>
    </tr>
    <tr>
      <td>6</td>
      <td>S5</td>
      <td>Carla Williamson</td>
      <td>200</td>
      <td>21</td>
    </tr>
    <tr>
      <td>7</td>
      <td>S7</td>
      <td>Kaiser William</td>
      <td>219</td>
      <td>55</td>
    </tr>
    <tr>
      <td>8</td>
      <td>S8</td>
      <td>Madeeha Preston</td>
      <td>201</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>



6. Write a Pandas program to join the two dataframes using the common column of both dataframes.


```python
df5_1.merge(df5_2,how='outer',on='student_id')
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
      <th>student_id</th>
      <th>name_x</th>
      <th>marks_x</th>
      <th>name_y</th>
      <th>marks_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>S1</td>
      <td>Danniella Fenton</td>
      <td>200.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>S2</td>
      <td>Ryder Storey</td>
      <td>210.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>S3</td>
      <td>Bryce Jensen</td>
      <td>190.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>S4</td>
      <td>Ed Bernal</td>
      <td>222.0</td>
      <td>Scarlette Fisher</td>
      <td>201.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>S5</td>
      <td>Kwame Morin</td>
      <td>199.0</td>
      <td>Carla Williamson</td>
      <td>200.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>S6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dante Morse</td>
      <td>198.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>S7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kaiser William</td>
      <td>219.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>S8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Madeeha Preston</td>
      <td>201.0</td>
    </tr>
  </tbody>
</table>
</div>



7. Write a Pandas program to join the two dataframes with matching records from both sides where available.


```python
df5_1.merge(df5_2,how='outer',on='student_id')
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
      <th>student_id</th>
      <th>name_x</th>
      <th>marks_x</th>
      <th>name_y</th>
      <th>marks_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>S1</td>
      <td>Danniella Fenton</td>
      <td>200.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>S2</td>
      <td>Ryder Storey</td>
      <td>210.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>S3</td>
      <td>Bryce Jensen</td>
      <td>190.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>S4</td>
      <td>Ed Bernal</td>
      <td>222.0</td>
      <td>Scarlette Fisher</td>
      <td>201.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>S5</td>
      <td>Kwame Morin</td>
      <td>199.0</td>
      <td>Carla Williamson</td>
      <td>200.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>S6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dante Morse</td>
      <td>198.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>S7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kaiser William</td>
      <td>219.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>S8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Madeeha Preston</td>
      <td>201.0</td>
    </tr>
  </tbody>
</table>
</div>



8. Write a Pandas program to join (left join) the two dataframes using keys from left dataframe only.


```python
d8_1 = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'P': ['P0', 'P1', 'P2', 'P3'],
                     'Q': ['Q0', 'Q1', 'Q2', 'Q3']})
d8_2 = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'R': ['R0', 'R1', 'R2', 'R3'],
                      'S': ['S0', 'S1', 'S2', 'S3']})
```


```python
pd.merge(d8_1,d8_2,how='left',on=['key1','key2'])
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
      <th>key1</th>
      <th>key2</th>
      <th>P</th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>K0</td>
      <td>K0</td>
      <td>P0</td>
      <td>Q0</td>
      <td>R0</td>
      <td>S0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>K0</td>
      <td>K1</td>
      <td>P1</td>
      <td>Q1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>K1</td>
      <td>K0</td>
      <td>P2</td>
      <td>Q2</td>
      <td>R1</td>
      <td>S1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>K1</td>
      <td>K0</td>
      <td>P2</td>
      <td>Q2</td>
      <td>R2</td>
      <td>S2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>K2</td>
      <td>K1</td>
      <td>P3</td>
      <td>Q3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



9. Write a Pandas program to join two dataframes using keys from right dataframe only.


```python
pd.merge(d8_1,d8_2,how='right',on=['key1','key2'])
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
      <th>key1</th>
      <th>key2</th>
      <th>P</th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>K0</td>
      <td>K0</td>
      <td>P0</td>
      <td>Q0</td>
      <td>R0</td>
      <td>S0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>K1</td>
      <td>K0</td>
      <td>P2</td>
      <td>Q2</td>
      <td>R1</td>
      <td>S1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>K1</td>
      <td>K0</td>
      <td>P2</td>
      <td>Q2</td>
      <td>R2</td>
      <td>S2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>K2</td>
      <td>K0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>R3</td>
      <td>S3</td>
    </tr>
  </tbody>
</table>
</div>



10. Write a Pandas program to merge two given datasets using multiple join keys.


```python
pd.merge(d8_1,d8_2,on=['key1','key2'])
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
      <th>key1</th>
      <th>key2</th>
      <th>P</th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>K0</td>
      <td>K0</td>
      <td>P0</td>
      <td>Q0</td>
      <td>R0</td>
      <td>S0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>K1</td>
      <td>K0</td>
      <td>P2</td>
      <td>Q2</td>
      <td>R1</td>
      <td>S1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>K1</td>
      <td>K0</td>
      <td>P2</td>
      <td>Q2</td>
      <td>R2</td>
      <td>S2</td>
    </tr>
  </tbody>
</table>
</div>



11. Write a Pandas program to create a new DataFrame based on existing series, using specified argument and override the existing columns names.


```python
s11_1 = pd.Series([0, 1, 2, 3], name='col1')
s11_2 = pd.Series([0, 1, 2, 3])
s11_3 = pd.Series([0, 1, 4, 5], name='col3')
```


```python
pd.concat([s11_1,s11_2,s11_3],axis=1,keys=['Column1','Columns2','Column3'])
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
      <th>Columns2</th>
      <th>Column3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



12. Write a Pandas program to create a combination from two dataframes where a column id combination appears more than once in both dataframes.


```python
df12_1 = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'P': ['P0', 'P1', 'P2', 'P3'],
                     'Q': ['Q0', 'Q1', 'Q2', 'Q3']})
df12_2 = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'R': ['R0', 'R1', 'R2', 'R3'],
                      'S': ['S0', 'S1', 'S2', 'S3']})
```


```python
pd.merge(df12_1,df12_2,on='key1')
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
      <th>key1</th>
      <th>key2_x</th>
      <th>P</th>
      <th>Q</th>
      <th>key2_y</th>
      <th>R</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>K0</td>
      <td>K0</td>
      <td>P0</td>
      <td>Q0</td>
      <td>K0</td>
      <td>R0</td>
      <td>S0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>K0</td>
      <td>K1</td>
      <td>P1</td>
      <td>Q1</td>
      <td>K0</td>
      <td>R0</td>
      <td>S0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>K1</td>
      <td>K0</td>
      <td>P2</td>
      <td>Q2</td>
      <td>K0</td>
      <td>R1</td>
      <td>S1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>K1</td>
      <td>K0</td>
      <td>P2</td>
      <td>Q2</td>
      <td>K0</td>
      <td>R2</td>
      <td>S2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>K2</td>
      <td>K1</td>
      <td>P3</td>
      <td>Q3</td>
      <td>K0</td>
      <td>R3</td>
      <td>S3</td>
    </tr>
  </tbody>
</table>
</div>



13. Write a Pandas program to combine the columns of two potentially differently-indexed DataFrames into a single result DataFrame.


```python
df13_1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                      'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])

df13_2 = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])
```


```python
df13_1.join(df13_2,how='outer')
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
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <td>K1</td>
      <td>A1</td>
      <td>B1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>K2</td>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <td>K3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>



14. Write a Pandas program to merge two given dataframes with different columns.


```python
df14_1 = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'P': ['P0', 'P1', 'P2', 'P3'],
                     'Q': ['Q0', 'Q1', 'Q2', 'Q3']})
df14_2 = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'R': ['R0', 'R1', 'R2', 'R3'],
                      'S': ['S0', 'S1', 'S2', 'S3']})
```


```python
pd.concat([df14_1,df14_2],axis=0,sort=False,ignore_index=True)
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
      <th>key1</th>
      <th>key2</th>
      <th>P</th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>K0</td>
      <td>K0</td>
      <td>P0</td>
      <td>Q0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>K0</td>
      <td>K1</td>
      <td>P1</td>
      <td>Q1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>K1</td>
      <td>K0</td>
      <td>P2</td>
      <td>Q2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>K2</td>
      <td>K1</td>
      <td>P3</td>
      <td>Q3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>K0</td>
      <td>K0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>R0</td>
      <td>S0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>K1</td>
      <td>K0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>R1</td>
      <td>S1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>K1</td>
      <td>K0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>R2</td>
      <td>S2</td>
    </tr>
    <tr>
      <td>7</td>
      <td>K2</td>
      <td>K0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>R3</td>
      <td>S3</td>
    </tr>
  </tbody>
</table>
</div>



15. Write a Pandas program to Combine two DataFrame objects by filling null values in one DataFrame with non-null values from other DataFrame.


```python
df15_1 = pd.DataFrame({'A': [None, 0, None], 'B': [3, 4, 5]})
df15_2 = pd.DataFrame({'A': [1, 1, 3], 'B': [3, None, 3]})
```


```python
df15_1.combine_first(df15_2)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>4</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
