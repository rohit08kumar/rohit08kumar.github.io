# Exercise - Pandas String and Regular Expression


```python
import pandas as pd
import numpy as np
import re
```

1. Write a Pandas program to convert all the string values to upper, lower cases in a given pandas series. Also find the length of the string values.


```python
s1 = pd.Series(['X', 'Y', 'Z', 'Aaba', 'Baca', np.nan, 'CABA', None, 'bird', 'horse', 'dog'])
```


```python
s1.str.upper()
```




    0         X
    1         Y
    2         Z
    3      AABA
    4      BACA
    5       NaN
    6      CABA
    7      None
    8      BIRD
    9     HORSE
    10      DOG
    dtype: object




```python
s1.str.lower()
```




    0         x
    1         y
    2         z
    3      aaba
    4      baca
    5       NaN
    6      caba
    7      None
    8      bird
    9     horse
    10      dog
    dtype: object




```python
s1.str.len()
```




    0     1.0
    1     1.0
    2     1.0
    3     4.0
    4     4.0
    5     NaN
    6     4.0
    7     NaN
    8     4.0
    9     5.0
    10    3.0
    dtype: float64



2. Write a Pandas program to remove whitespaces, left sided whitespaces and right sided whitespaces of the string values of a given pandas series.


```python
s2 = pd.Index([' Green', 'Black ', ' Red ', 'White', ' Pink '])
```


```python
s2.str.strip()
```




    Index(['Green', 'Black', 'Red', 'White', 'Pink'], dtype='object')




```python
s2.str.lstrip()
```




    Index(['Green', 'Black ', 'Red ', 'White', 'Pink '], dtype='object')




```python
s2.str.rstrip()
```




    Index([' Green', 'Black', ' Red', 'White', ' Pink'], dtype='object')



3. Write a Pandas program to add leading zeros to the integer column in a pandas series and makes the length of the field to 8 digit.


```python
df3=pd.DataFrame({'amount': [10, 250, 3000, 40000, 500000]})
```


```python
#Solution 1
df3['amount'].apply(lambda x:'{0:0>8}'.format(x))
```




    0    00000010
    1    00000250
    2    00003000
    3    00040000
    4    00500000
    Name: amount, dtype: object




```python
#Solution 2
df3['amount'].apply(lambda x:str(x).zfill(8))
```




    0    00000010
    1    00000250
    2    00003000
    3    00040000
    4    00500000
    Name: amount, dtype: object



4. Write a Pandas program to add leading zeros to the character column in a pandas series and makes the length of the field to 8 digit.


```python
df4 = pd.DataFrame({'amount': ['10', '250', '3000', '40000', '500000']})
```


```python
#solution 1
df3.amount.apply(lambda x:'{0:0>8}'.format(x))
```




    0    00000010
    1    00000250
    2    00003000
    3    00040000
    4    00500000
    Name: amount, dtype: object




```python
#Solution 2
df4.amount.apply(lambda x:x.zfill(8))
```




    0    00000010
    1    00000250
    2    00003000
    3    00040000
    4    00500000
    Name: amount, dtype: object



5. Write a Pandas program to capitalize all the string values of specified columns of a given DataFrame.


```python
df5 = pd.DataFrame({
    'name': ['alberto','gino','ryan', 'Eesha', 'syed'],
    'date_of_birth ': ['17/05/2002','16/02/1999','25/09/1998','11/05/2002','15/09/1997'],
    'age': [18.5, 21.2, 22.5, 22, 23]
})
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
      <th>name</th>
      <th>date_of_birth</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>alberto</td>
      <td>17/05/2002</td>
      <td>18.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>gino</td>
      <td>16/02/1999</td>
      <td>21.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>ryan</td>
      <td>25/09/1998</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha</td>
      <td>11/05/2002</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>syed</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df5.name.str.capitalize()
```




    0    Alberto
    1       Gino
    2       Ryan
    3      Eesha
    4       Syed
    Name: name, dtype: object



6. Write a Pandas program to count of occurrence of a specified substring in a DataFrame column.


```python
df6 = pd.DataFrame({
    'name_code': ['c001','c002','c022', 'c2002', 'c2222'],
    'date_of_birth ': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'age': [18.5, 21.2, 22.5, 22, 23]
})
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
      <th>name_code</th>
      <th>date_of_birth</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c001</td>
      <td>12/05/2002</td>
      <td>18.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c002</td>
      <td>16/02/1999</td>
      <td>21.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>c022</td>
      <td>25/09/1998</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c2002</td>
      <td>12/02/2022</td>
      <td>22.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c2222</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("\nCount occurrence of 2 in name_code column:")

df6['count']=df6.name_code.apply(lambda x:x.count('2'))
df6
```

    
    Count occurrence of 2 in name_code column:





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
      <th>name_code</th>
      <th>date_of_birth</th>
      <th>age</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c001</td>
      <td>12/05/2002</td>
      <td>18.5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c002</td>
      <td>16/02/1999</td>
      <td>21.2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>c022</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c2002</td>
      <td>12/02/2022</td>
      <td>22.0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c2222</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



7. Write a Pandas program to find the index of a given substring of a DataFrame column.


```python
df7 = pd.DataFrame({
    'name_code': ['c001','c002','c022', 'c2002', 'c2222'],
    'date_of_birth ': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'age': [18.5, 21.2, 22.5, 22, 23]
})
```


```python
print("\nCount occurrence of 22 in name_code column:")
df7['Occur_22']=df7.name_code.apply(lambda x: 'Not Available' if x.find('22')==-1 else x.find('22'))
#df7['Occur_22'] = list(map(lambda x: x.find('22'), df7['name_code']))
df7
```

    
    Count occurrence of 22 in name_code column:





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
      <th>name_code</th>
      <th>date_of_birth</th>
      <th>age</th>
      <th>Occur_22</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c001</td>
      <td>12/05/2002</td>
      <td>18.5</td>
      <td>Not Available</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c002</td>
      <td>16/02/1999</td>
      <td>21.2</td>
      <td>Not Available</td>
    </tr>
    <tr>
      <td>2</td>
      <td>c022</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c2002</td>
      <td>12/02/2022</td>
      <td>22.0</td>
      <td>Not Available</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c2222</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



8. Write a Pandas program to find the index of a substring of DataFrame within provided index boundary (beginning and end position).


```python
df8 = pd.DataFrame({
    'name_code': ['c0001','1000c','b00c2', 'b2c02', 'c2222'],
    'date_of_birth ': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'age': [18.5, 21.2, 22.5, 22, 23]
})
```


```python
df8['occur_c_within_bound']=df8.name_code.apply(lambda x:x.find('c',0,4)) # 0 is staring index, 4 is end position
df8
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
      <th>name_code</th>
      <th>date_of_birth</th>
      <th>age</th>
      <th>occur_c_within_bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c0001</td>
      <td>12/05/2002</td>
      <td>18.5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1000c</td>
      <td>16/02/1999</td>
      <td>21.2</td>
      <td>-1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>b00c2</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>b2c02</td>
      <td>12/02/2022</td>
      <td>22.0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c2222</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



9. Write a Pandas program to check whether alpha numeric values present in a given column of a DataFrame.


```python
df9 = pd.DataFrame({
    'name_code': ['Company','Company a001','Company 123', '1234', 'Company12'],
    'date_of_birth ': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'age': [18.5, 21.2, 22.5, 22, 23]
})
```


```python
df9['AphaNum_Flag']=df9.name_code.apply(lambda x:x.isalnum())
df9
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
      <th>name_code</th>
      <th>date_of_birth</th>
      <th>age</th>
      <th>AphaNum_Flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Company</td>
      <td>12/05/2002</td>
      <td>18.5</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Company a001</td>
      <td>16/02/1999</td>
      <td>21.2</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Company 123</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1234</td>
      <td>12/02/2022</td>
      <td>22.0</td>
      <td>True</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Company12</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



10. Write a Pandas program to check whether only alphabetic values present in a given column of a DataFrame.


```python
df10 = pd.DataFrame({
    'company_code': ['Company','Company a001','Company 123', 'abcd', 'Company 12'],
    'date_of_sale ': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]})
```


```python
df10['Alphabetic_flag']=df10.company_code.apply(lambda x: x.isalpha())
df10
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>Alphabetic_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Company</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Company a001</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Company 123</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>abcd</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>True</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Company 12</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



11. Write a Pandas program to check whether only numeric values present in a given column of a DataFrame


```python
df11 = pd.DataFrame({
    'company_code': ['Company','Company a001', '2055', 'abcd', '123345'],
    'date_of_sale ': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]})
```


```python
df11['Numeric_flag']=df11.company_code.apply(lambda x:x.isdigit())
df11
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>Numeric_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Company</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Company a001</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2055</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>True</td>
    </tr>
    <tr>
      <td>3</td>
      <td>abcd</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>123345</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



12. Write a Pandas program to check whether only lower case or upper case is present in a given column of a DataFrame.


```python
df12 = pd.DataFrame({
    'company_code': ['ABCD','EFGF', 'hhhh', 'abcd', 'EAWQaaa'],
    'date_of_sale ': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]})

```


```python
df12['islower']=df12.company_code.apply(lambda x:(x.islower()))
df12['isupper']=df12.company_code.apply(lambda x:(x.isupper()))
```


```python
df12
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>islower</th>
      <th>isupper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>ABCD</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <td>2</td>
      <td>hhhh</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>abcd</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>EAWQaaa</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



13. Write a Pandas program to check whether only proper case or title case is present in a given column of a DataFrame.



```python
df13 = pd.DataFrame({
    'company_code': ['ABCD','EFGF', 'Hhhh', 'abcd', 'EAWQaaa'],
    'date_of_sale ': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]})

```


```python
df13['istitle']=df13.company_code.apply(lambda x:x.istitle())
```


```python
df13
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>istitle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>ABCD</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Hhhh</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>True</td>
    </tr>
    <tr>
      <td>3</td>
      <td>abcd</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>EAWQaaa</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



14. Write a Pandas program to check whether only space is present in a given column of a DataFrame


```python
df14 = pd.DataFrame({
    'company_code': ['Abcd','EFGF ', '  ', 'abcd', ' '],
    'date_of_sale ': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]})
```


```python
df14['isspace']=df14.company_code.apply(lambda x:x.isspace())
#Alternative
#df14['isspace']=df14.company_code.str.isspace()
```


```python
df14
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>isspace</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td></td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>True</td>
    </tr>
    <tr>
      <td>3</td>
      <td>abcd</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td></td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



15. Write a Pandas program to get the length of the string present of a given column in a DataFrame.


```python
df15 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'skfsalf', 'sdfslew', 'safsdf'],
    'date_of_sale ': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]})
```


```python
df15['length_company']=df15.company_code.str.len()
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>length_company</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>4</td>
    </tr>
    <tr>
      <td>2</td>
      <td>skfsalf</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>7</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>7</td>
    </tr>
    <tr>
      <td>4</td>
      <td>safsdf</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



16. Write a Pandas program to get the length of the integer of a given column in a DataFrame.


```python
df16 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'skfsalf', 'sdfslew', 'safsdf'],
    'date_of_sale': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]
})
```


```python
df16['sale_amnt_len']=df16.sale_amount.astype('str').str.len()
```


```python
df16
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>sale_amnt_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>skfsalf</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>9</td>
    </tr>
    <tr>
      <td>4</td>
      <td>safsdf</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



17. Write a Pandas program to check if a specified column starts with a specified string in a DataFrame.


```python
df17 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'zefsalf', 'sdfslew', 'zekfsdf'],
    'date_of_sale': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]
})
```


```python
df17['company_code_starts_with']=df17.company_code.str.startswith('ze')
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>company_code_starts_with</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>zefsalf</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>True</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>zekfsdf</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



18. Write a Pandas program to swap the cases of a specified character column in a given DataFrame.


```python
df18 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'zefsalf', 'sdfslew', 'zekfsdf'],
    'date_of_sale': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]
})
```


```python
df18['swapcase_cc']=df18.company_code.str.swapcase()
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>swapcase_cc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>aBCD</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>efgf</td>
    </tr>
    <tr>
      <td>2</td>
      <td>zefsalf</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>ZEFSALF</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>SDFSLEW</td>
    </tr>
    <tr>
      <td>4</td>
      <td>zekfsdf</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>ZEKFSDF</td>
    </tr>
  </tbody>
</table>
</div>



19. Write a Pandas program to convert a specified character column in upper/lower cases in a given DataFrame.


```python
df19 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'zefsalf', 'sdfslew', 'zekfsdf'],
    'date_of_sale': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]
})
```


```python
df19['upper_cc']=df19.company_code.str.upper()
df19['lower_cc']=df19.company_code.str.lower()
df19
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>upper_cc</th>
      <th>lower_cc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>ABCD</td>
      <td>abcd</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>EFGF</td>
      <td>efgf</td>
    </tr>
    <tr>
      <td>2</td>
      <td>zefsalf</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>ZEFSALF</td>
      <td>zefsalf</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>SDFSLEW</td>
      <td>sdfslew</td>
    </tr>
    <tr>
      <td>4</td>
      <td>zekfsdf</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>ZEKFSDF</td>
      <td>zekfsdf</td>
    </tr>
  </tbody>
</table>
</div>



20. Write a Pandas program to convert a specified character column in title case in a given DataFrame.


```python
df20 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'zefsalf', 'sdfslew', 'zekfsdf'],
    'date_of_sale': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]
})
```


```python
df20['title_cc']=df20.company_code.str.title()
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>title_cc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>Abcd</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>Efgf</td>
    </tr>
    <tr>
      <td>2</td>
      <td>zefsalf</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>Zefsalf</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>Sdfslew</td>
    </tr>
    <tr>
      <td>4</td>
      <td>zekfsdf</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>Zekfsdf</td>
    </tr>
  </tbody>
</table>
</div>



21. Write a Pandas program to replace arbitrary values with other values in a given DataFrame. 


```python
df21 = pd.DataFrame({
    'company_code': ['A','B', 'C', 'D', 'A'],
    'date_of_sale': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]
})
```


```python
#replacing A with C
df21.replace('A','C')
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>C</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>B</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>C</td>
      <td>25/09/1998</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>D</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>C</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>



22. Write a Pandas program to replace more than one value with other values in a given DataFrame. 


```python
df22 = pd.DataFrame({
    'company_code': ['A','B', 'C', 'D', 'A'],
    'date_of_sale': ['12/05/2002','16/02/1999','25/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]
})
```


```python
df22.replace(['A','C'],['X','Y'])
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>X</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>B</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Y</td>
      <td>25/09/1998</td>
      <td>22.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>D</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>X</td>
      <td>15/09/1997</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>



23. Write a Pandas program to split a string of a column of a given DataFrame into multiple columns.


```python
df23 = pd.DataFrame({
    'name': ['Alberto  Franco','Gino Ann Mcneill','Ryan  Parkes', 'Eesha Artur Hinton', 'Syed  Wharton'],
    'date_of_birth ': ['17/05/2002','16/02/1999','25/09/1998','11/05/2002','15/09/1997'],
    'age': [18.5, 21.2, 22.5, 22, 23]
})
```


```python
df23['First Middle Last'.split()]=df23.name.str.split(expand=True)
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
      <th>name</th>
      <th>date_of_birth</th>
      <th>age</th>
      <th>First</th>
      <th>Middle</th>
      <th>Last</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Alberto  Franco</td>
      <td>17/05/2002</td>
      <td>18.5</td>
      <td>Alberto</td>
      <td>Franco</td>
      <td>None</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Ann Mcneill</td>
      <td>16/02/1999</td>
      <td>21.2</td>
      <td>Gino</td>
      <td>Ann</td>
      <td>Mcneill</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Ryan  Parkes</td>
      <td>25/09/1998</td>
      <td>22.5</td>
      <td>Ryan</td>
      <td>Parkes</td>
      <td>None</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Artur Hinton</td>
      <td>11/05/2002</td>
      <td>22.0</td>
      <td>Eesha</td>
      <td>Artur</td>
      <td>Hinton</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Syed  Wharton</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>Syed</td>
      <td>Wharton</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



24. Write a Pandas program to extract email from a specified column of string type of a given DataFrame


```python
df24 = pd.DataFrame({
    'name_email': ['Alberto Franco af@gmail.com','Gino Mcneill gm@yahoo.com','Ryan Parkes rp@abc.io', 'Eesha Hinton', 'Gino Mcneill gm@github.com']
    })
```


```python
df24['email']=df24.name_email.apply(lambda x:re.findall('\S+@+\S+',str(x)))
df24
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
      <th>name_email</th>
      <th>email</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Alberto Franco af@gmail.com</td>
      <td>[af@gmail.com]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Mcneill gm@yahoo.com</td>
      <td>[gm@yahoo.com]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Ryan Parkes rp@abc.io</td>
      <td>[rp@abc.io]</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gino Mcneill gm@github.com</td>
      <td>[gm@github.com]</td>
    </tr>
  </tbody>
</table>
</div>




```python

df24['email2']=df24.name_email.apply(lambda x:','.join(re.findall(r'[\w\.-]+@[\w\.-]+',str(x))))
df24
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
      <th>name_email</th>
      <th>email</th>
      <th>email2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Alberto Franco af@gmail.com</td>
      <td>[af@gmail.com]</td>
      <td>af@gmail.com</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gino Mcneill gm@yahoo.com</td>
      <td>[gm@yahoo.com]</td>
      <td>gm@yahoo.com</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Ryan Parkes rp@abc.io</td>
      <td>[rp@abc.io]</td>
      <td>rp@abc.io</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Eesha Hinton</td>
      <td>[]</td>
      <td></td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gino Mcneill gm@github.com</td>
      <td>[gm@github.com]</td>
      <td>gm@github.com</td>
    </tr>
  </tbody>
</table>
</div>



25. Write a Pandas program to extract hash attached word from twitter text from the specified column of a given DataFrame. 


```python
df25 = pd.DataFrame({
    'tweets': ['#Obama says goodbye','Retweets for #cash','A political endorsement in #Indonesia', '1 dog = many #retweets', 'Just a simple #egg']
    })
```


```python
df25['hashtags']=df25.tweets.apply(lambda x: ' '.join(re.findall(r'(?<=#)\w+',x)))
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
      <th>tweets</th>
      <th>hashtags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>#Obama says goodbye</td>
      <td>Obama</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Retweets for #cash</td>
      <td>cash</td>
    </tr>
    <tr>
      <td>2</td>
      <td>A political endorsement in #Indonesia</td>
      <td>Indonesia</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1 dog = many #retweets</td>
      <td>retweets</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Just a simple #egg</td>
      <td>egg</td>
    </tr>
  </tbody>
</table>
</div>



26. Write a Pandas program to extract word mention someone in tweets using @ from the specified column of a given DataFrame. 


```python
df26 = pd.DataFrame({
    'tweets': ['@Obama says goodbye','Retweets for @cash','A political endorsement in @Indonesia', '1 dog = many #retweets', 'Just a simple #egg']
    })
```


```python
df26['at_words']=df26.tweets.apply(lambda x:' '.join(re.findall(r'(?<=@)\w+',x)))
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
      <th>tweets</th>
      <th>at_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>@Obama says goodbye</td>
      <td>Obama</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Retweets for @cash</td>
      <td>cash</td>
    </tr>
    <tr>
      <td>2</td>
      <td>A political endorsement in @Indonesia</td>
      <td>Indonesia</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1 dog = many #retweets</td>
      <td></td>
    </tr>
    <tr>
      <td>4</td>
      <td>Just a simple #egg</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



27. Write a Pandas program to extract only number from the specified column of a given DataFrame. 


```python
df27 = pd.DataFrame({
    'company_code': ['c0001','c0002','c0003', 'c0003', 'c0004'],
    'address': ['7277 Surrey Ave.','920 N. Bishop Ave.','9910 Golden Star St.', '25 Dunbar St.', '17 West Livingston Court']
    })
```


```python
df27['numeric_in_address']=df27.address.apply(lambda x:','.join(re.findall(r'[0-9]+',x)))
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
      <th>company_code</th>
      <th>address</th>
      <th>numeric_in_address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c0001</td>
      <td>7277 Surrey Ave.</td>
      <td>7277</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c0002</td>
      <td>920 N. Bishop Ave.</td>
      <td>920</td>
    </tr>
    <tr>
      <td>2</td>
      <td>c0003</td>
      <td>9910 Golden Star St.</td>
      <td>9910</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c0003</td>
      <td>25 Dunbar St.</td>
      <td>25</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c0004</td>
      <td>17 West Livingston Court</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



28. Write a Pandas program to extract only phone number from the specified column of a given DataFrame.


```python
df28 = pd.DataFrame({
    'company_code': ['c0001','c0002','c0003', 'c0003', 'c0004'],
    'company_phone_no': ['Company1-Phone no. 4695168357','Company2-Phone no. 8088729013','Company3-Phone no. 6204658086', 'Company4-Phone no. 5159530096', 'Company5-Phone no. 9037952371']
    })
```


```python
re.findall(r'\b\d{10}\b',df28.company_phone_no[1])
```




    ['8088729013']




```python
df28['phone_number']=df28.company_phone_no.apply(lambda x:','.join(re.findall(r'\b\d{10}\b',x)))
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
      <th>company_code</th>
      <th>company_phone_no</th>
      <th>phone_number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c0001</td>
      <td>Company1-Phone no. 4695168357</td>
      <td>4695168357</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c0002</td>
      <td>Company2-Phone no. 8088729013</td>
      <td>8088729013</td>
    </tr>
    <tr>
      <td>2</td>
      <td>c0003</td>
      <td>Company3-Phone no. 6204658086</td>
      <td>6204658086</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c0003</td>
      <td>Company4-Phone no. 5159530096</td>
      <td>5159530096</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c0004</td>
      <td>Company5-Phone no. 9037952371</td>
      <td>9037952371</td>
    </tr>
  </tbody>
</table>
</div>



29. Write a Pandas program to extract year between 1800 to 2200 from the specified column of a given DataFrame.


```python
df29 = pd.DataFrame({
    'company_code': ['c0001','c0002','c0003', 'c0003', 'c0004'],
    'year': ['year 1800','year 1700','year 2300', 'year 1900', 'year 2200']
    })
```


```python
df29.year.str.split(expand=True)
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
      <td>year</td>
      <td>1800</td>
    </tr>
    <tr>
      <td>1</td>
      <td>year</td>
      <td>1700</td>
    </tr>
    <tr>
      <td>2</td>
      <td>year</td>
      <td>2300</td>
    </tr>
    <tr>
      <td>3</td>
      <td>year</td>
      <td>1900</td>
    </tr>
    <tr>
      <td>4</td>
      <td>year</td>
      <td>2200</td>
    </tr>
  </tbody>
</table>
</div>




```python
re.findall(r'\b([1][8-9]+\d{02}|[2][1]\d{2}|2200)\b','year 1800 year 1700 year 2300 year 1900 year 2200 year 1711')
```




    ['1800', '1900', '2200']




```python
df29['year_btw']=df29.year.apply(lambda x:','.join(re.findall(r'\b([1][8-9]+\d{02}|[2][1]\d{2}|2200)\b',x)))
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
      <th>company_code</th>
      <th>year</th>
      <th>year_btw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c0001</td>
      <td>year 1800</td>
      <td>1800</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c0002</td>
      <td>year 1700</td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>c0003</td>
      <td>year 2300</td>
      <td></td>
    </tr>
    <tr>
      <td>3</td>
      <td>c0003</td>
      <td>year 1900</td>
      <td>1900</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c0004</td>
      <td>year 2200</td>
      <td>2200</td>
    </tr>
  </tbody>
</table>
</div>



30. Write a Pandas program to extract only non alphanumeric characters from the specified column of a given DataFrame.


```python
df30 = pd.DataFrame({
    'company_code': ['c0001#','c00@0^2','$c0003', 'c0003', '&c0004'],
    'year': ['year 1800','year 1700','year 2300', 'year 1900', 'year 2200']
    })
```


```python
re.findall(r'[^A-Za-z0-9]','c00@0^2')
```




    ['@', '^']




```python
df30['special_char']=df30.company_code.apply(lambda x:','.join(re.findall(r'[^a-z0-9]',x)))
df30
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
      <th>company_code</th>
      <th>year</th>
      <th>special_char</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c0001#</td>
      <td>year 1800</td>
      <td>#</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c00@0^2</td>
      <td>year 1700</td>
      <td>@,^</td>
    </tr>
    <tr>
      <td>2</td>
      <td>$c0003</td>
      <td>year 2300</td>
      <td>$</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c0003</td>
      <td>year 1900</td>
      <td></td>
    </tr>
    <tr>
      <td>4</td>
      <td>&amp;c0004</td>
      <td>year 2200</td>
      <td>&amp;</td>
    </tr>
  </tbody>
</table>
</div>



31. Write a Pandas program to extract only punctuations from the specified column of a given DataFrame.


```python
df31 = pd.DataFrame({
    'company_code': ['c0001.','c000,2','c0003', 'c0003#', 'c0004,'],
    'year': ['year 1800','year 1700','year 2300', 'year 1900', 'year 2200']
    })
```


```python
df31['puct']=df31.company_code.apply(lambda x:re.findall(r'[^\w\s]',x))
df31
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
      <th>company_code</th>
      <th>year</th>
      <th>puct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c0001.</td>
      <td>year 1800</td>
      <td>[.]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c000,2</td>
      <td>year 1700</td>
      <td>[,]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>c0003</td>
      <td>year 2300</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c0003#</td>
      <td>year 1900</td>
      <td>[#]</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c0004,</td>
      <td>year 2200</td>
      <td>[,]</td>
    </tr>
  </tbody>
</table>
</div>



32. Write a Pandas program to remove repetitive characters from the specified column of a given DataFrame.


```python
df32 = pd.DataFrame({
    'text_code': ['t0001.','t0002','t0003', 't0004'],
    'text_lang': ['She livedd a long life.', 'How oold is your father?', 'What is tthe problem?','TThhis desk is used by Tom.']
    })
```


```python

```


```python
def rep_char(str1):
    tchr = str1.group(0)
    if len(tchr) > 1:
        return tchr[0:1] # can change the value here on repetition
def unique_char(rep, sent_text):
    convert = re.sub(r'(\w)\1+', rep, sent_text) 
    return convert
df32['normal_text']=df32['text_lang'].apply(lambda x : unique_char(rep_char,x))
df32
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
      <th>text_code</th>
      <th>text_lang</th>
      <th>normal_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>t0001.</td>
      <td>She livedd a long life.</td>
      <td>She lived a long life.</td>
    </tr>
    <tr>
      <td>1</td>
      <td>t0002</td>
      <td>How oold is your father?</td>
      <td>How old is your father?</td>
    </tr>
    <tr>
      <td>2</td>
      <td>t0003</td>
      <td>What is tthe problem?</td>
      <td>What is the problem?</td>
    </tr>
    <tr>
      <td>3</td>
      <td>t0004</td>
      <td>TThhis desk is used by Tom.</td>
      <td>This desk is used by Tom.</td>
    </tr>
  </tbody>
</table>
</div>



33. Write a Pandas program to extract numbers greater than 940 from the specified column of a given DataFrame. 


```python
df33 = pd.DataFrame({
    'company_code': ['c0001','c0002','c0003', 'c0003', 'c0004'],
    'address': ['7277 Surrey Ave.1111','920 N. Bishop Ave.','9910 Golden Star St.', '1025 Dunbar St.', '1700 West Livingston Court']
    })
df33
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
      <th>company_code</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c0001</td>
      <td>7277 Surrey Ave.1111</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c0002</td>
      <td>920 N. Bishop Ave.</td>
    </tr>
    <tr>
      <td>2</td>
      <td>c0003</td>
      <td>9910 Golden Star St.</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c0003</td>
      <td>1025 Dunbar St.</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c0004</td>
      <td>1700 West Livingston Court</td>
    </tr>
  </tbody>
</table>
</div>




```python
df33['num_great']=df33.address.apply(lambda x:','.join(re.findall(r'([9][4-9]\d|[1-9]\d{3})',x)))
df33
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
      <th>company_code</th>
      <th>address</th>
      <th>num_great</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c0001</td>
      <td>7277 Surrey Ave.1111</td>
      <td>7277,1111</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c0002</td>
      <td>920 N. Bishop Ave.</td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>c0003</td>
      <td>9910 Golden Star St.</td>
      <td>991</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c0003</td>
      <td>1025 Dunbar St.</td>
      <td>1025</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c0004</td>
      <td>1700 West Livingston Court</td>
      <td>1700</td>
    </tr>
  </tbody>
</table>
</div>



34. Write a Pandas program to extract numbers less than 100 from the specified column of a given DataFrame.


```python
df34 = pd.DataFrame({
    'company_code': ['c0001','c0002','c0003', 'c0003', 'c0004'],
    'address': ['72 Surrey Ave.11','92 N. Bishop Ave.','9910 Golden Star St.', '102 Dunbar St.', '17 West Livingston Court']
    })
```


```python
df34['num_less']=df34.address.apply(lambda x:re.findall(r'(\b\d{2}\b)',x))
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
      <th>company_code</th>
      <th>address</th>
      <th>num_less</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c0001</td>
      <td>72 Surrey Ave.11</td>
      <td>[72, 11]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c0002</td>
      <td>92 N. Bishop Ave.</td>
      <td>[92]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>c0003</td>
      <td>9910 Golden Star St.</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c0003</td>
      <td>102 Dunbar St.</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c0004</td>
      <td>17 West Livingston Court</td>
      <td>[17]</td>
    </tr>
  </tbody>
</table>
</div>



35. Write a Pandas program to check whether two given words present in a specified column of a given DataFrame. 


```python
df35 = pd.DataFrame({
    'company_code': ['c0001','c0002','c0003', 'c0003', 'c0004'],
    'address': ['9910 Surrey Ave.','92 N. Bishop Ave.','9910 Golden Star Ave.', '102 Dunbar St.', '17 West Livingston Court']
    })
```


```python
df35['check_words']=df35.address.apply(lambda x:re.findall(r'(?=.*Ave.)(?=.*9910).*',x))
df35
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
      <th>company_code</th>
      <th>address</th>
      <th>check_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>c0001</td>
      <td>9910 Surrey Ave.</td>
      <td>[9910 Surrey Ave.]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>c0002</td>
      <td>92 N. Bishop Ave.</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>c0003</td>
      <td>9910 Golden Star Ave.</td>
      <td>[9910 Golden Star Ave.]</td>
    </tr>
    <tr>
      <td>3</td>
      <td>c0003</td>
      <td>102 Dunbar St.</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c0004</td>
      <td>17 West Livingston Court</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
</div>



36. Write a Pandas program to extract valid date (format: mm-dd-yyyy) from a given column of a given DataFrame. 


```python
df36 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'zefsalf', 'sdfslew', 'zekfsdf'],
    'date_of_sale': ['12/05/2002','16/02/1999','05/09/1998','12/02/2022','15/09/1997'],
    'sale_amount': [12348.5, 233331.2, 22.5, 2566552.0, 23.0]
})
```


```python
df36['valid_dates']=df36.date_of_sale.apply(lambda x:re.findall(r'(1[0-2]|0[1-9])/(3[0-1]|[1-2][0-9]|0\d)/(\d{4})',x))
df36
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>sale_amount</th>
      <th>valid_dates</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>12348.5</td>
      <td>[(12, 05, 2002)]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>233331.2</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>zefsalf</td>
      <td>05/09/1998</td>
      <td>22.5</td>
      <td>[(05, 09, 1998)]</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>2566552.0</td>
      <td>[(12, 02, 2022)]</td>
    </tr>
    <tr>
      <td>4</td>
      <td>zekfsdf</td>
      <td>15/09/1997</td>
      <td>23.0</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
</div>



37. Write a Pandas program to extract only words from a given column of a given DataFrame. 


```python
df37 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'zefsalf', 'sdfslew', 'zekfsdf'],
    'date_of_sale': ['12/05/2002','16/02/1999','05/09/1998','12/02/2022','15/09/1997'],
    'address': ['9910 Surrey Ave.','92 N. Bishop Ave.','9910 Golden Star Ave.', '102 Dunbar St.', '17 West Livingston Court']
})
```


```python
re.findall(r'\S+[a-zA-Z]+\S+','92 N. Bishop Ave.')
```




    ['Bishop', 'Ave.']




```python
df37['only_words']=df37.address.apply(lambda x:' '.join(re.findall(r'\b[^\d\W]+\b',x)))
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>address</th>
      <th>only_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>9910 Surrey Ave.</td>
      <td>Surrey Ave</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>92 N. Bishop Ave.</td>
      <td>N Bishop Ave</td>
    </tr>
    <tr>
      <td>2</td>
      <td>zefsalf</td>
      <td>05/09/1998</td>
      <td>9910 Golden Star Ave.</td>
      <td>Golden Star Ave</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>102 Dunbar St.</td>
      <td>Dunbar St</td>
    </tr>
    <tr>
      <td>4</td>
      <td>zekfsdf</td>
      <td>15/09/1997</td>
      <td>17 West Livingston Court</td>
      <td>West Livingston Court</td>
    </tr>
  </tbody>
</table>
</div>



38. Write a Pandas program to extract the sentences where a specific word is present in a given column of a given DataFrame.


```python
df38 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'zefsalf', 'sdfslew', 'zekfsdf'],
    'date_of_sale': ['12/05/2002','16/02/1999','05/09/1998','12/02/2022','15/09/1997'],
    'address': ['9910 Surrey Avenue','92 N. Bishop Avenue','9910 Golden Star Avenue', '102 Dunbar St.', '17 West Livingston Court']
})
```


```python
word='Avenue'
df38['Ext_Sent']=df38.address.apply(lambda x:re.findall(r'([^.]*'+word+'[^.]*)',x))
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>address</th>
      <th>Ext_Sent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>9910 Surrey Avenue</td>
      <td>[9910 Surrey Avenue]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>92 N. Bishop Avenue</td>
      <td>[ Bishop Avenue]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>zefsalf</td>
      <td>05/09/1998</td>
      <td>9910 Golden Star Avenue</td>
      <td>[9910 Golden Star Avenue]</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>102 Dunbar St.</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>4</td>
      <td>zekfsdf</td>
      <td>15/09/1997</td>
      <td>17 West Livingston Court</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
</div>



39. Write a Pandas program to extract the unique sentences from a given column of a given DataFrame.


```python
df39 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'zefsalf', 'sdfslew', 'zekfsdf'],
    'date_of_sale': ['12/05/2002','16/02/1999','05/09/1998','12/02/2022','15/09/1997'],
    'address': ['9910 Surrey Avenue\n9910 Surrey Avenue','92 N. Bishop Avenue','9910 Golden Star Avenue', '102 Dunbar St.\n102 Dunbar St.', '17 West Livingston Court']
})
df39
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>9910 Surrey Avenue\n9910 Surrey Avenue</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>92 N. Bishop Avenue</td>
    </tr>
    <tr>
      <td>2</td>
      <td>zefsalf</td>
      <td>05/09/1998</td>
      <td>9910 Golden Star Avenue</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>102 Dunbar St.\n102 Dunbar St.</td>
    </tr>
    <tr>
      <td>4</td>
      <td>zekfsdf</td>
      <td>15/09/1997</td>
      <td>17 West Livingston Court</td>
    </tr>
  </tbody>
</table>
</div>




```python
df39['unique_Statements']=df39.address.apply(lambda x:re.findall(r'(?sm)(^[^\r\n]+$)(?!.*^\1$)', x))
df39
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>address</th>
      <th>unique_Statements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>9910 Surrey Avenue\n9910 Surrey Avenue</td>
      <td>[9910 Surrey Avenue]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>92 N. Bishop Avenue</td>
      <td>[92 N. Bishop Avenue]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>zefsalf</td>
      <td>05/09/1998</td>
      <td>9910 Golden Star Avenue</td>
      <td>[9910 Golden Star Avenue]</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>102 Dunbar St.\n102 Dunbar St.</td>
      <td>[102 Dunbar St.]</td>
    </tr>
    <tr>
      <td>4</td>
      <td>zekfsdf</td>
      <td>15/09/1997</td>
      <td>17 West Livingston Court</td>
      <td>[17 West Livingston Court]</td>
    </tr>
  </tbody>
</table>
</div>




```python
re.findall(r'(?sm)(^[^\r\n]+$)(?!.*^\1$)', '9910 Surrey Avenue\n9910 Surrey Avenue')
```




    ['9910 Surrey Avenue']




```python

```

40. Write a Pandas program to extract words starting with capital words from a given column of a given DataFrame


```python
df40 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'zefsalf', 'sdfslew', 'zekfsdf'],
    'date_of_sale': ['12/05/2002','16/02/1999','05/09/1998','12/02/2022','15/09/1997'],
    'address': ['9910 Surrey Avenue','92 N. Bishop Avenue','9910 Golden Star Avenue', '102 Dunbar St.', '17 West Livingston Court']
})
```


```python
df40['star_Capital']=df40.address.apply(lambda x:",".join(re.findall(r'\b[A-Z]+\S+',x)))
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>address</th>
      <th>star_Capital</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>9910 Surrey Avenue</td>
      <td>Surrey,Avenue</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>92 N. Bishop Avenue</td>
      <td>N.,Bishop,Avenue</td>
    </tr>
    <tr>
      <td>2</td>
      <td>zefsalf</td>
      <td>05/09/1998</td>
      <td>9910 Golden Star Avenue</td>
      <td>Golden,Star,Avenue</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>102 Dunbar St.</td>
      <td>Dunbar,St.</td>
    </tr>
    <tr>
      <td>4</td>
      <td>zekfsdf</td>
      <td>15/09/1997</td>
      <td>17 West Livingston Court</td>
      <td>West,Livingston,Court</td>
    </tr>
  </tbody>
</table>
</div>



41. Write a Pandas program to remove the html tags within the specified column of a given DataFrame.


```python
df41 = pd.DataFrame({
    'company_code': ['Abcd','EFGF', 'zefsalf', 'sdfslew', 'zekfsdf'],
    'date_of_sale': ['12/05/2002','16/02/1999','05/09/1998','12/02/2022','15/09/1997'],
    'address': ['9910 Surrey <b>Avenue</b>','92 N. Bishop Avenue','9910 <br>Golden Star Avenue', '102 Dunbar <i></i>St.', '17 West Livingston Court']
})
```


```python
re.sub('<.*?>','','9910 Surrey <b>Avenue</b>')
```




    '9910 Surrey Avenue'




```python
df41['removed_html']=df41.address.apply(lambda x:re.sub(r'<.*?>','',x))
df41
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
      <th>company_code</th>
      <th>date_of_sale</th>
      <th>address</th>
      <th>removed_html</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Abcd</td>
      <td>12/05/2002</td>
      <td>9910 Surrey &lt;b&gt;Avenue&lt;/b&gt;</td>
      <td>9910 Surrey Avenue</td>
    </tr>
    <tr>
      <td>1</td>
      <td>EFGF</td>
      <td>16/02/1999</td>
      <td>92 N. Bishop Avenue</td>
      <td>92 N. Bishop Avenue</td>
    </tr>
    <tr>
      <td>2</td>
      <td>zefsalf</td>
      <td>05/09/1998</td>
      <td>9910 &lt;br&gt;Golden Star Avenue</td>
      <td>9910 Golden Star Avenue</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sdfslew</td>
      <td>12/02/2022</td>
      <td>102 Dunbar &lt;i&gt;&lt;/i&gt;St.</td>
      <td>102 Dunbar St.</td>
    </tr>
    <tr>
      <td>4</td>
      <td>zekfsdf</td>
      <td>15/09/1997</td>
      <td>17 West Livingston Court</td>
      <td>17 West Livingston Court</td>
    </tr>
  </tbody>
</table>
</div>


