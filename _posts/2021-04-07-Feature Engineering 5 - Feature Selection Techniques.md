---
title: "Feature Engineering 5 - Feature Selection Techniques"
date: 2021-04-07
tags: [data science, Feature Engineering, messy data]
header:
  image: "/images/Feature_engineering.jpeg"
excerpt: "Data Science, Feature Engineering, Messy Data"
mathjax: "true"
---



# Feature Selection Techniques

- To reduce the dimensions in models
- Overcoming curse of dimensionality


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df=pd.read_csv('Datasets/mobile_dataset.csv')
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
      <th>battery_power</th>
      <th>blue</th>
      <th>clock_speed</th>
      <th>dual_sim</th>
      <th>fc</th>
      <th>four_g</th>
      <th>int_memory</th>
      <th>m_dep</th>
      <th>mobile_wt</th>
      <th>n_cores</th>
      <th>...</th>
      <th>px_height</th>
      <th>px_width</th>
      <th>ram</th>
      <th>sc_h</th>
      <th>sc_w</th>
      <th>talk_time</th>
      <th>three_g</th>
      <th>touch_screen</th>
      <th>wifi</th>
      <th>price_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>842</td>
      <td>0</td>
      <td>2.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>0.6</td>
      <td>188</td>
      <td>2</td>
      <td>...</td>
      <td>20</td>
      <td>756</td>
      <td>2549</td>
      <td>9</td>
      <td>7</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1021</td>
      <td>1</td>
      <td>0.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>53</td>
      <td>0.7</td>
      <td>136</td>
      <td>3</td>
      <td>...</td>
      <td>905</td>
      <td>1988</td>
      <td>2631</td>
      <td>17</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>563</td>
      <td>1</td>
      <td>0.5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>41</td>
      <td>0.9</td>
      <td>145</td>
      <td>5</td>
      <td>...</td>
      <td>1263</td>
      <td>1716</td>
      <td>2603</td>
      <td>11</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>615</td>
      <td>1</td>
      <td>2.5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0.8</td>
      <td>131</td>
      <td>6</td>
      <td>...</td>
      <td>1216</td>
      <td>1786</td>
      <td>2769</td>
      <td>16</td>
      <td>8</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1821</td>
      <td>1</td>
      <td>1.2</td>
      <td>0</td>
      <td>13</td>
      <td>1</td>
      <td>44</td>
      <td>0.6</td>
      <td>141</td>
      <td>2</td>
      <td>...</td>
      <td>1208</td>
      <td>1212</td>
      <td>1411</td>
      <td>8</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df.columns
```




    Index(['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
           'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
           'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
           'touch_screen', 'wifi', 'price_range'],
          dtype='object')



## 1. Univariate Selection

- Data should have numerical value (if we have categorial variable in data we can perform feature engineering and convert the same into numerical one).


```python
# Segregating into dependent and independent features
X=df.drop('price_range',axis=1) #Independent Feature
y=df['price_range'] #Dependent Feature
```


```python
X.head()
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
      <th>battery_power</th>
      <th>blue</th>
      <th>clock_speed</th>
      <th>dual_sim</th>
      <th>fc</th>
      <th>four_g</th>
      <th>int_memory</th>
      <th>m_dep</th>
      <th>mobile_wt</th>
      <th>n_cores</th>
      <th>pc</th>
      <th>px_height</th>
      <th>px_width</th>
      <th>ram</th>
      <th>sc_h</th>
      <th>sc_w</th>
      <th>talk_time</th>
      <th>three_g</th>
      <th>touch_screen</th>
      <th>wifi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>842</td>
      <td>0</td>
      <td>2.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>0.6</td>
      <td>188</td>
      <td>2</td>
      <td>2</td>
      <td>20</td>
      <td>756</td>
      <td>2549</td>
      <td>9</td>
      <td>7</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1021</td>
      <td>1</td>
      <td>0.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>53</td>
      <td>0.7</td>
      <td>136</td>
      <td>3</td>
      <td>6</td>
      <td>905</td>
      <td>1988</td>
      <td>2631</td>
      <td>17</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>563</td>
      <td>1</td>
      <td>0.5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>41</td>
      <td>0.9</td>
      <td>145</td>
      <td>5</td>
      <td>6</td>
      <td>1263</td>
      <td>1716</td>
      <td>2603</td>
      <td>11</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>615</td>
      <td>1</td>
      <td>2.5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0.8</td>
      <td>131</td>
      <td>6</td>
      <td>9</td>
      <td>1216</td>
      <td>1786</td>
      <td>2769</td>
      <td>16</td>
      <td>8</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1821</td>
      <td>1</td>
      <td>1.2</td>
      <td>0</td>
      <td>13</td>
      <td>1</td>
      <td>44</td>
      <td>0.6</td>
      <td>141</td>
      <td>2</td>
      <td>14</td>
      <td>1208</td>
      <td>1212</td>
      <td>1411</td>
      <td>8</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.head()
```




    0    1
    1    2
    2    2
    3    2
    4    1
    Name: price_range, dtype: int64




```python
from sklearn.feature_selection import SelectKBest # To select top k best features
from sklearn.feature_selection import chi2
```


```python
X.shape
```




    (2000, 20)




```python
## Apply SelectKBest Algorithm
ordered_rank_features=SelectKBest(score_func=chi2,k=20)
Ordered_feature=ordered_rank_features.fit(X,y)
```


```python
features_rank=pd.DataFrame(Ordered_feature.scores_,columns=['Score'],index=X.columns).reset_index().rename(columns={'index': 'Feature'})
```


```python
features_rank.nlargest(10,'Score',keep='all')
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
      <th>Feature</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>13</td>
      <td>ram</td>
      <td>931267.519053</td>
    </tr>
    <tr>
      <td>11</td>
      <td>px_height</td>
      <td>17363.569536</td>
    </tr>
    <tr>
      <td>0</td>
      <td>battery_power</td>
      <td>14129.866576</td>
    </tr>
    <tr>
      <td>12</td>
      <td>px_width</td>
      <td>9810.586750</td>
    </tr>
    <tr>
      <td>8</td>
      <td>mobile_wt</td>
      <td>95.972863</td>
    </tr>
    <tr>
      <td>6</td>
      <td>int_memory</td>
      <td>89.839124</td>
    </tr>
    <tr>
      <td>15</td>
      <td>sc_w</td>
      <td>16.480319</td>
    </tr>
    <tr>
      <td>16</td>
      <td>talk_time</td>
      <td>13.236400</td>
    </tr>
    <tr>
      <td>4</td>
      <td>fc</td>
      <td>10.135166</td>
    </tr>
    <tr>
      <td>14</td>
      <td>sc_h</td>
      <td>9.614878</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Feature Importance

This technique gives us a score for each feature of our data, the higher the score, more relevant it is


```python
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(X,y)
```

    /opt/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)





    ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                         max_depth=None, max_features='auto', max_leaf_nodes=None,
                         min_impurity_decrease=0.0, min_impurity_split=None,
                         min_samples_leaf=1, min_samples_split=2,
                         min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                         oob_score=False, random_state=None, verbose=0,
                         warm_start=False)




```python
print(model.feature_importances_)
```

    [0.06141965 0.02410379 0.03427029 0.02055451 0.02979936 0.01864498
     0.03502876 0.03265603 0.03416619 0.03286097 0.03296015 0.04391382
     0.05001435 0.39492892 0.03641193 0.03377228 0.03264326 0.01511883
     0.01718066 0.01955127]



```python
ranked_features=pd.DataFrame(model.feature_importances_,columns=['Score'],index=X.columns).reset_index().rename(columns={'index': 'Feature'})
```


```python
ranked_features
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
      <th>Feature</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>battery_power</td>
      <td>0.061420</td>
    </tr>
    <tr>
      <td>1</td>
      <td>blue</td>
      <td>0.024104</td>
    </tr>
    <tr>
      <td>2</td>
      <td>clock_speed</td>
      <td>0.034270</td>
    </tr>
    <tr>
      <td>3</td>
      <td>dual_sim</td>
      <td>0.020555</td>
    </tr>
    <tr>
      <td>4</td>
      <td>fc</td>
      <td>0.029799</td>
    </tr>
    <tr>
      <td>5</td>
      <td>four_g</td>
      <td>0.018645</td>
    </tr>
    <tr>
      <td>6</td>
      <td>int_memory</td>
      <td>0.035029</td>
    </tr>
    <tr>
      <td>7</td>
      <td>m_dep</td>
      <td>0.032656</td>
    </tr>
    <tr>
      <td>8</td>
      <td>mobile_wt</td>
      <td>0.034166</td>
    </tr>
    <tr>
      <td>9</td>
      <td>n_cores</td>
      <td>0.032861</td>
    </tr>
    <tr>
      <td>10</td>
      <td>pc</td>
      <td>0.032960</td>
    </tr>
    <tr>
      <td>11</td>
      <td>px_height</td>
      <td>0.043914</td>
    </tr>
    <tr>
      <td>12</td>
      <td>px_width</td>
      <td>0.050014</td>
    </tr>
    <tr>
      <td>13</td>
      <td>ram</td>
      <td>0.394929</td>
    </tr>
    <tr>
      <td>14</td>
      <td>sc_h</td>
      <td>0.036412</td>
    </tr>
    <tr>
      <td>15</td>
      <td>sc_w</td>
      <td>0.033772</td>
    </tr>
    <tr>
      <td>16</td>
      <td>talk_time</td>
      <td>0.032643</td>
    </tr>
    <tr>
      <td>17</td>
      <td>three_g</td>
      <td>0.015119</td>
    </tr>
    <tr>
      <td>18</td>
      <td>touch_screen</td>
      <td>0.017181</td>
    </tr>
    <tr>
      <td>19</td>
      <td>wifi</td>
      <td>0.019551</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(x='Score',y='Feature',data=ranked_features.sort_values(['Score'],ascending=False)[:10])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc6380eca50>




<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 5 - Feature Selection Techniques_files/Feature Engineering 5 - Feature Selection Techniques_19_1.png" alt="linearly separable data">


## 3. Correlation


```python
df.corr()
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
      <th>battery_power</th>
      <th>blue</th>
      <th>clock_speed</th>
      <th>dual_sim</th>
      <th>fc</th>
      <th>four_g</th>
      <th>int_memory</th>
      <th>m_dep</th>
      <th>mobile_wt</th>
      <th>n_cores</th>
      <th>...</th>
      <th>px_height</th>
      <th>px_width</th>
      <th>ram</th>
      <th>sc_h</th>
      <th>sc_w</th>
      <th>talk_time</th>
      <th>three_g</th>
      <th>touch_screen</th>
      <th>wifi</th>
      <th>price_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>battery_power</td>
      <td>1.000000</td>
      <td>0.011252</td>
      <td>0.011482</td>
      <td>-0.041847</td>
      <td>0.033334</td>
      <td>0.015665</td>
      <td>-0.004004</td>
      <td>0.034085</td>
      <td>0.001844</td>
      <td>-0.029727</td>
      <td>...</td>
      <td>0.014901</td>
      <td>-0.008402</td>
      <td>-0.000653</td>
      <td>-0.029959</td>
      <td>-0.021421</td>
      <td>0.052510</td>
      <td>0.011522</td>
      <td>-0.010516</td>
      <td>-0.008343</td>
      <td>0.200723</td>
    </tr>
    <tr>
      <td>blue</td>
      <td>0.011252</td>
      <td>1.000000</td>
      <td>0.021419</td>
      <td>0.035198</td>
      <td>0.003593</td>
      <td>0.013443</td>
      <td>0.041177</td>
      <td>0.004049</td>
      <td>-0.008605</td>
      <td>0.036161</td>
      <td>...</td>
      <td>-0.006872</td>
      <td>-0.041533</td>
      <td>0.026351</td>
      <td>-0.002952</td>
      <td>0.000613</td>
      <td>0.013934</td>
      <td>-0.030236</td>
      <td>0.010061</td>
      <td>-0.021863</td>
      <td>0.020573</td>
    </tr>
    <tr>
      <td>clock_speed</td>
      <td>0.011482</td>
      <td>0.021419</td>
      <td>1.000000</td>
      <td>-0.001315</td>
      <td>-0.000434</td>
      <td>-0.043073</td>
      <td>0.006545</td>
      <td>-0.014364</td>
      <td>0.012350</td>
      <td>-0.005724</td>
      <td>...</td>
      <td>-0.014523</td>
      <td>-0.009476</td>
      <td>0.003443</td>
      <td>-0.029078</td>
      <td>-0.007378</td>
      <td>-0.011432</td>
      <td>-0.046433</td>
      <td>0.019756</td>
      <td>-0.024471</td>
      <td>-0.006606</td>
    </tr>
    <tr>
      <td>dual_sim</td>
      <td>-0.041847</td>
      <td>0.035198</td>
      <td>-0.001315</td>
      <td>1.000000</td>
      <td>-0.029123</td>
      <td>0.003187</td>
      <td>-0.015679</td>
      <td>-0.022142</td>
      <td>-0.008979</td>
      <td>-0.024658</td>
      <td>...</td>
      <td>-0.020875</td>
      <td>0.014291</td>
      <td>0.041072</td>
      <td>-0.011949</td>
      <td>-0.016666</td>
      <td>-0.039404</td>
      <td>-0.014008</td>
      <td>-0.017117</td>
      <td>0.022740</td>
      <td>0.017444</td>
    </tr>
    <tr>
      <td>fc</td>
      <td>0.033334</td>
      <td>0.003593</td>
      <td>-0.000434</td>
      <td>-0.029123</td>
      <td>1.000000</td>
      <td>-0.016560</td>
      <td>-0.029133</td>
      <td>-0.001791</td>
      <td>0.023618</td>
      <td>-0.013356</td>
      <td>...</td>
      <td>-0.009990</td>
      <td>-0.005176</td>
      <td>0.015099</td>
      <td>-0.011014</td>
      <td>-0.012373</td>
      <td>-0.006829</td>
      <td>0.001793</td>
      <td>-0.014828</td>
      <td>0.020085</td>
      <td>0.021998</td>
    </tr>
    <tr>
      <td>four_g</td>
      <td>0.015665</td>
      <td>0.013443</td>
      <td>-0.043073</td>
      <td>0.003187</td>
      <td>-0.016560</td>
      <td>1.000000</td>
      <td>0.008690</td>
      <td>-0.001823</td>
      <td>-0.016537</td>
      <td>-0.029706</td>
      <td>...</td>
      <td>-0.019236</td>
      <td>0.007448</td>
      <td>0.007313</td>
      <td>0.027166</td>
      <td>0.037005</td>
      <td>-0.046628</td>
      <td>0.584246</td>
      <td>0.016758</td>
      <td>-0.017620</td>
      <td>0.014772</td>
    </tr>
    <tr>
      <td>int_memory</td>
      <td>-0.004004</td>
      <td>0.041177</td>
      <td>0.006545</td>
      <td>-0.015679</td>
      <td>-0.029133</td>
      <td>0.008690</td>
      <td>1.000000</td>
      <td>0.006886</td>
      <td>-0.034214</td>
      <td>-0.028310</td>
      <td>...</td>
      <td>0.010441</td>
      <td>-0.008335</td>
      <td>0.032813</td>
      <td>0.037771</td>
      <td>0.011731</td>
      <td>-0.002790</td>
      <td>-0.009366</td>
      <td>-0.026999</td>
      <td>0.006993</td>
      <td>0.044435</td>
    </tr>
    <tr>
      <td>m_dep</td>
      <td>0.034085</td>
      <td>0.004049</td>
      <td>-0.014364</td>
      <td>-0.022142</td>
      <td>-0.001791</td>
      <td>-0.001823</td>
      <td>0.006886</td>
      <td>1.000000</td>
      <td>0.021756</td>
      <td>-0.003504</td>
      <td>...</td>
      <td>0.025263</td>
      <td>0.023566</td>
      <td>-0.009434</td>
      <td>-0.025348</td>
      <td>-0.018388</td>
      <td>0.017003</td>
      <td>-0.012065</td>
      <td>-0.002638</td>
      <td>-0.028353</td>
      <td>0.000853</td>
    </tr>
    <tr>
      <td>mobile_wt</td>
      <td>0.001844</td>
      <td>-0.008605</td>
      <td>0.012350</td>
      <td>-0.008979</td>
      <td>0.023618</td>
      <td>-0.016537</td>
      <td>-0.034214</td>
      <td>0.021756</td>
      <td>1.000000</td>
      <td>-0.018989</td>
      <td>...</td>
      <td>0.000939</td>
      <td>0.000090</td>
      <td>-0.002581</td>
      <td>-0.033855</td>
      <td>-0.020761</td>
      <td>0.006209</td>
      <td>0.001551</td>
      <td>-0.014368</td>
      <td>-0.000409</td>
      <td>-0.030302</td>
    </tr>
    <tr>
      <td>n_cores</td>
      <td>-0.029727</td>
      <td>0.036161</td>
      <td>-0.005724</td>
      <td>-0.024658</td>
      <td>-0.013356</td>
      <td>-0.029706</td>
      <td>-0.028310</td>
      <td>-0.003504</td>
      <td>-0.018989</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.006872</td>
      <td>0.024480</td>
      <td>0.004868</td>
      <td>-0.000315</td>
      <td>0.025826</td>
      <td>0.013148</td>
      <td>-0.014733</td>
      <td>0.023774</td>
      <td>-0.009964</td>
      <td>0.004399</td>
    </tr>
    <tr>
      <td>pc</td>
      <td>0.031441</td>
      <td>-0.009952</td>
      <td>-0.005245</td>
      <td>-0.017143</td>
      <td>0.644595</td>
      <td>-0.005598</td>
      <td>-0.033273</td>
      <td>0.026282</td>
      <td>0.018844</td>
      <td>-0.001193</td>
      <td>...</td>
      <td>-0.018465</td>
      <td>0.004196</td>
      <td>0.028984</td>
      <td>0.004938</td>
      <td>-0.023819</td>
      <td>0.014657</td>
      <td>-0.001322</td>
      <td>-0.008742</td>
      <td>0.005389</td>
      <td>0.033599</td>
    </tr>
    <tr>
      <td>px_height</td>
      <td>0.014901</td>
      <td>-0.006872</td>
      <td>-0.014523</td>
      <td>-0.020875</td>
      <td>-0.009990</td>
      <td>-0.019236</td>
      <td>0.010441</td>
      <td>0.025263</td>
      <td>0.000939</td>
      <td>-0.006872</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.510664</td>
      <td>-0.020352</td>
      <td>0.059615</td>
      <td>0.043038</td>
      <td>-0.010645</td>
      <td>-0.031174</td>
      <td>0.021891</td>
      <td>0.051824</td>
      <td>0.148858</td>
    </tr>
    <tr>
      <td>px_width</td>
      <td>-0.008402</td>
      <td>-0.041533</td>
      <td>-0.009476</td>
      <td>0.014291</td>
      <td>-0.005176</td>
      <td>0.007448</td>
      <td>-0.008335</td>
      <td>0.023566</td>
      <td>0.000090</td>
      <td>0.024480</td>
      <td>...</td>
      <td>0.510664</td>
      <td>1.000000</td>
      <td>0.004105</td>
      <td>0.021599</td>
      <td>0.034699</td>
      <td>0.006720</td>
      <td>0.000350</td>
      <td>-0.001628</td>
      <td>0.030319</td>
      <td>0.165818</td>
    </tr>
    <tr>
      <td>ram</td>
      <td>-0.000653</td>
      <td>0.026351</td>
      <td>0.003443</td>
      <td>0.041072</td>
      <td>0.015099</td>
      <td>0.007313</td>
      <td>0.032813</td>
      <td>-0.009434</td>
      <td>-0.002581</td>
      <td>0.004868</td>
      <td>...</td>
      <td>-0.020352</td>
      <td>0.004105</td>
      <td>1.000000</td>
      <td>0.015996</td>
      <td>0.035576</td>
      <td>0.010820</td>
      <td>0.015795</td>
      <td>-0.030455</td>
      <td>0.022669</td>
      <td>0.917046</td>
    </tr>
    <tr>
      <td>sc_h</td>
      <td>-0.029959</td>
      <td>-0.002952</td>
      <td>-0.029078</td>
      <td>-0.011949</td>
      <td>-0.011014</td>
      <td>0.027166</td>
      <td>0.037771</td>
      <td>-0.025348</td>
      <td>-0.033855</td>
      <td>-0.000315</td>
      <td>...</td>
      <td>0.059615</td>
      <td>0.021599</td>
      <td>0.015996</td>
      <td>1.000000</td>
      <td>0.506144</td>
      <td>-0.017335</td>
      <td>0.012033</td>
      <td>-0.020023</td>
      <td>0.025929</td>
      <td>0.022986</td>
    </tr>
    <tr>
      <td>sc_w</td>
      <td>-0.021421</td>
      <td>0.000613</td>
      <td>-0.007378</td>
      <td>-0.016666</td>
      <td>-0.012373</td>
      <td>0.037005</td>
      <td>0.011731</td>
      <td>-0.018388</td>
      <td>-0.020761</td>
      <td>0.025826</td>
      <td>...</td>
      <td>0.043038</td>
      <td>0.034699</td>
      <td>0.035576</td>
      <td>0.506144</td>
      <td>1.000000</td>
      <td>-0.022821</td>
      <td>0.030941</td>
      <td>0.012720</td>
      <td>0.035423</td>
      <td>0.038711</td>
    </tr>
    <tr>
      <td>talk_time</td>
      <td>0.052510</td>
      <td>0.013934</td>
      <td>-0.011432</td>
      <td>-0.039404</td>
      <td>-0.006829</td>
      <td>-0.046628</td>
      <td>-0.002790</td>
      <td>0.017003</td>
      <td>0.006209</td>
      <td>0.013148</td>
      <td>...</td>
      <td>-0.010645</td>
      <td>0.006720</td>
      <td>0.010820</td>
      <td>-0.017335</td>
      <td>-0.022821</td>
      <td>1.000000</td>
      <td>-0.042688</td>
      <td>0.017196</td>
      <td>-0.029504</td>
      <td>0.021859</td>
    </tr>
    <tr>
      <td>three_g</td>
      <td>0.011522</td>
      <td>-0.030236</td>
      <td>-0.046433</td>
      <td>-0.014008</td>
      <td>0.001793</td>
      <td>0.584246</td>
      <td>-0.009366</td>
      <td>-0.012065</td>
      <td>0.001551</td>
      <td>-0.014733</td>
      <td>...</td>
      <td>-0.031174</td>
      <td>0.000350</td>
      <td>0.015795</td>
      <td>0.012033</td>
      <td>0.030941</td>
      <td>-0.042688</td>
      <td>1.000000</td>
      <td>0.013917</td>
      <td>0.004316</td>
      <td>0.023611</td>
    </tr>
    <tr>
      <td>touch_screen</td>
      <td>-0.010516</td>
      <td>0.010061</td>
      <td>0.019756</td>
      <td>-0.017117</td>
      <td>-0.014828</td>
      <td>0.016758</td>
      <td>-0.026999</td>
      <td>-0.002638</td>
      <td>-0.014368</td>
      <td>0.023774</td>
      <td>...</td>
      <td>0.021891</td>
      <td>-0.001628</td>
      <td>-0.030455</td>
      <td>-0.020023</td>
      <td>0.012720</td>
      <td>0.017196</td>
      <td>0.013917</td>
      <td>1.000000</td>
      <td>0.011917</td>
      <td>-0.030411</td>
    </tr>
    <tr>
      <td>wifi</td>
      <td>-0.008343</td>
      <td>-0.021863</td>
      <td>-0.024471</td>
      <td>0.022740</td>
      <td>0.020085</td>
      <td>-0.017620</td>
      <td>0.006993</td>
      <td>-0.028353</td>
      <td>-0.000409</td>
      <td>-0.009964</td>
      <td>...</td>
      <td>0.051824</td>
      <td>0.030319</td>
      <td>0.022669</td>
      <td>0.025929</td>
      <td>0.035423</td>
      <td>-0.029504</td>
      <td>0.004316</td>
      <td>0.011917</td>
      <td>1.000000</td>
      <td>0.018785</td>
    </tr>
    <tr>
      <td>price_range</td>
      <td>0.200723</td>
      <td>0.020573</td>
      <td>-0.006606</td>
      <td>0.017444</td>
      <td>0.021998</td>
      <td>0.014772</td>
      <td>0.044435</td>
      <td>0.000853</td>
      <td>-0.030302</td>
      <td>0.004399</td>
      <td>...</td>
      <td>0.148858</td>
      <td>0.165818</td>
      <td>0.917046</td>
      <td>0.022986</td>
      <td>0.038711</td>
      <td>0.021859</td>
      <td>0.023611</td>
      <td>-0.030411</td>
      <td>0.018785</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>21 rows × 21 columns</p>
</div>




```python
corr=df.corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc6385a9f50>



<img src="{{ site.url }}{{ site.baseurl }}/images/Feature Engineering 5 - Feature Selection Techniques_files/Feature Engineering 5 - Feature Selection Techniques_22_1.png" alt="linearly separable data">



```python
# Removing highly correlated independent features as both will have same kind of impact on dependent variable
```


```python
threshold=0.5 # This threshold can be decided with domain knowledge
```


```python
# Find and remove correlated features
def correlation(dataset,threshold):
    col_corr=set() #set of all the names of correlated columns
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
```


```python
correlation(X,threshold)
```




    {'pc', 'px_width', 'sc_w', 'three_g'}




```python
X.drop(list(correlation(X,threshold)),axis=1).columns
```




    Index(['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
           'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'px_height', 'ram',
           'sc_h', 'talk_time', 'touch_screen', 'wifi'],
          dtype='object')



* In case of two highly correlated column we usually keep the column which have higher correlated value with dependent variable.

## 4. Information Gain

Mutual information (MI) between two random variables is a non-negative
value, which measures the dependency between the variables. It is equal
to zero if and only if two random variables are independent, and higher
values mean higher dependency.


```python
from sklearn.feature_selection import mutual_info_classif
```


```python
mutual_info_classif(X,y)
```




    array([2.76010315e-02, 2.30000049e-03, 1.74688267e-02, 3.30799027e-04,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.35307972e-02,
           1.32492474e-02, 0.00000000e+00, 1.31108117e-02, 2.41517674e-02,
           2.75452757e-02, 8.46137627e-01, 9.90067340e-03, 2.28336326e-03,
           0.00000000e+00, 3.42595117e-03, 0.00000000e+00, 0.00000000e+00])




```python
mutual_data=pd.Series(mutual_info_classif(X,y),index=X.columns)
```


```python
mutual_data.sort_values(ascending=False)
```




    ram              0.849398
    battery_power    0.030416
    px_width         0.029987
    px_height        0.027528
    wifi             0.022673
    m_dep            0.021102
    mobile_wt        0.018650
    four_g           0.011000
    sc_h             0.010010
    pc               0.007630
    sc_w             0.006359
    touch_screen     0.004327
    three_g          0.001078
    talk_time        0.000000
    int_memory       0.000000
    fc               0.000000
    dual_sim         0.000000
    clock_speed      0.000000
    blue             0.000000
    n_cores          0.000000
    dtype: float64



* we will take only those variable in consideration which have nonzero info gain.


```python
mutual_data[mutual_data.sort_values(ascending=False)>0]
```




    battery_power    0.030416
    four_g           0.011000
    m_dep            0.021102
    mobile_wt        0.018650
    pc               0.007630
    px_height        0.027528
    px_width         0.029987
    ram              0.849398
    sc_h             0.010010
    sc_w             0.006359
    three_g          0.001078
    touch_screen     0.004327
    wifi             0.022673
    dtype: float64



## 5. Dropping Constant Features using VarianceThreshold

* Variance Threshold
    - Feature selector that removes all low-variance features.

    - This feature selection algorithm looks only at the features (X), not the
desired outputs (y), and can thus be used for unsupervised learning.


```python
data=pd.DataFrame({'A':[1,2,4,1,2,4],
                  'B':[4,5,6,7,8,9],
                  'C':[0,0,0,0,0,0],
                  'D':[1,1,1,1,1,1]})
```


```python
data
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
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.feature_selection import VarianceThreshold
```


```python
var_thres=VarianceThreshold(threshold=0)
var_thres.fit(data)
```




    VarianceThreshold(threshold=0)




```python
var_thres.get_support() #Indicated which of the columns have variance above the threshold value
```




    array([ True,  True, False, False])




```python
# Non-Constant Columns
pd.Series(data.columns.tolist())[var_thres.get_support()]
```




    0    A
    1    B
    dtype: object




```python
# Constant Columns
pd.Series(data.columns.tolist())[~var_thres.get_support()]
```




    2    C
    3    D
    dtype: object



* Let's practise on bigger dataset


```python
df5=pd.read_csv('Datasets/Santander_customer_Satisfaction/train.csv',nrows=10000)
```


```python
df5.shape
```




    (10000, 371)




```python
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
      <th>ID</th>
      <th>var3</th>
      <th>var15</th>
      <th>imp_ent_var16_ult1</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>imp_op_var39_comer_ult3</th>
      <th>imp_op_var40_comer_ult1</th>
      <th>imp_op_var40_comer_ult3</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>...</th>
      <th>saldo_medio_var33_hace2</th>
      <th>saldo_medio_var33_hace3</th>
      <th>saldo_medio_var33_ult1</th>
      <th>saldo_medio_var33_ult3</th>
      <th>saldo_medio_var44_hace2</th>
      <th>saldo_medio_var44_hace3</th>
      <th>saldo_medio_var44_ult1</th>
      <th>saldo_medio_var44_ult3</th>
      <th>var38</th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>39205.170000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>34</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>49278.030000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>67333.770000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>8</td>
      <td>2</td>
      <td>37</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>64007.970000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>39</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117310.979016</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 371 columns</p>
</div>




```python
X5=df5.drop('TARGET',axis=1)
y5=df5['TARGET']
```


```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,y_test=train_test_split(X5,y5,test_size=0.3,random_state=0)
```


```python
X_train.shape,X_test.shape
```




    ((7000, 370), (3000, 370))




```python
# Let's apply variance method
```


```python
var_thres5=VarianceThreshold(threshold=0)
```


```python
var_thres5.fit(X_train)
```




    VarianceThreshold(threshold=0)




```python
#Finding number of non constant feature
var_thres5.get_support().sum()
```




    284




```python
# Non-Constant Columns
pd.Series(X_train.columns.tolist())[var_thres5.get_support()].tolist()
```




    ['ID',
     'var3',
     'var15',
     'imp_ent_var16_ult1',
     'imp_op_var39_comer_ult1',
     'imp_op_var39_comer_ult3',
     'imp_op_var40_comer_ult1',
     'imp_op_var40_comer_ult3',
     'imp_op_var40_efect_ult1',
     'imp_op_var40_efect_ult3',
     'imp_op_var40_ult1',
     'imp_op_var41_comer_ult1',
     'imp_op_var41_comer_ult3',
     'imp_op_var41_efect_ult1',
     'imp_op_var41_efect_ult3',
     'imp_op_var41_ult1',
     'imp_op_var39_efect_ult1',
     'imp_op_var39_efect_ult3',
     'imp_op_var39_ult1',
     'imp_sal_var16_ult1',
     'ind_var1_0',
     'ind_var1',
     'ind_var5_0',
     'ind_var5',
     'ind_var6_0',
     'ind_var6',
     'ind_var8_0',
     'ind_var8',
     'ind_var12_0',
     'ind_var12',
     'ind_var13_0',
     'ind_var13_corto_0',
     'ind_var13_corto',
     'ind_var13_largo_0',
     'ind_var13_largo',
     'ind_var13',
     'ind_var14_0',
     'ind_var14',
     'ind_var17_0',
     'ind_var17',
     'ind_var19',
     'ind_var20_0',
     'ind_var20',
     'ind_var24_0',
     'ind_var24',
     'ind_var25_cte',
     'ind_var26_0',
     'ind_var26_cte',
     'ind_var26',
     'ind_var25_0',
     'ind_var25',
     'ind_var29_0',
     'ind_var29',
     'ind_var30_0',
     'ind_var30',
     'ind_var31_0',
     'ind_var31',
     'ind_var32_cte',
     'ind_var32_0',
     'ind_var32',
     'ind_var33_0',
     'ind_var33',
     'ind_var37_cte',
     'ind_var37_0',
     'ind_var37',
     'ind_var39_0',
     'ind_var40_0',
     'ind_var40',
     'ind_var41_0',
     'ind_var39',
     'ind_var44_0',
     'ind_var44',
     'num_var1_0',
     'num_var1',
     'num_var4',
     'num_var5_0',
     'num_var5',
     'num_var6_0',
     'num_var6',
     'num_var8_0',
     'num_var8',
     'num_var12_0',
     'num_var12',
     'num_var13_0',
     'num_var13_corto_0',
     'num_var13_corto',
     'num_var13_largo_0',
     'num_var13_largo',
     'num_var13',
     'num_var14_0',
     'num_var14',
     'num_var17_0',
     'num_var17',
     'num_var20_0',
     'num_var20',
     'num_var24_0',
     'num_var24',
     'num_var26_0',
     'num_var26',
     'num_var25_0',
     'num_var25',
     'num_op_var40_hace2',
     'num_op_var40_hace3',
     'num_op_var40_ult1',
     'num_op_var40_ult3',
     'num_op_var41_hace2',
     'num_op_var41_hace3',
     'num_op_var41_ult1',
     'num_op_var41_ult3',
     'num_op_var39_hace2',
     'num_op_var39_hace3',
     'num_op_var39_ult1',
     'num_op_var39_ult3',
     'num_var29_0',
     'num_var29',
     'num_var30_0',
     'num_var30',
     'num_var31_0',
     'num_var31',
     'num_var32_0',
     'num_var32',
     'num_var33_0',
     'num_var33',
     'num_var35',
     'num_var37_med_ult2',
     'num_var37_0',
     'num_var37',
     'num_var39_0',
     'num_var40_0',
     'num_var40',
     'num_var41_0',
     'num_var39',
     'num_var42_0',
     'num_var42',
     'num_var44_0',
     'num_var44',
     'saldo_var1',
     'saldo_var5',
     'saldo_var6',
     'saldo_var8',
     'saldo_var12',
     'saldo_var13_corto',
     'saldo_var13_largo',
     'saldo_var13',
     'saldo_var14',
     'saldo_var17',
     'saldo_var20',
     'saldo_var24',
     'saldo_var26',
     'saldo_var25',
     'saldo_var29',
     'saldo_var30',
     'saldo_var31',
     'saldo_var32',
     'saldo_var33',
     'saldo_var37',
     'saldo_var40',
     'saldo_var42',
     'saldo_var44',
     'var36',
     'delta_imp_aport_var13_1y3',
     'delta_imp_aport_var17_1y3',
     'delta_imp_aport_var33_1y3',
     'delta_imp_compra_var44_1y3',
     'delta_imp_reemb_var13_1y3',
     'delta_imp_trasp_var17_in_1y3',
     'delta_imp_trasp_var33_in_1y3',
     'delta_imp_venta_var44_1y3',
     'delta_num_aport_var13_1y3',
     'delta_num_aport_var17_1y3',
     'delta_num_aport_var33_1y3',
     'delta_num_compra_var44_1y3',
     'delta_num_reemb_var13_1y3',
     'delta_num_trasp_var17_in_1y3',
     'delta_num_trasp_var33_in_1y3',
     'delta_num_venta_var44_1y3',
     'imp_aport_var13_hace3',
     'imp_aport_var13_ult1',
     'imp_aport_var17_hace3',
     'imp_aport_var17_ult1',
     'imp_aport_var33_hace3',
     'imp_aport_var33_ult1',
     'imp_var7_recib_ult1',
     'imp_compra_var44_hace3',
     'imp_compra_var44_ult1',
     'imp_reemb_var13_ult1',
     'imp_var43_emit_ult1',
     'imp_trans_var37_ult1',
     'imp_trasp_var17_in_ult1',
     'imp_trasp_var33_in_ult1',
     'imp_venta_var44_ult1',
     'ind_var7_recib_ult1',
     'ind_var10_ult1',
     'ind_var10cte_ult1',
     'ind_var9_cte_ult1',
     'ind_var9_ult1',
     'ind_var43_emit_ult1',
     'ind_var43_recib_ult1',
     'var21',
     'num_aport_var13_hace3',
     'num_aport_var13_ult1',
     'num_aport_var17_hace3',
     'num_aport_var17_ult1',
     'num_aport_var33_hace3',
     'num_aport_var33_ult1',
     'num_var7_recib_ult1',
     'num_compra_var44_hace3',
     'num_compra_var44_ult1',
     'num_ent_var16_ult1',
     'num_var22_hace2',
     'num_var22_hace3',
     'num_var22_ult1',
     'num_var22_ult3',
     'num_med_var22_ult3',
     'num_med_var45_ult3',
     'num_meses_var5_ult3',
     'num_meses_var8_ult3',
     'num_meses_var12_ult3',
     'num_meses_var13_corto_ult3',
     'num_meses_var13_largo_ult3',
     'num_meses_var17_ult3',
     'num_meses_var29_ult3',
     'num_meses_var33_ult3',
     'num_meses_var39_vig_ult3',
     'num_meses_var44_ult3',
     'num_op_var39_comer_ult1',
     'num_op_var39_comer_ult3',
     'num_op_var40_comer_ult1',
     'num_op_var40_comer_ult3',
     'num_op_var40_efect_ult1',
     'num_op_var40_efect_ult3',
     'num_op_var41_comer_ult1',
     'num_op_var41_comer_ult3',
     'num_op_var41_efect_ult1',
     'num_op_var41_efect_ult3',
     'num_op_var39_efect_ult1',
     'num_op_var39_efect_ult3',
     'num_reemb_var13_ult1',
     'num_sal_var16_ult1',
     'num_var43_emit_ult1',
     'num_var43_recib_ult1',
     'num_trasp_var11_ult1',
     'num_trasp_var17_in_ult1',
     'num_trasp_var33_in_ult1',
     'num_venta_var44_ult1',
     'num_var45_hace2',
     'num_var45_hace3',
     'num_var45_ult1',
     'num_var45_ult3',
     'saldo_medio_var5_hace2',
     'saldo_medio_var5_hace3',
     'saldo_medio_var5_ult1',
     'saldo_medio_var5_ult3',
     'saldo_medio_var8_hace2',
     'saldo_medio_var8_hace3',
     'saldo_medio_var8_ult1',
     'saldo_medio_var8_ult3',
     'saldo_medio_var12_hace2',
     'saldo_medio_var12_hace3',
     'saldo_medio_var12_ult1',
     'saldo_medio_var12_ult3',
     'saldo_medio_var13_corto_hace2',
     'saldo_medio_var13_corto_hace3',
     'saldo_medio_var13_corto_ult1',
     'saldo_medio_var13_corto_ult3',
     'saldo_medio_var13_largo_hace2',
     'saldo_medio_var13_largo_hace3',
     'saldo_medio_var13_largo_ult1',
     'saldo_medio_var13_largo_ult3',
     'saldo_medio_var17_hace2',
     'saldo_medio_var17_hace3',
     'saldo_medio_var17_ult1',
     'saldo_medio_var17_ult3',
     'saldo_medio_var29_ult1',
     'saldo_medio_var29_ult3',
     'saldo_medio_var33_hace2',
     'saldo_medio_var33_hace3',
     'saldo_medio_var33_ult1',
     'saldo_medio_var33_ult3',
     'saldo_medio_var44_hace2',
     'saldo_medio_var44_hace3',
     'saldo_medio_var44_ult1',
     'saldo_medio_var44_ult3',
     'var38']




```python
# Constant Columns
const_col=pd.Series(X_train.columns.tolist())[~var_thres5.get_support()].tolist()
```


```python
X_train.drop(const_col,axis=1)
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
      <th>ID</th>
      <th>var3</th>
      <th>var15</th>
      <th>imp_ent_var16_ult1</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>imp_op_var39_comer_ult3</th>
      <th>imp_op_var40_comer_ult1</th>
      <th>imp_op_var40_comer_ult3</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>...</th>
      <th>saldo_medio_var29_ult3</th>
      <th>saldo_medio_var33_hace2</th>
      <th>saldo_medio_var33_hace3</th>
      <th>saldo_medio_var33_ult1</th>
      <th>saldo_medio_var33_ult3</th>
      <th>saldo_medio_var44_hace2</th>
      <th>saldo_medio_var44_hace3</th>
      <th>saldo_medio_var44_ult1</th>
      <th>saldo_medio_var44_ult3</th>
      <th>var38</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7681</td>
      <td>15431</td>
      <td>2</td>
      <td>42</td>
      <td>840.0</td>
      <td>4477.02</td>
      <td>4989.54</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37491.21</td>
    </tr>
    <tr>
      <td>9031</td>
      <td>18181</td>
      <td>2</td>
      <td>31</td>
      <td>0.0</td>
      <td>52.32</td>
      <td>52.32</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>106685.94</td>
    </tr>
    <tr>
      <td>3691</td>
      <td>7411</td>
      <td>2</td>
      <td>51</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>66144.66</td>
    </tr>
    <tr>
      <td>202</td>
      <td>407</td>
      <td>2</td>
      <td>36</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>92121.36</td>
    </tr>
    <tr>
      <td>5625</td>
      <td>11280</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>74650.83</td>
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
      <td>9225</td>
      <td>18564</td>
      <td>2</td>
      <td>33</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117547.89</td>
    </tr>
    <tr>
      <td>4859</td>
      <td>9723</td>
      <td>2</td>
      <td>24</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>71050.83</td>
    </tr>
    <tr>
      <td>3264</td>
      <td>6557</td>
      <td>2</td>
      <td>24</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>141069.33</td>
    </tr>
    <tr>
      <td>9845</td>
      <td>19796</td>
      <td>2</td>
      <td>38</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>86412.15</td>
    </tr>
    <tr>
      <td>2732</td>
      <td>5441</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>45985.68</td>
    </tr>
  </tbody>
</table>
<p>7000 rows × 284 columns</p>
</div>




```python

```
