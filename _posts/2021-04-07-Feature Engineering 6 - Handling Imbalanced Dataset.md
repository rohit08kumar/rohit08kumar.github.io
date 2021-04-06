---
title: "Feature Engineering 6 - Handling Imbalanced Dataset"
date: 2021-04-07
tags: [data science, Feature Engineering, messy data]
header:
  image: "/images/Feature_engineering.jpeg"
excerpt: "Data Science, Feature Engineering, Messy Data"
mathjax: "true"
---


# Handling Imbalanced Datasets


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df=pd.read_csv('Datasets/Creditcard/creditcard.csv')
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
df.shape
```




    (284807, 31)




```python
df.isna().sum()
```




    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64




```python
#To check balance of data

df['Class'].value_counts() # as Class is dependent variable
```




    0    284315
    1       492
    Name: Class, dtype: int64



* As we can see the data is not balanced


```python
X=df.drop('Class',axis=1) #Independent Features
y=df.Class                #Dependent Feature
```


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import KFold,train_test_split,GridSearchCV
```


```python
log_class=LogisticRegression()
```


```python
#Final Solution that we should try in imbalanced dataset :- Perform GridSearchCV and KFold
grid_param={'C':10.0**np.arange(-2,3),
           'penalty':['l1','l2']}

cv=KFold(n_splits=5,shuffle=False,random_state=None)
```


```python
grid_param
```




    {'C': array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02]), 'penalty': ['l1', 'l2']}




```python
clf=GridSearchCV(log_class,param_grid=grid_param,n_jobs=-1,cv=cv,scoring='f1_macro')
```


```python
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=0)
```


```python
clf.fit(X_train,y_train)
```

    /opt/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:921: UserWarning: One or more of the test scores are non-finite: [       nan 0.8150325         nan 0.83872168        nan 0.84009661
            nan 0.84109159        nan 0.83551738]
      category=UserWarning
    /opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:765: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)





    GridSearchCV(cv=KFold(n_splits=5, random_state=None, shuffle=False),
                 estimator=LogisticRegression(), n_jobs=-1,
                 param_grid={'C': array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02]),
                             'penalty': ['l1', 'l2']},
                 scoring='f1_macro')




```python
y_pred=clf.predict(X_test)
print('confusion_matrix\n',confusion_matrix(y_test,y_pred))
print('\nclassification_report\n',classification_report(y_test,y_pred))
print('\naccuracy_score\n',accuracy_score(y_test,y_pred))
```

    confusion_matrix
     [[85262    34]
     [   53    94]]

    classification_report
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00     85296
               1       0.73      0.64      0.68       147

        accuracy                           1.00     85443
       macro avg       0.87      0.82      0.84     85443
    weighted avg       1.00      1.00      1.00     85443


    accuracy_score
     0.9989817773252344


* We should not look into accuracy score in case of imbalanced dataset.
* we should focus on precision and recall values.


```python
#Trying to implement RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
```


```python
classifier=RandomForestClassifier()
```

* Explore class weight parameter to give weightage to the categories


```python
classifier.fit(X_train,y_train)
```




    RandomForestClassifier()




```python
y_pred=classifier.predict(X_test)
print('confusion_matrix\n',confusion_matrix(y_test,y_pred))
print('\nclassification_report\n',classification_report(y_test,y_pred))
print('\naccuracy_score\n',accuracy_score(y_test,y_pred))
```

    confusion_matrix
     [[85289     7]
     [   34   113]]

    classification_report
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00     85296
               1       0.94      0.77      0.85       147

        accuracy                           1.00     85443
       macro avg       0.97      0.88      0.92     85443
    weighted avg       1.00      1.00      1.00     85443


    accuracy_score
     0.9995201479348805



```python

```

## 1. Under Sampling
* Reduce the data points of maximum labels

- Disadvantage
    - Loss of data
    - we should use only where there is very less dataset


```python
from collections import Counter
from imblearn.under_sampling import NearMiss
```


```python
Counter(y_train)
```




    Counter({0: 199019, 1: 345})




```python
ns=NearMiss(0.8)
X_train_ns,y_train_ns=ns.fit_resample(X_train,y_train)
print('The number of classes before fit {}'.format(Counter(y_train)))
print('The number of classes before fit {}'.format(Counter(y_train_ns)))
```

    /opt/anaconda3/lib/python3.7/site-packages/imblearn/utils/_validation.py:591: FutureWarning: Pass sampling_strategy=0.8 as keyword args. From version 0.9 passing these as positional arguments will result in an error
      FutureWarning,


    The number of classes before fit Counter({0: 199019, 1: 345})
    The number of classes before fit Counter({0: 431, 1: 345})



```python
# 0.8 of x = 345
# x=345/0.8

356/0.8  # approximately Number of reduced maximum labels
```




    445.0




```python
classifier=RandomForestClassifier()
classifier.fit(X_train_ns,y_train_ns)
```




    RandomForestClassifier()




```python
y_pred=classifier.predict(X_test)
print('confusion_matrix\n',confusion_matrix(y_test,y_pred))
print('\nclassification_report\n',classification_report(y_test,y_pred))
print('\naccuracy_score\n',accuracy_score(y_test,y_pred))
```

    confusion_matrix
     [[62817 22479]
     [    8   139]]

    classification_report
                   precision    recall  f1-score   support

               0       1.00      0.74      0.85     85296
               1       0.01      0.95      0.01       147

        accuracy                           0.74     85443
       macro avg       0.50      0.84      0.43     85443
    weighted avg       1.00      0.74      0.85     85443


    accuracy_score
     0.7368186978453471


## 2. Over Sampling
- Adding more points belonging to minority category
- Existing points will be created multiple times/ replicated multiple times to oversample.


```python
from imblearn.over_sampling import RandomOverSampler

```


```python
os=RandomOverSampler(0.75)
X_train_os,y_train_os=os.fit_resample(X_train,y_train)
print('The number of classes before fit {}'.format(Counter(y_train)))
print('The number of classes before fit {}'.format(Counter(y_train_os)))
```

    /opt/anaconda3/lib/python3.7/site-packages/imblearn/utils/_validation.py:591: FutureWarning: Pass sampling_strategy=0.75 as keyword args. From version 0.9 passing these as positional arguments will result in an error
      FutureWarning,


    The number of classes before fit Counter({0: 199019, 1: 345})
    The number of classes before fit Counter({0: 199019, 1: 149264})



```python
#number of minority sample after oversampling
199019*0.75
```




    149264.25




```python
classifier.fit(X_train_os,y_train_os)
y_pred=classifier.predict(X_test)
print('confusion_matrix\n',confusion_matrix(y_test,y_pred))
print('\nclassification_report\n',classification_report(y_test,y_pred))
print('\naccuracy_score\n',accuracy_score(y_test,y_pred))
```

    confusion_matrix
     [[85290     6]
     [   33   114]]

    classification_report
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00     85296
               1       0.95      0.78      0.85       147

        accuracy                           1.00     85443
       macro avg       0.97      0.89      0.93     85443
    weighted avg       1.00      1.00      1.00     85443


    accuracy_score
     0.9995435553526912


## 3. SMOTETomek

- Additional new points created belonging to minority category.
- Not overlapping points but altogether new points will be created (based on nearest neighbour)



```python
from imblearn.combine import SMOTETomek
```


```python
smote=SMOTETomek(0.5)

X_train_sm,y_train_sm=smote.fit_resample(X_train,y_train)
print('The number of classes before fit {}'.format(Counter(y_train)))
print('The number of classes before fit {}'.format(Counter(y_train_sm)))

```

    The number of classes before fit Counter({0: 199019, 1: 345})
    The number of classes before fit Counter({0: 198085, 1: 98575})



```python
198085*0.5
```




    99042.5




```python
classifier.fit(X_train_sm,y_train_sm)
y_pred=classifier.predict(X_test)
print('confusion_matrix\n',confusion_matrix(y_test,y_pred))
print('\nclassification_report\n',classification_report(y_test,y_pred))
print('\naccuracy_score\n',accuracy_score(y_test,y_pred))
```

    confusion_matrix
     [[85283    13]
     [   27   120]]

    classification_report
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00     85296
               1       0.90      0.82      0.86       147

        accuracy                           1.00     85443
       macro avg       0.95      0.91      0.93     85443
    weighted avg       1.00      1.00      1.00     85443


    accuracy_score
     0.9995318516437859


* Oversampling/Undersampling should be applied during model creation

## 4. Ensemble Technique


```python
from imblearn.ensemble import EasyEnsembleClassifier
```


```python
easy=EasyEnsembleClassifier()
easy.fit(X_train,y_train)
```




    EasyEnsembleClassifier()




```python

y_pred=easy.predict(X_test)
print('confusion_matrix\n',confusion_matrix(y_test,y_pred))
print('\nclassification_report\n',classification_report(y_test,y_pred))
print('\naccuracy_score\n',accuracy_score(y_test,y_pred))
```

    confusion_matrix
     [[82902  2394]
     [   16   131]]

    classification_report
                   precision    recall  f1-score   support

               0       1.00      0.97      0.99     85296
               1       0.05      0.89      0.10       147

        accuracy                           0.97     85443
       macro avg       0.53      0.93      0.54     85443
    weighted avg       1.00      0.97      0.98     85443


    accuracy_score
     0.9717940615381014



```python

```
