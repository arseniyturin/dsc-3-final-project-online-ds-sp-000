## Final Project Submission
Please fill out:

* Student name: Arseniy Tyurin
* Student pace: self paced
* Scheduled project review date/time: 1:00PM EAST, 05.17.2019
* Instructor name: Eli Thomas
* Blog post URL: <a href="https://medium.com/@arseniytyurin/bank-marketing-campaign-unbalanced-classification-problem-c61793a559cd" target="_blank">Bank Marketing Campaign: unbalanced classification problem</a>

# Introduction

This project is about building and optimization of classification models. Essentially I'm going to do exploratory data analysis and build and tune classification models to predict which client will subscribe to the term deposit and what features have biggest effect on persons decision.

# The Goal

The classification goal is to predict if the client will subscribe a term deposit (variable y). The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

Source:
<a href="https://www.kaggle.com/henriqueyamahata/bank-marketing">Kaggle</a> or <a href="https://archive.ics.uci.edu/ml/datasets/Bank+Marketing">UCI</a>

## Dataset Description:
- **age** (numeric)
- **job**: type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
- **marital** : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
- **education** (categorical:'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
- **default**: has credit in default? (categorical: 'no','yes','unknown')
- **housing**: has housing loan? (categorical: 'no','yes','unknown')
- **loan**: has personal loan? (categorical: 'no','yes','unknown')
- **contact**: contact communication type (categorical: 'cellular','telephone') 
- **month**: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
- **day_of_week**: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
- **duration**: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

- **campaign**: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- **pdays**: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
- **previous**: number of contacts performed before this campaign and for this client (numeric)
- **poutcome**: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
- **emp.var.rate**: employment variation rate - quarterly indicator (numeric)
- **cons.price.idx**: consumer price index - monthly indicator (numeric)     
- **cons.conf.idx**: consumer confidence index - monthly indicator (numeric)     
- **euribor3m**: euribor 3 month rate - daily indicator (numeric)
- **nr.employed**: number of employees - quarterly indicator (numeric)
- **y** - has the client subscribed a term deposit? (binary: 'yes','no')

# Data Science Process

A successful completion of the analysis requires clear understanding of business problem.
For this project I'm going to use <a href="https://towardsdatascience.com/5-steps-of-a-data-science-project-lifecycle-26c50372b492" target="_blank">OSEMN Framework</a> (Obtain-Scrab-Explore-Model-Interpret), which is one of the most common ways of doing analysis and building predictive models.

# Importing Libraries


```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Data Import and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load # save/load trained model
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Validation
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import calinski_harabaz_score

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb 
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")
```


```python
plt.style.use('seaborn-whitegrid')
```

## Supporting Functions


```python
def conf_matrix(cnf_matrix):
    '''
    Visualization function for confusion matrix
    '''
    plt.imshow(cnf_matrix,  cmap=plt.cm.Blues) # Create the basic matrix.

    #Add title and Axis Labels
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    #Add appropriate Axis Scales
    class_names = set(y) #Get class labels to add to matrix
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.style.use('seaborn-white')
    #Add Labels to Each Cell
    thresh = cnf_matrix.max() / 2. #Used for text coloring below
    #Here we iterate through the confusion matrix and append labels to our visualization.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, cnf_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black")

    #Add a Side Bar Legend Showing Colors
    plt.colorbar()
    
def plot_feature_importances(model):
    '''
    Visualization for future importance
    '''
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,12))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X.columns.values) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
```

# 1. Obtain

This is a simple step because I already have all the data I need in .csv file.


```python
# Reading data into 'bank' dataframe
bank = pd.read_csv('assets/bank/bank-additional-full.csv', sep=';')
```

To make sure data is loaded properly I would look at top 5 rows


```python
bank.head()
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
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>...</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>housemaid</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>unknown</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>admin.</td>
      <td>married</td>
      <td>basic.6y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Data is loaded correctly, I'm going to the next step of cleaning and transforming data

# 2. Scrub

### Checking for null values


```python
bank.isna().sum()
```




    age               0
    job               0
    marital           0
    education         0
    default           0
    housing           0
    loan              0
    contact           0
    month             0
    day_of_week       0
    duration          0
    campaign          0
    pdays             0
    previous          0
    poutcome          0
    emp.var.rate      0
    cons.price.idx    0
    cons.conf.idx     0
    euribor3m         0
    nr.employed       0
    y                 0
    dtype: int64



There are no missing values in the entire dataset

### Checking for data types


```python
bank.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41188 entries, 0 to 41187
    Data columns (total 21 columns):
    age               41188 non-null int64
    job               41188 non-null object
    marital           41188 non-null object
    education         41188 non-null object
    default           41188 non-null object
    housing           41188 non-null object
    loan              41188 non-null object
    contact           41188 non-null object
    month             41188 non-null object
    day_of_week       41188 non-null object
    duration          41188 non-null int64
    campaign          41188 non-null int64
    pdays             41188 non-null int64
    previous          41188 non-null int64
    poutcome          41188 non-null object
    emp.var.rate      41188 non-null float64
    cons.price.idx    41188 non-null float64
    cons.conf.idx     41188 non-null float64
    euribor3m         41188 non-null float64
    nr.employed       41188 non-null float64
    y                 41188 non-null object
    dtypes: float64(5), int64(5), object(11)
    memory usage: 6.6+ MB


### Removing `campaign` and `duration`

Since we can't predict how long we gonna talk to the client and how many calls would require to get the answer about deposit, I will drop these feature from the analysis.


```python
bank.drop(['campaign','duration'], axis=1, inplace=True)
```

# 3. Explore


```python
bank.head()
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
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>housemaid</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>unknown</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>admin.</td>
      <td>married</td>
      <td>basic.6y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
bank.describe()
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
      <th>pdays</th>
      <th>previous</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>41188.00000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.02406</td>
      <td>962.475454</td>
      <td>0.172963</td>
      <td>0.081886</td>
      <td>93.575664</td>
      <td>-40.502600</td>
      <td>3.621291</td>
      <td>5167.035911</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.42125</td>
      <td>186.910907</td>
      <td>0.494901</td>
      <td>1.570960</td>
      <td>0.578840</td>
      <td>4.628198</td>
      <td>1.734447</td>
      <td>72.251528</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-3.400000</td>
      <td>92.201000</td>
      <td>-50.800000</td>
      <td>0.634000</td>
      <td>4963.600000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.00000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>-1.800000</td>
      <td>93.075000</td>
      <td>-42.700000</td>
      <td>1.344000</td>
      <td>5099.100000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>38.00000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>1.100000</td>
      <td>93.749000</td>
      <td>-41.800000</td>
      <td>4.857000</td>
      <td>5191.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>47.00000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>1.400000</td>
      <td>93.994000</td>
      <td>-36.400000</td>
      <td>4.961000</td>
      <td>5228.100000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>98.00000</td>
      <td>999.000000</td>
      <td>7.000000</td>
      <td>1.400000</td>
      <td>94.767000</td>
      <td>-26.900000</td>
      <td>5.045000</td>
      <td>5228.100000</td>
    </tr>
  </tbody>
</table>
</div>



### Target Variable

First of all I want to see how my target variable looks like


```python
plt.bar(['No', 'Yes'], bank.y.value_counts().values, facecolor = '#2ab0ee', edgecolor='#167aaa', linewidth=0.5)
plt.title('Target Variable', fontsize=14)
plt.xlabel('Classes')
plt.ylabel('Amount')
plt.show()
```


![png](images/Project%203%20Student_34_0.png)


Target variable is extremely inbalanced. This is important to remember when performing classification and evaluation, because even without using of machine learning I can make predictions with roughly 90% accuracy just by guessing none of the clients subscribed to the term deposit. Since we are focused only on the clients that said 'Yes', chances to get predictions are very thin.


```python
# Trainsforming target variable to the boolean
bank.y.replace({'yes': 1, 'no': 0}, inplace=True)
```

### Previous results of marketing campaign


```python
plt.bar(['Failure', 'Success'], [sum(bank.poutcome=='failure'), sum(bank.poutcome=='success')], \
        facecolor = '#2ab0ee', edgecolor='#167aaa', linewidth=0.5)
plt.title('Previous Campaign', fontsize=14)
plt.xlabel('Classes')
plt.ylabel('Amount')
plt.show()
```


![png](images/Project%203%20Student_38_0.png)


### Returned clients that subscribed to the new term deposit


```python
returned_customers = len(bank[(bank.poutcome=='success') & (bank.y==1)])/len(bank[bank.poutcome=='success'])
print('Returned clients who has subscribed to the new term deposit: {}%'.format(round(returned_customers*100)))
```

    Returned clients who has subscribed to the new term deposit: 65%


### Euro Interbank Offered Rate

Euribor is short for Euro Interbank Offered Rate. The Euribor rates are based on the average interest rates at which a large panel of European banks borrow funds from one another. There are different maturities, ranging from one week to one year. The Euribor rates are considered to be the most important reference rates in the European money market.

<a href="https://www.euribor-rates.eu/what-is-euribor.asp" target="_blank">Click here</a> for more information on the subject


```python
plt.bar(['Failure', 'Success'], [bank[bank.poutcome=='failure'].euribor3m.mean(), bank[bank.poutcome=='success'].euribor3m.mean()], \
        facecolor = '#2ab0ee', edgecolor='#167aaa', linewidth=0.5)
plt.title('Euribor', fontsize=14)
plt.xlabel('Classes')
plt.ylabel('Mean Rate')
plt.show()
```


![png](images/Project%203%20Student_43_0.png)



```python
# Mean Euribor for subscribed/not subscribed client
euribor_1 = bank[bank.y==1].euribor3m.mean()
euribor_0 = bank[bank.y==0].euribor3m.mean()

print('Mean Euribor when client said "Yes":', round(euribor_1, 2))
print('Mean Euribor when client said "No":', round(euribor_0, 2))
```

    Mean Euribor when client said "Yes": 2.12
    Mean Euribor when client said "No": 3.81


Lower the interest rate - better product a bank can offer to the client.

### Job distribution


```python
plt.figure(figsize=(16,6))
plt.bar(bank.job.value_counts().keys(), bank.job.value_counts().values, label='Jobs', facecolor = '#2ab0ee', edgecolor='#167aaa', linewidth=0.5)
plt.bar(bank.job.value_counts().keys(), [bank[bank.job==i].y.sum() for i in bank.job.value_counts().keys()], label='Subscribed')
plt.bar(bank.job.value_counts().keys(), [bank[(bank.poutcome=='success') & (bank.job==i)].y.sum() for i in bank.job.value_counts().keys()], label='Prev. Subscribed')
plt.title('Jobs', fontsize=14)
plt.xlabel('Jobs')
plt.ylabel('Amount')
plt.legend()
plt.show()
```


![png](images/Project%203%20Student_47_0.png)


### Correlation between Job and Euribor


```python
jobs = sorted(bank.job.unique())
```


```python
# Proportion of clients who subscribed to the term deposit grouped by occupation
yes = (round(bank.groupby('job').y.sum()/bank.groupby('job').y.count(),3)*100).values

# Total amount of clients per occupation
total = bank.groupby('job').y.count().values

# Average Euribor parameter per occupation
euribor = round(bank.groupby('job').euribor3m.mean(),3).values
```


```python
jobs_euribor_df = pd.DataFrame({'Job': jobs, 'Yes': yes, 'Total': total, 'Euribor': euribor})
jobs_euribor_df.sort_values(by='Yes', ascending=False)
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
      <th>Job</th>
      <th>Yes</th>
      <th>Total</th>
      <th>Euribor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>student</td>
      <td>31.4</td>
      <td>875</td>
      <td>1.884</td>
    </tr>
    <tr>
      <th>5</th>
      <td>retired</td>
      <td>25.2</td>
      <td>1720</td>
      <td>2.770</td>
    </tr>
    <tr>
      <th>10</th>
      <td>unemployed</td>
      <td>14.2</td>
      <td>1014</td>
      <td>3.467</td>
    </tr>
    <tr>
      <th>0</th>
      <td>admin.</td>
      <td>13.0</td>
      <td>10422</td>
      <td>3.550</td>
    </tr>
    <tr>
      <th>4</th>
      <td>management</td>
      <td>11.2</td>
      <td>2924</td>
      <td>3.611</td>
    </tr>
    <tr>
      <th>11</th>
      <td>unknown</td>
      <td>11.2</td>
      <td>330</td>
      <td>3.949</td>
    </tr>
    <tr>
      <th>9</th>
      <td>technician</td>
      <td>10.8</td>
      <td>6743</td>
      <td>3.820</td>
    </tr>
    <tr>
      <th>6</th>
      <td>self-employed</td>
      <td>10.5</td>
      <td>1421</td>
      <td>3.689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>housemaid</td>
      <td>10.0</td>
      <td>1060</td>
      <td>4.010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entrepreneur</td>
      <td>8.5</td>
      <td>1456</td>
      <td>3.791</td>
    </tr>
    <tr>
      <th>7</th>
      <td>services</td>
      <td>8.1</td>
      <td>3969</td>
      <td>3.699</td>
    </tr>
    <tr>
      <th>1</th>
      <td>blue-collar</td>
      <td>6.9</td>
      <td>9254</td>
      <td>3.772</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(jobs_euribor_df.corr(), annot=True)
plt.show()
```


![png](images/Project%203%20Student_52_0.png)


There is **extremely** strong inverse correlation (96%) between Euribor parameter and proportion of the clients that subscribed to the term deposit (said 'Yes'). 

### Education Distribution


```python
plt.figure(figsize=(16,6))
plt.bar(bank.education.value_counts().keys(), bank.education.value_counts().values, label='Education')
plt.bar(bank.education.value_counts().keys(), [bank[bank.education==i].y.sum() for i in bank.education.value_counts().keys()], label='Subscribed')
plt.bar(bank.education.value_counts().keys(), [bank[(bank.poutcome=='success')&(bank.education==i)].y.sum() for i in bank.education.value_counts().keys()], label='Prev. Subscribed')
plt.title('Education', fontsize=14)
plt.xlabel('Education')
plt.ylabel('Amount')
plt.legend()
plt.show()
```


![png](images/Project%203%20Student_55_0.png)


### Marital Status Distribution


```python
plt.figure(figsize=(16,6))
plt.bar(bank.marital.value_counts().keys(), bank.marital.value_counts().values, label='Marital')
plt.bar(bank.marital.value_counts().keys(), [bank[bank.marital==i].y.sum() for i in bank.marital.value_counts().keys()], label='Subscribed')
plt.title('Marital', fontsize=14)
plt.xlabel('Marital')
plt.ylabel('Amount')
plt.legend()
plt.show()
```


![png](images/Project%203%20Student_57_0.png)



```python
married = bank[bank.marital=='married'].y.sum()/len(bank[bank.marital=='married'])
single = bank[bank.marital=='single'].y.sum()/len(bank[bank.marital=='single'])

print('Married clients campaign success rate: {0:.0f}%'.format(married*100))
print('Single clients campaign success rate: {0:.0f}%'.format(single*100))
```

    Married clients campaign success rate: 10%
    Single clients campaign success rate: 14%


Even though we have a lot more clients who are married, single clients responded to the campaign better

### Cellphone or Landline?


```python
plt.figure(figsize=(10,6))
plt.bar(['Cellular', 'Line'], [bank.contact.value_counts()[0], bank.contact.value_counts()[1]], label='Contacts')
plt.bar(['Cellular', 'Line'], [sum(bank[(bank.contact=='cellular') & (bank.y==1)].y), sum(bank[(bank.contact=='telephone') & (bank.y==1)].y)], label='Success')
plt.title('Way of Communication', fontsize=14)
plt.ylabel('Number of Calls')
plt.legend()
plt.show()
```


![png](images/Project%203%20Student_61_0.png)


### Age Distribution


```python
plt.figure(figsize=(16,6))
plt.bar(bank.groupby('age').y.sum().keys(), [sum(bank.age==i) for i in bank.groupby('age').y.sum().keys()], label='Ages')
plt.bar(bank.groupby('age').y.sum().keys(), [bank[bank.age==i].y.sum() for i in bank.groupby('age').y.sum().keys()], label='Subscribed')
plt.bar(bank.groupby('age').y.sum().keys(), [bank[(bank.poutcome=='success')&(bank.age==i)].y.sum() for i in bank.groupby('age').y.sum().keys()], label='Subscribed')
plt.title('Ages', fontsize=14)
plt.xlabel('Ages')
plt.ylabel('Amount')
plt.legend()
plt.show()
```


![png](images/Project%203%20Student_63_0.png)


### Age After 60

There is substential drop in number of clients after 60 years old, but as you can see proportion of subscribed clients are higher. Lets compare groups before/after 60 y.o.


```python
before_60 = round((bank[bank.age < 60].y.sum()/sum(bank.age < 60))*100, 2)
after_60 = round((bank[bank.age > 60].y.sum()/sum(bank.age > 60))*100, 2)

print('Before 60: {}% subscribed\nAfter 60: {}% subscribed'.format(before_60, after_60))
```

    Before 60: 10.42% subscribed
    After 60: 45.49% subscribed


### Euribor and Age after 60


```python
print('Euribor before 60 y.o:',bank[bank.age < 60].euribor3m.mean())
print('Euribor after 60 y.o:',bank[bank.age > 60].euribor3m.mean())
```

    Euribor before 60 y.o: 3.6842671333915917
    Euribor after 60 y.o: 0.9235384615384649


### Distribution of contacts made each month of the campaign


```python
plt.figure(figsize=(16,6))
plt.bar(bank.month.value_counts().keys(), bank.month.value_counts().values, label='Months')
plt.bar(bank.month.value_counts().keys(), [bank[bank.month==i].y.sum() for i in bank.month.value_counts().keys()], label='Subscribed')
plt.title('Month Clients Were Contacted', fontsize=14)
plt.xlabel('Months')
plt.ylabel('Amount')
plt.legend()
plt.show()
```


![png](images/Project%203%20Student_70_0.png)


# 4. Model

## Dealing with categorical values


```python
bank = pd.get_dummies(bank, columns=['housing','loan','job','marital', \
                                     'education','contact','month','day_of_week', \
                                     'poutcome','default'])
```

## Separating features and target variable


```python
X = bank.drop(['y'], axis=1)
y = bank.y
```

## Scaling


```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Splitting Data into Train and Test


```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
```

## SMOTE: Synthetic Minority Over-sampling Technique


```python
smote = SMOTE(random_state=0)
```


```python
X_res, y_res = smote.fit_resample(X_train, y_train)
```


```python
plt.bar(['No','Yes'], [sum(y_res), len(y_res)-sum(y_res)], facecolor = '#2ab0ee', edgecolor='#167aaa', linewidth=0.5)
plt.title('Balanced Target Variable', fontsize=14)
plt.xlabel('Classes')
plt.ylabel('Amounth')
plt.show()
```


![png](images/Project%203%20Student_83_0.png)


## Logistic Regression

I start my analysis with simple logistic regression as a benchmark. One of the features of logistic regression is the **decision function**. It predicts confidence scores for samples. Basically it shows balance between true positive / false positive rates.


```python
plt.style.use('seaborn-whitegrid')
logreg = LogisticRegression(fit_intercept=False, n_jobs=-1)

#Probability scores for test set
y_score = logreg.fit(X_res, y_res).decision_function(X_res)
y_pred = logreg.fit(X_res, y_res).predict(X_test)
#False positive Rate and true positive rate
fpr, tpr, thresholds = roc_curve(y_res, y_score)

print('AUC: {}'.format(auc(fpr, tpr)))
plt.figure(figsize=(10,8))
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks([i/20.0 for i in range(21)])
plt.xticks([i/20.0 for i in range(21)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

    AUC: 0.7996767918841073



![png](images/Project%203%20Student_86_1.png)


## Evaluation of Logistic Regression


```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.95      0.76      0.84     10969
               1       0.27      0.69      0.38      1388
    
       micro avg       0.75      0.75      0.75     12357
       macro avg       0.61      0.72      0.61     12357
    weighted avg       0.87      0.75      0.79     12357
    



```python
plt.style.use('seaborn-white')
logreg = LogisticRegression(fit_intercept=False, n_jobs=-1)
y_pred = logreg.fit(X_res, y_res).predict(X_test)
conf_matrix(confusion_matrix(y_test, y_pred))
```


![png](images/Project%203%20Student_89_0.png)


Model accurately identified true positives 956 times out of 1388, but it also got 2644 false positives, which made f1-score very low. My next step would be try different models and see which one is the best fit for my problem

## Classifiers with default parameters

At this step I will compare different models


```python
logistic_regression = LogisticRegression(n_jobs=-1)
random_forest = RandomForestClassifier(n_jobs=-1)
adaboost = AdaBoostClassifier()
gboost = GradientBoostingClassifier()
xgboost = xgb.XGBClassifier(n_jobs=-1)
naive_bayes = GaussianNB()

classifiers = [logistic_regression, random_forest, xgboost, adaboost, gboost, naive_bayes]
classifiers_names = ['Logistic Regression', 'Random Forest','XGBoost', 'AdaBoost', 'Gradient Boost','Naive Bayes']
```

## Testing classifiers with unbalanced target variable


```python
scores = []
for i in range(len(classifiers)):
    y_pred = classifiers[i].fit(X_train, y_train).predict(X_test)
    scores.append(round(f1_score(y_test, y_pred), 3))

pd.DataFrame({'Classifier': classifiers_names, 'F1-Score': scores}).sort_values('F1-Score', ascending=False)
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
      <th>Classifier</th>
      <th>F1-Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Naive Bayes</td>
      <td>0.367</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gradient Boost</td>
      <td>0.362</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest</td>
      <td>0.344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XGBoost</td>
      <td>0.334</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.327</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AdaBoost</td>
      <td>0.327</td>
    </tr>
  </tbody>
</table>
</div>



## Testing classifiers with balanced target variable


```python
scores = []
for i in range(len(classifiers)):
    y_pred = classifiers[i].fit(X_res, y_res).predict(X_test)
    scores.append(round(f1_score(y_test, y_pred), 3))

pd.DataFrame({'Classifier': classifiers_names, 'F1-Score': scores}).sort_values('F1-Score', ascending=False)
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
      <th>Classifier</th>
      <th>F1-Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Gradient Boost</td>
      <td>0.485</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XGBoost</td>
      <td>0.483</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AdaBoost</td>
      <td>0.464</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.438</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest</td>
      <td>0.375</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Naive Bayes</td>
      <td>0.323</td>
    </tr>
  </tbody>
</table>
</div>



**Gradient Boost** and **XGBoost** has shown the best performance and are pretty close on F1-Score. I will keep both of them for now.

## Gradient Boost Tuning With GridSearch


```python
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [8, 10, 12],
    'subsample': [0.5]
}
```


```python
grid = GridSearchCV(gboost, param_grid, n_jobs=-1)
```


```python
grid.fit(X_res, y_res).best_params_
```




    {'max_depth': 12, 'n_estimators': 100, 'subsample': 0.5}




```python
gboost = GradientBoostingClassifier(max_depth=12, n_estimators=100, subsample=0.5)
y_pred = gboost.fit(X_res, y_res).predict(X_test)

conf_matrix(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.92      0.96      0.94     10969
               1       0.53      0.34      0.41      1388
    
       micro avg       0.89      0.89      0.89     12357
       macro avg       0.72      0.65      0.68     12357
    weighted avg       0.88      0.89      0.88     12357
    



![png](images/Project%203%20Student_103_1.png)


## XGBoost Tuning With GridSearchCV


```python
param_grid = {
    'max_depth': [7, 10, 11], 
    'n_estimators': [15, 20, 50, 70], 
    'min_child_weight': [2, 3], 
    'subsample': [0.5],
}
gridsearch = GridSearchCV(xgboost, param_grid, n_jobs=-1)
```


```python
gridsearch.fit(X_res, y_res).best_params_
```




    {'max_depth': 11, 'min_child_weight': 2, 'n_estimators': 70, 'subsample': 0.5}




```python
xgboost = xgb.XGBClassifier(max_depth=11, 
                            learning_rate=0.1, 
                            n_estimators=70, 
                            min_child_weight=2, 
                            subsample=0.5, 
                            n_jobs=-1)
y_pred = xgboost.fit(X_res, y_res).predict(X_test)

print(classification_report(y_test, y_pred))
conf_matrix(confusion_matrix(y_test, y_pred))
#plot_feature_importances(xgboost)
```

                  precision    recall  f1-score   support
    
               0       0.92      0.97      0.94     10969
               1       0.56      0.35      0.43      1388
    
       micro avg       0.90      0.90      0.90     12357
       macro avg       0.74      0.66      0.69     12357
    weighted avg       0.88      0.90      0.89     12357
    



![png](images/Project%203%20Student_107_1.png)


## Predicting On The Entire Dataset


```python
xgboost = xgb.XGBClassifier(max_depth=11, 
                            learning_rate=0.1, 
                            n_estimators=70, 
                            min_child_weight=3, 
                            subsample=0.7, 
                            n_jobs=-1)
y_pred = xgboost.fit(X_res, y_res).predict(X_scaled)

print(classification_report(y, y_pred))
conf_matrix(confusion_matrix(y, y_pred))
plot_feature_importances(xgboost)
```

                  precision    recall  f1-score   support
    
               0       0.93      0.98      0.95     36548
               1       0.70      0.45      0.54      4640
    
       micro avg       0.92      0.92      0.92     41188
       macro avg       0.82      0.71      0.75     41188
    weighted avg       0.91      0.92      0.91     41188
    



![png](images/Project%203%20Student_109_1.png)



![png](images/Project%203%20Student_109_2.png)


## Cross Evaluation


```python
cross_val_score(xgboost, X_scaled, y, cv=10).mean()
```


    0.3648291638471358



# 5. Interpret

- Two most important factors on client's decision are Age and Euro Interbank Offered Rate (Euribor)
- 11% of clients subscribed to the new term deposit
- 65% of previous clients subscribed to the new term deposit
- 45% of clients who are over 60 y.o has subscribed to the term deposit
- XGBoost Model showed best performance
- Overall performance of the model is not great, there is a room for improvement
