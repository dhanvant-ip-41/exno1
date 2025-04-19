# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
REG NO : 212224040070
NAME : Dhanvant Kumar V

```python
import pandas as pd
df=pd.read_csv("/content/SAMPLEIDS.csv")
df
```
![alt text](<Screenshot 2025-04-19 090307.png>)
```python
df.shape
```
![alt text](<Screenshot 2025-04-19 090617.png>)
```python
df.describe()
```
![alt text](<Screenshot 2025-04-19 090636.png>)
```python
df.info()
```
![alt text](<Screenshot 2025-04-19 090645.png>)
```python
df.head(10)
```
![alt text](<Screenshot 2025-04-19 090700.png>)
```python
df.tail(10)
```
![alt text](<Screenshot 2025-04-19 090715.png>)
```python
df.isna().sum()
```
![alt text](<Screenshot 2025-04-19 090726.png>)
```python
df.dropna(how='any').shape
```
![alt text](<Screenshot 2025-04-19 090743.png>)
```python
x=df.dropna(how='any')
x
```
![alt text](<Screenshot 2025-04-19 090913.png>)
```python
mn=df.TOTAL.mean()
mn
```
![alt text](<Screenshot 2025-04-19 090924.png>)
```python
df.TOTAL.fillna(mn,inplace=True)
df
```
![alt text](<Screenshot 2025-04-19 091002.png>)
```python
df.isnull().sum()
```
![alt text](<Screenshot 2025-04-19 091013.png>)
```python
df.M1.fillna(method='ffill',inplace=True)
df
```
![alt text](<Screenshot 2025-04-19 091030.png>)
```python
df.isnull().sum()
```
![alt text](<Screenshot 2025-04-19 091041.png>)
```python
df.M2.fillna(method='ffill',inplace=True)
df
```
![alt text](<Screenshot 2025-04-19 091056.png>)
```python
df.isna().sum()
```
![alt text](<Screenshot 2025-04-19 091108.png>)
```python
df.M3.fillna(method='ffill',inplace=True)
df
```
![alt text](<Screenshot 2025-04-19 091129.png>)
```python
df.isnull().sum()
```
![alt text](<Screenshot 2025-04-19 091540.png>)
```python
df.duplicated()
```
![alt text](<Screenshot 2025-04-19 091557.png>)
```python
df.drop_duplicates(inplace=True)
df
```
![alt text](<Screenshot 2025-04-19 091656.png>)
```python
df.duplicated()
```
![alt text](<Screenshot 2025-04-19 091734.png>)
```python
df['DOB']
```
![alt text](<Screenshot 2025-04-19 091747.png>)
```python
import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
```
![alt text](<Screenshot 2025-04-19 091901.png>)
```python
df.dropna(inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
```
![alt text](<Screenshot 2025-04-19 091912.png>)
```python
age=[1,3,28,27,25,92,30,39,40,50,26,24,29,94]
dr=pd.DataFrame(age)
dr
```
![alt text](<Screenshot 2025-04-19 091920.png>)
```python
sns.boxplot(data=dr)
```
![alt text](<Screenshot 2025-04-19 092020.png>)
```python
sns.scatterplot(data=dr)
```
![alt text](<Screenshot 2025-04-19 092028.png>)
```python
q1=dr.quantile(0.25)
q2=dr.quantile(0.5)
q3=dr.quantile(0.75)
iqr=q3-q1
iqr
```
![alt text](<Screenshot 2025-04-19 092138.png>)
```python
import numpy as np
Q1=np.percentile(dr,25)
Q3=np.percentile(dr,75)
IQR=Q3-Q1
IQR
```
![alt text](<Screenshot 2025-04-19 092146.png>)
```python
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
lower_bound
```
![alt text](<Screenshot 2025-04-19 092212.png>)
```python
upper_bound
```
![alt text](<Screenshot 2025-04-19 092218.png>)
```python
outliers=[x for x in age if x<lower_bound or x>upper_bound]
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("Lower Bound:",lower_bound)
print("Upper Bound:",upper_bound)
print("Outliers:",outliers)
```
![alt text](<Screenshot 2025-04-19 092227.png>)
```python
dr=dr[(dr>=lower_bound)&(dr<=upper_bound)]
dr
```
![alt text](<Screenshot 2025-04-19 092237.png>)
```python
dr=dr.dropna()
dr
```
![alt text](<Screenshot 2025-04-19 095445.png>)
```python
sns.boxplot(data=dr)
```
![alt text](<Screenshot 2025-04-19 095456.png>)
```python
sns.scatterplot(dr)
```
![alt text](<Screenshot 2025-04-19 095505.png>)
```python
data=[1,2,2,2,3,1,1,15,2,2,2,3,1,1,2]
mean=np.mean(data)
std=np.std(data)
print("mean of the dataset is",mean)
print("std.deviation is",std)
```
![alt text](<Screenshot 2025-04-19 095546.png>)
```python
threshold=3
outlier=[]
for i in data:
  z=i-mean/std
  if(z>threshold):
    outlier.append(i)
print("Outlier in dataset is",outlier)
```
![alt text](<Screenshot 2025-04-19 095608.png>)
```python
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
data={'weight':[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]}
df=pd.DataFrame(data)
df
```
![alt text](<Screenshot 2025-04-19 095652.png>)
```python
z=np.abs(stats.zscore(df))
print(df[z['weight']>3])
```
![alt text](<Screenshot 2025-04-19 095706.png>)
# Result
Thus we have cleaned the data and remove the outliers by detection using IQR and Z-score method.
