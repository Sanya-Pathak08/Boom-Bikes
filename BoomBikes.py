import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
#Reading the dataset
dataset=pd.read_csv("BoomBikes.csv")
dataset.head()
dataset.shape
dataset.columns
dataset.describe()
dataset.info()
#Assigning string value to different seasons instead o numeric values
#1-Spring
dataset.loc[(dataset['season']==1),'Season']='spring'
#2-summer
dataset.loc[(dataset['season']==2),'Season']='summer'
#3-fall
dataset.loc[(dataset['season']==3),'Season']='fall'
#4-winter
dataset.loc[(dataset['season']==7),'Season']='winter'
dataset['mnth'].astype('category').value_counts()
dataset['yr'].astype('category').value_counts()
def object_map_mnths(x):
    return x.map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
dataset[['mnth']]=dataset[['mnth']].apply(object_map_mnths)
dataset['mnth'].astype('category').value_counts()
dataset['holiday'].astype('category').value_counts()
def str_map_weekday(x):
    return x.map({1:'Mon',2:'Tue',3:'Wed',4:'Thrus',5:'Fri',6:'Sat',7:'Sun'})
dataset[['weekday']] =dataset[['weekday']].apply(str_map_weekday)
dataset['weekday'].astype('category').value_counts()
dataset['workingday'].astype('category').value_counts()
#1= Clear,few clouds, Partly cloudly
dataset.loc[(dataset['weathersit']==1),'weathersit'] ='A'

#1= Mist, cloudy
dataset.loc[(dataset['weathersit']==2),'weathersit'] ='B'

#1= Light Snow,Heavy Rain
dataset.loc[(dataset['weathersit']==3),'weathersit'] ='C'
dataset['weathersit'].astype('category').value_counts()


##2.Data Visualisation
#Importing libs
import matplotlib.pyplot as plt
import seaborn as sns
#Temprature
sns.distplot(dataset['temp'])
#Actual Temperature
sns.distplot(dataset['atemp'])
plt.show()
#Wind Speed
sns.distplot(dataset['windspeed'])
plt.show()
#Target variable:count of total rental bikes including both casual and registered
sns.distplot(dataset['cnt'])
plt.show()
dataset['dteday'] = pd.to_datetime(dataset['dteday'], format='%d-%m-%Y')
dataset_categorical = dataset.select_dtypes(exclude=['float64','datetime64','int64'])
dataset_categorical.columns
dataset_categorical
plt.figure(figsize=(20,20))
plt.subplot(3,3,1)
sns.boxplot(x='season',y='cnt',data=dataset)
plt.subplot(3,3,2)
sns.boxplot(x='mnth',y='cnt',data=dataset)
plt.subplot(3,3,3)
sns.boxplot(x='weekday',y='cnt',data=dataset)
plt.subplot(3,3,4)
sns.boxplot(x='weathersit',y='cnt',data=dataset)
plt.subplot(3,3,5)
sns.boxplot(x='workingday',y='cnt',data=dataset)
plt.subplot(3,3,6)
sns.boxplot(x='yr',y='cnt',data=dataset)
plt.subplot(3,3,7)
sns.boxplot(x='holiday',y='cnt',data=dataset)
plt.show()
intVarlist=['casual','registered','cnt']
for var in intVarlist:
    dataset[var]=dataset[var].astype('float')
dataset_numeric=dataset.select_dtypes(include=['float64'])
dataset_numeric.head()
sns.pairplot(dataset_numeric)
plt.show()
cor=dataset_numeric.corr()
cor
#beatmap
mask=np.array(cor)
mask[np.tril_indices_from(mask)] =False
fig, ax=plt.subplots()
fig.set_size_inches(10,10)
sns.heatmap(cor,mask=mask,vmax=1,square=True,annot=True)
#Removing atemp as it is highly corealed with temp
dataset.drop('atemp',axis=1,inplace=True)
dataset.head()

##Data -Preparation
dataset_categorical=dataset.select_dtypes(include=['object'])
dataset_categorical.head()
dataset_dummies=pd.get_dummies(dataset_categorical,drop_first=True)
dataset_dummies.head()
#Drop Categorical variable columns
dataset=dataset.drop(list(dataset_categorical.columns),axis=1)
dataset
#Concatenate dummy variables with the dataset
dataset=pd.concat([dataset,dataset_dummies],axis=1)
dataset.head()
dataset=dataset.drop(['instant','dteday'],axis=1,inplace=False)
dataset.head()

#4.Model Building and Evaluation
#Import libs
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
#Split the dataframe into train an test dataset
from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train,df_test= train_test_split(dataset,train_size=0.7,test_size=0.3,random_state=100)
df_train
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
#Apply scalar to all columns except dummy variable
var=["temp","hum","windspeed","casual","registered","cnt"]
df_train[var]=scaler.fit_transform(df_train[var])
df_train.describe()
#checking the corelation coefficient to see which variables are highly corelated
plt.figure(figsize=(40,40))
sns.heatmap(df_train.corr(),annot=True ,cmap="YlGnBu")
plt.show()
#diving into X & Y
y_train=df_train.pop('cnt')
x_train=df_train.drop(['casual','registered'],axis=1)
x_train.head()
np.array(x_train)
import statsmodels.api as sm
x_train_lm=sm.add_constant(x_train)
lr=sm.OLS(y_train,x_train_lm).fit()
lr.params
lm=LinearRegression()
lm.fit(x_train,y_train)
print(lm.coef_)
print(lm.intercept_)
lm.summary()
#import rfe
from sklearn.feature_selection import RFE
lm=LinearRegression()
rfe1=RFE(lm,15)

#fit with 15feature
rfe1.fit(x_train,y_train)
print(rfe1.support_)
print(rfe1.ranking_)

col1=x_train.columns[rfe1.support_]
col1
x_train_rfe1=x_train[col1]
x_train_rf1=sm.add_constant(x_train_rfe1)
lm1=sm.OLS(y_train,x_train_rfe1).fit()
lm1.summary()

from statsmodels.stats.ouliers_influence import variance_inflation_factor
a=x_train_rfe1.drop('const',axis=1)
#Evaluating VIFs
vif=pd.DataFrame()
vif['features']=a.columns
vif['VIF']=[variance_inflation_factor(a.values,i) for i in range(a.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif
lm=LinearRegression()
rfe2=RFE(lm,7)
rfe2.fit(x_train,y_train)
print(rfe2.support_)
print(rfe2.ranking_)

col2=x_train.columns[rfe2.support_]
col2
x_train_rfe2=x_train[col1]
x_train_rfe2=sm.add_constant(x_train_rfe2)
lm2=sm.OLS(y_train,x_train_rfe2).fit()
lm2.summary()

b=x_train_rfe2.drop('const',axis=1)
#Evaluating VIFs
vif1=pd.DataFrame()
vif1['features']=b.columns
vif1['VIF']=[variance_inflation_factor(b.values,i) for i in range(b.shape[1])]
vif1['VIF']=round(vif['VIF'],2)
vif1=vif1.sort_values(by="VIF",ascending=False)
vif1
y_train_cnt=lm2.predict(x_train_rfe2)
fig=plt.figure()
sns.distplot((y_train,y_train_cnt),bins=20)
df_test[var]=scaler.transform(df_test[var])
df_test
y_test=df_test.pop('cnt')
x_test=df_test.drop(['casual','registered'],axis=1)
x_test.head()
c=x_train_rfe2.drop('const',axis=1)
col2=c.columns
x_test_rfe2=x_test[col2]
x_test_rfe2.info()
y_pred=lm2.predict(x_test_rfe2)
plt.figure()
plt.scatter(y_test,y_pred)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

plt.figure(figsize=(8,5))
sns.heatmap(dataset[col2].corr(),cmap="YlGnBu",annot=True)
plt.show()