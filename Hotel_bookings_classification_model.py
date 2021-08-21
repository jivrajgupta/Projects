#!/usr/bin/env python
# coding: utf-8

# In[344]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',200)


# In[345]:


df = pd.read_csv(r"C:\Users\jivraj\Desktop\hotel_bookings.csv")


# In[346]:


df.head(5)


# In[347]:


df.tail(5)


# In[348]:


df.shape


# In[349]:


sns.histplot(data = df, x = 'arrival_date_year')


# In[350]:


df.info()


# In[351]:


df.isna()


# In[352]:


df.isna().sum()


# In[353]:


mode_country = df['country'].mode()
mode_country


# In[354]:


mode_agent = df['agent'].mode()
mode_agent


# In[355]:


mode_children = df['children'].mode()
mode_children


# In[356]:


df['country'].fillna(value = 'PRT', inplace = True)


# In[357]:


df['agent'].fillna(value = 9.0, inplace = True)


# In[358]:


df['children'].fillna(value = 0.0, inplace = True)


# In[359]:


df.isna().sum()


# In[360]:


df.drop('company', axis = 1, inplace = True)


# In[361]:


df.drop('required_car_parking_spaces', inplace = True, axis = 1)


# In[362]:


cols = df.columns
def unique_values(col):
    print('name of the column {}'.format(col))
    print(df[col].value_counts().sort_index())
    print('\n')
for col in cols:
    unique_values(col)


# In[363]:


for i in range(0,119390):
    if df['stays_in_weekend_nights'].values[i] > 4:
        df['stays_in_weekend_nights'].values[i] = 5


# In[364]:


for i in range(0,119390):
    if df['market_segment'].values[i] == 'Aviation' or 'Complementary' or 'Undefined':
        df['market_segment'].values[i] = 'others'


# In[365]:


for i in range(0,119390):
    if df['stays_in_week_nights'].values[i] > 7:
        df['stays_in_week_nights'].values[i] = 7


# In[366]:


for i in range(0,119390):
    if df['adults'].values[i] > 3:
        df['adults'].values[i] = 3


# In[367]:


for i in range(0,119390):
    if df['children'].values[i] > 4:
        df['children'].values[i] = 4


# In[368]:


for i in range(0,119390):
    if df['babies'].values[i] > 0:
        df['babies'].values[i] = 1


# In[369]:


for i in range(0,119390):
    if df['previous_cancellations'].values[i] >= 2:
        df['previous_cancellations'].values[i] = 2


# In[370]:


df.drop('total_of_special_requests', axis = 1 , inplace = True)


# In[371]:


df.drop('days_in_waiting_list', axis = 1, inplace = True)


# In[372]:


for i in range(0,119390):
    if df['booking_changes'].values[i] > 3:
        df['booking_changes'].values[i] = 3


# In[373]:


for i in range(0,119390):
    if df['previous_bookings_not_canceled'].values[i] > 6:
        df['previous_bookings_not_canceled'].values[i] = 7


# In[374]:


print("unique values for adults is ", df['adults'].unique())
print("unique values for children is ",df['children'].unique())
print("unique values for babies is ",df['babies'].unique())


# In[375]:


df.isna().sum()


# In[376]:


filter = (df['adults']==0) & (df['children']==0) & (df['babies']==0) 
filter.shape


# In[377]:


df[filter] 


# In[378]:


df['agent'] = df['agent'].astype(int)


# In[379]:


df.drop('agent', axis = 1, inplace = True)


# In[380]:


df[~filter]   


# In[381]:


data = df[~filter]   


# In[382]:


#from where do the guests come from ?


# In[383]:


country_data = data[data['is_canceled']==0]['country'].value_counts().reset_index()


# In[384]:


country_data.columns = ['Name_of_country', 'no_of_people_from_that_country']


# In[385]:


country_data


# In[386]:


get_ipython().system('pip install folium ')


# In[387]:


import folium 
from folium.plugins import HeatMap
basemap = folium.Map
get_ipython().system('pip install plotly')
import plotly.express as px
plt.figure(figsize = (10,32))
px.choropleth(country_data, 
             locations = 'Name_of_country',
             color = 'no_of_people_from_that_country' )


# In[388]:


#how much do the guests pay for a night 


# In[389]:


data2 = data[data['is_canceled']==0]
data2.columns


# In[390]:


plt.figure(figsize=(12,8))
sns.boxplot(data = data2, x = 'reserved_room_type', y = 'adr', hue = 'hotel')
plt.title('price per person, hotel types')
plt.xlabel('room type')
plt.ylabel('rate')


# In[391]:


#how does price per night vary over the years 


# In[392]:


data_resort = data[(data['hotel']=='Resort Hotel') & (data['is_canceled']==0)]
data_city = data[(data['hotel']=='City Hotel') & (data['is_canceled']==0)]


# In[393]:


data_resort.head()


# In[394]:


data_city


# In[395]:


resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
resort_city = data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()


# In[396]:


resort_hotel


# In[397]:


resort_city


# In[398]:


final = resort_hotel.merge(resort_city, on ='arrival_date_month')


# In[399]:


final.columns = ['arrival month', 'mean price for resort', 'mean price for city']


# In[400]:


final 


# In[401]:


get_ipython().system('pip install sort-dataframeby-monthorweek')


# In[402]:


get_ipython().system('pip install sorted-months-weekdays')


# In[403]:


import sort_dataframeby_monthorweek as sd
final = sd.Sort_Dataframeby_Month(final, 'arrival month')


# In[404]:


final 


# In[405]:


plt.figure(figsize=(12,12))
sns.lineplot(data = final, y = 'mean price for resort', x = 'arrival month' )
sns.lineplot(data = final, y = 'mean price for city', x = 'arrival month' )


# In[406]:


px.line(final, x = 'arrival month', y =['mean price for resort','mean price for city'], title = 'room price over the months')


# In[407]:


#which are the most busiest months or in which were the guests high?


# In[408]:


rush_resort = data_resort['arrival_date_month'].value_counts().reset_index()
rush_resort.columns = ['month', 'no of guests resort']


# In[409]:


rush_city = data_city['arrival_date_month'].value_counts().reset_index()
rush_city.columns = ['month', 'no of guests city']


# In[410]:


rush_resort


# In[411]:


rush_city


# In[412]:


final2 = rush_resort.merge(rush_city, on = 'month')


# In[413]:


final2


# In[414]:


final2 = sd.Sort_Dataframeby_Month(final2, 'month')


# In[415]:


px.line(final2, x = 'month', y = ['no of guests resort','no of guests city'])


# In[416]:


plt.figure(figsize=(24,12))
sns.lineplot(data = final2, y = 'no of guests resort', x = 'month' )
sns.lineplot(data = final2, y = 'no of guests city', x = 'month' )


# In[417]:


plt.figure(figsize=(12,12))
sns.heatmap(data.corr().abs(), annot = True)


# In[418]:


correlation = data.corr()['is_canceled']


# In[419]:


correlation


# In[420]:


correlation.abs().sort_values(ascending=False)


# In[421]:


data.groupby('is_canceled')['reservation_status'].value_counts()


# In[422]:


data.info()


# In[423]:


list_not = ['days_in_waiting_list','arrival_date_year']


# In[424]:


num_features = []
for col in data.columns:
    if data[col].dtype!='object' and col not in list_not:
        num_features.append(col)


# In[425]:


num_features


# In[426]:


data[num_features].head(5)


# In[427]:


data.columns


# In[428]:


cat_not = ['arrival_date_year','assigned_room_type','booking_changes','reservation_status','country','days_in_waiting_list']


# In[429]:


cat_features = []
for col in data.columns:
    if data[col].dtype=='object' and col not in cat_not:
        cat_features.append(col)


# In[430]:


cat_features


# In[431]:


data_cat = data[cat_features]


# In[432]:


data_cat.head(5)


# In[433]:


data_cat['reservation_status_date'] = pd.to_datetime(data_cat['reservation_status_date'])


# In[434]:


data_cat


# In[435]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[436]:


data_cat['year'] = data_cat['reservation_status_date'].dt.year


# In[437]:


data_cat['month'] = data_cat['reservation_status_date'].dt.month


# In[438]:


data_cat['day'] = data_cat['reservation_status_date'].dt.day


# In[439]:


data_cat.head(5)


# In[440]:


data_cat.drop('reservation_status_date', inplace = True, axis = 1)


# In[441]:


data_cat


# In[442]:


data_cat['cancellation'] = data['is_canceled']


# In[443]:


data_cat.head()


# In[444]:


cols = data_cat.columns[0:8]


# In[445]:


cols


# In[446]:


for col in cols:
    print(data.groupby('is_canceled')[col].value_counts())


# In[447]:


for col in cols:
    print(data_cat.groupby([col])['cancellation'].mean().to_dict())
    print('\n')


# In[448]:


for col in cols:
    dict = data_cat.groupby([col])['cancellation'].mean().to_dict()
    data_cat[col] = data_cat[col].map(dict)


# In[449]:


data_cat.head(5)


# In[450]:


dataframe =  pd.concat([data_cat,data[num_features]], axis = 1)


# In[451]:


dataframe


# In[452]:


dataframe.drop('cancellation',axis = 1,  inplace = True)


# In[453]:


dataframe


# In[454]:


sns.distplot(dataframe['lead_time'])


# In[455]:


import numpy as np
def handle_outlier(col):
    dataframe[col] = np.log1p(dataframe[col])


# In[456]:


handle_outlier('lead_time')


# In[457]:


sns.distplot(dataframe['lead_time'])


# In[458]:


sns.distplot(dataframe['adr'])


# In[459]:


handle_outlier('adr')


# In[460]:


sns.distplot(dataframe['adr'].dropna())


# In[461]:


dataframe.isnull().sum()


# In[462]:


dataframe.dropna(inplace = True)


# In[463]:


y1 = dataframe['is_canceled']
x1 = dataframe.drop(['is_canceled'], axis = 1)


# In[464]:


x1


# In[465]:


y1


# In[466]:


dataframe.info()


# In[467]:


cols = x1.columns 


# In[468]:


x1


# In[469]:


x1


# In[470]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.25, random_state = 0, stratify = y1 )


# In[471]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[472]:


logreg.fit(x_train,y_train)


# In[473]:


y_pred = logreg.predict(x_test)


# In[474]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[475]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[476]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(logreg, x1, y1, cv = 10)


# In[477]:


score.mean()


# In[478]:


#applying multiple algorithms and check the accuracy 


# In[479]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[480]:


models = []
models.append(('Model name : LogisticRegression',LogisticRegression()))
models.append(('Model name : Naive bayes', GaussianNB()))
models.append(('Model name : KNN', KNeighborsClassifier()))
models.append(('Model name : random forest', RandomForestClassifier()))
models.append(('Model name : decision tree', DecisionTreeClassifier()))


# In[481]:


for name,model in models:
    print(name)
    print('\n')
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(confusion_matrix(y_test, predictions))
    print('\n')
    
    print ('accuracy score', accuracy_score(y_test, predictions))
    print('\n')
    score = cross_val_score(model, x1, y1, cv = 10)
    print('mean 10 cross validation accuracy', score.mean())
    print('\n')
    from sklearn.metrics import classification_report
    print('classification report {}'.format(classification_report(y_test,predictions)))
    print('\n')


# In[ ]:





# In[ ]:





# In[ ]:




