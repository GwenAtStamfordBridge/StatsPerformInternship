import datetime
from datetime import datetime
from datetime import timedelta
import math
import matplotlib as mpl
# %matplotlib inline
import numpy
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')



rdf=pd.read_csv('/Users/mengwenwu/PycharmProjects/Stage22/EDA/df_for_heatmap.csv')
df=pd.read_csv('/Users/mengwenwu/PycharmProjects/Stage22/SportVu2.0.csv')

df.head()
df.isnull()
df.isnull().sum()
df.info()


df['FULL DATE']=pd.to_datetime(df['FULL DATE'])

DATE_ONLY=pd.DatetimeIndex(df['DATE']).day
df['DATE']=DATE_ONLY

MONTH_ONLY=pd.DatetimeIndex(df['MONTH']).month
df['MONTH']=MONTH_ONLY

HOUR_ONLY=df['FULL DATE'].dt.time
df['HOUR']=HOUR_ONLY

df['REPORT DELIVERY TO CLIENT FULL(CET)']=pd.to_datetime(df['REPORT DELIVERY TO CLIENT FULL(CET)'])
df['END OF GAME TIME FULL']=pd.to_datetime(df['END OF GAME TIME FULL'])

df['NEW_DELTA']=df['REPORT DELIVERY TO CLIENT FULL(CET)']-df['END OF GAME TIME FULL']

df['CONFIRMATIONS']=df['CONFIRMATIONS'].astype('int64')

df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1199 entries, 0 to 1198
# Data columns (total 32 columns):
#  #   Column                               Non-Null Count  Dtype
# ---  ------                               --------------  -----
#  0   DATE                                 1199 non-null   int64
#  1   REPORT DELIVERY DATE                 1199 non-null   object
#  2   MONTH                                1199 non-null   int64
#  3   HOUR                                 1199 non-null   object
#  4   FULL DATE                            1199 non-null   datetime64[ns]
#  5   CITY - VENUE                         1199 non-null   object
#  6   HOME                                 1199 non-null   object
#  7   AWAY                                 1199 non-null   object
#  8   COMPETITION                          1199 non-null   object
#  9   ROUND                                1198 non-null   object
#  10  SYSTEM TYPE                          1199 non-null   object
#  11  SYSTEM N°                            1197 non-null   object
#  12  OPERATOR                             1199 non-null   object
#  13  STABILIZATION RESS. NEEDS            1146 non-null   object
#  14  WEATHER CONDITIONS                   1199 non-null   object
#  15  STADIUM CONDITIONS                   1199 non-null   object
#  16  INTERNET CONDITIONS                  1195 non-null   object
#  17  ISSUE TYPE                           1198 non-null   object
#  18  LIVE TRACKING ISSUE TYPE             1189 non-null   object
#  19  POST  ISSUE TYPE                     1101 non-null   object
#  20  DELTA END OF GAME/CLIENT DELIVERY    1195 non-null   object
#  21  COMPLETENESS(%)                      1199 non-null   float64
#  22  LIVE SCOUT TIME (S)                  1185 non-null   float64
#  23  LIVE SCOUTS COUNT                    1189 non-null   float64
#  24  LIVE CONFIRMATIONS (%)               1188 non-null   float64
#  25  TGV AUTO TRAJ COUNTS                 1062 non-null   float64
#  26  CONFIRMATIONS                        1151 non-null   object
#  27  END OF GAME TIME (CET)               1199 non-null   object
#  28  END OF GAME TIME FULL                1199 non-null   datetime64[ns]
#  29  REPORT DELIVERY TO CLIENT (CET)      1162 non-null   object
#  30  REPORT DELIVERY TO CLIENT FULL(CET)  1199 non-null   datetime64[ns]
#  31  NEW_DELTA                            1199 non-null   timedelta64[ns]
# dtypes: datetime64[ns](3), float64(5), int64(2), object(21), timedelta64[ns](1)
# memory usage: 299.9+ KB

df.describe()
#               DATE  ...                  NEW_DELTA
# count  1199.000000  ...                       1199
# mean     15.768140  ...  0 days 17:18:44.737281067
# std       8.454609  ...  0 days 04:54:47.515764895
# min       1.000000  ...            0 days 01:10:00
# 25%       8.000000  ...            0 days 14:53:30
# 50%      16.000000  ...            0 days 18:04:00
# 75%      22.000000  ...            0 days 20:31:00
# max      31.000000  ...            1 days 07:02:00
# [8 rows x 8 columns]


#Examining numerical features
numeric_features = df.select_dtypes(include=[np.number])
numeric_features.columns
# Index(['DATE', 'MONTH', 'COMPLETENESS(%)', 'LIVE SCOUT TIME (S)',
#        'LIVE SCOUTS COUNT', 'LIVE CONFIRMATIONS (%)', 'TGV AUTO TRAJ COUNTS',
#        'NEW_DELTA'],
#       dtype='object'

#Examining categorical features
categorical_features = df.select_dtypes(include=[np.object])
categorical_features.columns
# Index(['REPORT DELIVERY DATE', 'HOUR', 'CITY - VENUE', 'HOME', 'AWAY',
#        'COMPETITION', 'ROUND', 'SYSTEM TYPE', 'SYSTEM N°', 'OPERATOR',
#        'STABILIZATION RESS. NEEDS', 'WEATHER CONDITIONS', 'STADIUM CONDITIONS',
#        'INTERNET CONDITIONS', 'ISSUE TYPE', 'LIVE TRACKING ISSUE TYPE',
#        'POST  ISSUE TYPE', 'DELTA END OF GAME/CLIENT DELIVERY',
#        'CONFIRMATIONS', 'END OF GAME TIME (CET)',
#        'REPORT DELIVERY TO CLIENT (CET)'],
#       dtype='object')

#Visualising missing values for a sample of 250
msno.matrix(df.sample(250))
<AxesSubplot:>

#Heatmap
# The missingno correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another:
#Because of the way we cleaned the data the heat map will not be that interesting, but we can also take a look at the heatmap for the uncleaned data

msno.heatmap(df)
msno.heatmap(rdf)

# Dendrogram
# The dendrogram allows you to
# more fully correlate variable completion, revealing trends deeper than the pairwise ones visible in the correlation heatmap
msno.dendrogram(rdf)
msno.dendrogram(df)

#Skewness and kurtosis
df.skew()
# COMPLETENESS(%)          -3.153189
# LIVE SCOUT TIME (S)       1.754807
# LIVE SCOUTS COUNT         0.613201
# LIVE CONFIRMATIONS (%)   -2.412783
# TGV AUTO TRAJ COUNTS      2.755585
# dtype: float64

rdf.skew()
# PLAYER/LINESMAN ID  ACCURACY (%)    -1.349128
# PLAYER ONLY ID ACCURACY             -2.055505
# LIVE SCOUTS COUNT                    0.611782
# LIVE CONFIRMATIONS (%)              -2.413721
# TGV AUTO TRAJ COUNTS                 2.748385
# dtype: float64

df.kurt()
# COMPLETENESS(%)           13.280751
# LIVE SCOUT TIME (S)        7.857440
# LIVE SCOUTS COUNT          0.404366
# LIVE CONFIRMATIONS (%)     9.111705
# TGV AUTO TRAJ COUNTS      12.591303
# dtype: float64

rdf.kurt()
# PLAYER/LINESMAN ID  ACCURACY (%)      3.388728
# PLAYER ONLY ID ACCURACY               6.592900
# LIVE SCOUTS COUNT                     0.403820
# LIVE CONFIRMATIONS (%)                9.121618
# TGV AUTO TRAJ COUNTS                 12.052650
# dtype: float64

#Now we will try to see the distribution of the data with seaborn
y = df['COMPLETENESS(%)']
plt.figure(1); plt.title('COMPLETENESS(%)')
sns.distplot(y, kde=True)
#We notice that the distribution is not normal so we need to make some transformations

S=[]
for i in range(1199):
    a=df['NEW_DELTA'].loc[i].total_seconds()
    S.append(a)
print(S)
S=pd.Series(S)
df['NEW_DELTA(s)']=S
#S is the time delta in seconds

#Scatter plots for Numerical Variables
# df.plot.scatter(x = 'DATE', y = 'COMPLETENESS(%)')
# df.plot.scatter(x = 'DATE', y = 'COMPLETENESS(%)')
# <AxesSubplot:xlabel='DATE', ylabel='COMPLETENESS(%)'>
# df.plot.scatter(x = 'MONTH', y = 'COMPLETENESS(%)')
# <AxesSubplot:xlabel='MONTH', ylabel='COMPLETENESS(%)'>
# df.plot.scatter(x = 'HOUR', y = 'COMPLETENESS(%)')
# df.plot.scatter(x = 'LIVE SCOUT TIME (S)', y = 'COMPLETENESS(%)')
# <AxesSubplot:xlabel='LIVE SCOUT TIME (S)', ylabel='COMPLETENESS(%)'>
# df.plot.scatter(x = 'LIVE SCOUTS COUNT', y = 'COMPLETENESS(%)')
# <AxesSubplot:xlabel='LIVE SCOUTS COUNT', ylabel='COMPLETENESS(%)'>
# df.plot.scatter(x = 'LIVE CONFIRMATIONS (%)', y = 'COMPLETENESS(%)')
# <AxesSubplot:xlabel='LIVE CONFIRMATIONS (%)', ylabel='COMPLETENESS(%)'>
# df.plot.scatter(x = 'TGV AUTO TRAJ COUNTS', y = 'COMPLETENESS(%)')
# <AxesSubplot:xlabel='TGV AUTO TRAJ COUNTS', ylabel='COMPLETENESS(%)'>
# df.plot.scatter(x = 'NEW_DELTA(s)', y = 'COMPLETENESS(%)')
# <AxesSubplot:xlabel='NEW_DELTA(s)', ylabel='COMPLETENESS(%)'>

#Box plots for Categorical data

# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# fig.axis(ymin=75, ymax=100);

# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# fig.axis(ymin=75, ymax=100);
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(12, 9))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# fig.axis(ymin=75, ymax=100);
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(10, 9))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# fig.axis(ymin=75, ymax=100);
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(10, 10))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# fig.axis(ymin=75, ymax=100);
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(5, 5))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# fig.axis(ymin=75, ymax=100);
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(5, 6))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# fig.axis(ymin=75, ymax=100);
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(12, 9))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# fig.axis(ymin=75, ymax=100);
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# fig.axis(ymin=75, ymax=100);
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# fig.axis(ymin=75, ymax=100);
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# plt.setp(labels, rotation=45)
# fig.axis(ymin=75, ymax=100);
# Traceback (most recent call last):
#   File "/Users/mengwenwu/.conda/envs/Stage22/lib/python3.10/code.py", line 90, in runcode
#     exec(code, self.locals)
#   File "<input>", line 5, in <module>
# NameError: name 'labels' is not defined
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=75, ymax=100);
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=75, ymax=100);
# var = 'WEATHER CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=75, ymax=100);
# var = 'CITY - VENUE'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=75, ymax=100);
# var = 'CITY - VENUE'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=75, ymax=100);
# var = 'CITY - VENUE'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=75, ymax=100);
# var = 'CITY - VENUE'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(17, 9))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=75, ymax=100);
# var = 'CITY - VENUE'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(17, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=75, ymax=100);
# var = 'CITY - VENUE'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(17, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=77, ymax=100);
# df['COMPLETENESS(%)'].min()
# 80.56
# var = 'CITY - VENUE'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(17, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'HOME'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(17, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'AWAY'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(17, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'COMPETITION'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=80, ymax=100);
# var = 'ROUND'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=80, ymax=100);
# var = 'ROUND'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'SYSTEM TYPE'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'SYSTEM N°'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'OPERATOR'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'OPERATOR'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(17, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'OPERATOR'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'STABILIZATION RESS. NEEDS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'STABILIZATION RESS. NEEDS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'STABILIZATION RESS. NEEDS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'STADIUM CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# fig.axis(ymin=80, ymax=100);
# var = 'STADIUM CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=80, ymax=100);
# var = 'INTERNET CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=80, ymax=100);
# var = 'INTERNET CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=80, ymax=100);
# var = 'STADIUM CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=80, ymax=100);
# var = 'INTERNET CONDITIONS'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=80, ymax=100);
# var = 'ISSUE TYPE'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=80, ymax=100);
# var = 'LIVE TRACKING ISSUE TYPE'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=80, ymax=100);
# var = 'POST  ISSUE TYPE'
# data = pd.concat([df['COMPLETENESS(%)'], df[var]], axis=1)
# f, ax = plt.subplots(figsize=(15, 8))
# fig = sns.boxplot(x=var, y="COMPLETENESS(%)", data=data)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=45)
# fig.axis(ymin=80, ymax=100);

#Now let's make a correlation matrix to have a more objectiv view of our analysis

corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, vmin=-1, annot=True, square=True, cmap='BrBG');

plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df.corr()[['COMPLETENESS(%)']].sort_values(by='COMPLETENESS(%)', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with COMPLETENESS(%)', fontdict={'fontsize':18}, pad=16);

#Pairplots

sns.set()
cols = ['HOUR', 'LIVE CONFIRMATIONS (%)', 'LIVE SCOUTS COUNT', 'LIVE SCOUT TIME (S)', 'NEW_DELTA(s)','TGV AUTO TRAJ COUNTS']
fig=sns.pairplot(df[cols], size = 2.5)
fig.savefig("pairplot.png")
plt.show();

#Now we need to deal with ouliers, we will pick them out as they might have an impact on our model later on

COMPLETENESS_scaled = StandardScaler().fit_transform(df['COMPLETENESS(%)'][:,np.newaxis]);
low_range = COMPLETENESS_scaled[COMPLETENESS_scaled[:,0].argsort()][:10]
high_range= COMPLETENESS_scaled[COMPLETENESS_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
outer range (low) of the distribution:
# [[-7.08354777]
#  [-6.7848835 ]
#  [-6.02262096]
#  [-5.97358653]
#  [-5.6303455 ]
#  [-5.60805713]
#  [-5.56793804]
#  [-5.51890361]
#  [-5.42083475]
#  [-5.26927378]]
# outer range (high) of the distribution:
# [[1.10520242]
#  [1.10520242]
#  [1.1096601 ]
#  [1.12303312]
#  [1.12303312]
#  [1.16760988]
#  [1.18544058]
#  [1.19435593]
#  [1.19881361]
# [1.38157831]]

#We have found our outliers, we will leave them be for now, but we will need to be careful about [1.38157831] and [-7.08354777]

#We will now check for normality with distribution curves and qq plots


#Completeness

def t1(val): #sqrt()
    return(np.sqrt(100-val))
T1=np.frompyfunc(t1,1,1)
print(T1(df['COMPLETENESS(%)']))

CompSqrt=pd.Series(T1(df['COMPLETENESS(%)']))

def t2(val): #inv()
    return(1/(100-val))
T2=np.frompyfunc(t2,1,1)
print(T2(df['COMPLETENESS(%)']))

CompInv=pd.Series(T2(df['COMPLETENESS(%)']))

def t3(val): #log()
    return(np.log(100-val))
T3=np.frompyfunc(t3,1,1)
print(T3(df['NEW_DELTA(s)']))

CompLog=pd.Series(T3(df['COMPLETENESS(%)']))
print(CompLog)


#For COMPLETENESS(%) we will apply a log(100-x) transformation
sns.distplot(CompLog, fit=norm);

#For Live Scout Time we will need to use a log(x) transformation
sns.distplot(np.log(df['LIVE SCOUT TIME (S)']), fit=norm);
fig = plt.figure()
res = stats.probplot(np.log(df['LIVE SCOUT TIME (S)']), plot=plt)

#For TGV we will need to used a log(x) transformation
sns.distplot(np.log(df['TGV AUTO TRAJ COUNTS']), fit=norm);
fig = plt.figure()
res = stats.probplot(np.log(df['TGV AUTO TRAJ COUNTS']), plot=plt)