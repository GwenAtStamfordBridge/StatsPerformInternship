import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from datetime import timedelta
import math
from datetime import datetime

#To run selected code Alt+Shift+E
#To Block comment cmd+/

df=pd.read_csv('/Users/mengwenwu/PycharmProjects/Stage22/SportVu2.0.csv',header=[0])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.head()
df.size
df.shape
df.info()

#Handling missing values
#Checking for missing values
pd.isna().sum()

# DATE                                    0
# MONTH                                   0
# HOUR                                    0
# CITY - VENUE                            0
# HOME                                    0
# AWAY                                    0
# COMPETITION                             0
# ROUND                                   1
# SYSTEM TYPE                             0
# SOFTWARE VERSION                      336
# SYSTEM N°                               2
# OPERATOR                                0
# STABILIZATION RESS. NEEDS              53
# WEATHER CONDITIONS                      0
# STADIUM CONDITIONS                      0
# INTERNET CONDITIONS                     4
# ISSUE TYPE                              1
# LIVE TRACKING ISSUE TYPE               10
# LIVE TECH SUPPORT TIME               1057
# POST  ISSUE TYPE                       98
# DELTA END OF GAME/CLIENT DELIVERY       4
# PLAYER/LINESMAN ID  ACCURACY (%)       97
# PLAYER ONLY ID ACCURACY                97
# COMPLETENESS(%)                         0
# LIVE SCOUT TIME (S)                    14
# LIVE SCOUTS COUNT                      10
# LIVE CONFIRMATIONS (%)                 11
# TGV AUTO TRAJ COUNTS                  137
# CONFIRMATIONS                          48
# END OF GAME TIME (CET)                  0
# REPORT DELIVERY TO CLIENT (CET)        36
# PLAYERS  TRACKING TIME                127
# BALL TRACKING TIME                    153
# dtype: int64


#To get an idea of the umerical values
df.describe()

      # PLAYER/LINESMAN ID  ACCURACY (%)  ...  TGV AUTO TRAJ COUNTS
# count                       1102.000000  ...           1062.000000
# mean                          80.685476  ...            537.728814
# std                            8.354036  ...            215.363043
# min                           33.540000  ...            231.000000
# 25%                           76.801250  ...            407.000000
# 50%                           82.002500  ...            490.000000
# 75%                           86.157500  ...            603.750000
# max                           96.320000  ...           2299.000000
# [8 rows x 7 columns]

#Outliers check
df.hist(bins=10,figsize=(20,10))
low_comp=df[df['COMPLETENESS(%)']<94]
low_comp[['DATE','COMPLETENESS(%)']].plot(kind='scatter',x='DATE',y='COMPLETENESS(%)',figsize=(20,10))

#Categorical values check
#Venues for example

# df['CITY - VENUE'].value_counts()
# Brugge - Jan BreidelStadion                     44
# Nice - Allianz Riviera                          28
# Monaco - Stade Louis II                         27
# Anvers - Bosuilstadion                          24
# Marseille - Stade Vélodrome                     24
#                                                 ..
# Wolfsburg - Volkswagen Arena                     1
# Graz - Stadion Graz Liebenau                     1
# Lyon - Stade de Balmont                          1
# Saint-Ouen - Stade Bauer                         1
# Haikou - Mission Hills football base Stadium     1
# Name: CITY - VENUE, Length: 116, dtype: int64
# pd.DataFrame(df['CITY - VENUE'].value_counts()).plot(kind='bar',figsize=(20,10))

#Looking at the possible values for Weather conditions

# Good            805
# Sun/Shadow      279
# OK               64
# Sun/shadow       11
# Fog/Mist         11
# Smoke             7
# Heavy rain        5
# Fireworks         3
# Fog               3
# Mist/Fog          2
# Snow              2
# Raining           2
# Light change      1
# Heavy Fog         1
# Heavy snow        1
# Heavy fog         1
# Rainy             1
# Name: WEATHER CONDITIONS, dtype: int64

#We will want to put rainy and raining together

#Dealing with dates and times
#Checking the types of our variables
# df.loc[0,'DATE']
# '7/17/21'
#By trying to run a date/time method on it we can know if it actually a date time type
#And we get df.loc[0,'DATE'].day_name()
# Traceback (most recent call last):
#   File "/Users/mengwenwu/.conda/envs/Stage22/lib/python3.10/code.py", line 90, in runcode
#     exec(code, self.locals)
#   File "<input>", line 1, in <module>
# AttributeError: 'str' object has no attribute 'day_name'
#Because we actually read it as a string

#We will convert the column into a date with pandas method
df['DATE']
pd.to_datetime(df['DATE'])
# 0      2021-07-17
# 1      2021-07-17
# 2      2021-07-17
# 3      2021-07-23
# 4      2021-07-24
#           ...
# 1194   2022-06-12
# 1195   2022-06-12
# 1196   2022-06-12
# 1197   2022-06-13
# 1198   2022-06-13
# Name: DATE, Length: 1199, dtype: datetime64[ns]

#OK

#We will replace the month by having the month and not year and month
MONTH_ONLY=pd.DatetimeIndex(df['MONTH']).month
df['MONTH']=MONTH_ONLY

#We will convert end of game time and client delivery time to contain the full date to have less issue when calculating deltas
df['END OF GAME TIME (CET)']=pd.to_datetime(df['DATE'].astype(str) +" "+ df["END OF GAME TIME (CET)"].astype(str))


df['DATE_DELIVERY']=df['DATE'].astype(str) +" "+ df["REPORT DELIVERY TO CLIENT (CET)"].astype(str)
#Let's create a new Column called DELIVERY_CLIENT
L=[]
for i in range(1199):
    if df['DATE_DELIVERY'].loc[i].__contains__('nan'):
        L.append(df['END OF GAME TIME (CET)'].loc[i])
        print(L)
    else:
        L.append(df['DATE_DELIVERY'].loc[i])
print(L)
df['DELIVERY_CLIENT']=pd.Series(L)
df['DELIVERY_CLIENT'] = df['DELIVERY_CLIENT'].astype('datetime64[ns]')
df['DELIVERY_CLIENT']=df['DELIVERY_CLIENT'] + timedelta(days=1)

#df['DELIVERY_CLIENT'] is the time where the client receives the repport for the games where we have those values, if we dont have the values it's just the end of the game time plus 24h

#Let's compute a Delta
S=[]
for i in range(1199):
    if df['DELIVERY_CLIENT'].loc[i] == df['END OF GAME TIME (CET)'].loc[0]+timedelta(days=1):
        S.append('0 days 00:00:00')
    else:
        S.append(df['DELIVERY_CLIENT'].loc[i]-df['END OF GAME TIME (CET)'].loc[i])
print(S)
pd.Series(S)
df['NEW_DELTA']=pd.Series(S,dtype='timedelta64[ns]')

#Using format codes we will convert: LIVE TECH SUPPORT TIME, PLAYERS  TRACKING TIME, BALL TRACKING TIME to pandas time

pd.to_datetime(df['BALL TRACKING TIME'].loc[1194], format='%M:%S:00')
pd.to_datetime(df['BALL TRACKING TIME'].loc[1194], format='%M:%S:00').time()
df['BALL TRACKING TIME'] = df['BALL TRACKING TIME'].fillna('00:00:00')
from datetime import datetime

time_obj = datetime.strptime('00:00:00', '%H:%M:%S')
print ("The type of the time is now",  type(time_obj))
print ("The date is", time_obj)
print(time_obj.time())

datetime.strptime('00:00:00', '%H:%M:%S').time()

L1=[]
for i in range(1198):
    if math.isnan(df['BALL TRACKING TIME'].loc[i]):
        L1.append(datetime.strptime('00:00:00', '%H:%M:%S').time())
    else:
        L1.append(pd.to_datetime(df['BALL TRACKING TIME'].loc[i], format='%M:%S:00').time())
print(L1)

