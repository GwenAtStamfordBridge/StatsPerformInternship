import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from datetime import timedelta
import math

df=pd.read_csv('/Users/mengwenwu/PycharmProjects/Stage22/SportVu2.0.csv',header=[0])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df['DATE']=pd.to_datetime(df['DATE'])
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
df['NEW_DELTA']=pd.Series(S)
df['HOUR']=pd.to_datetime(df['HOUR']).dt.time
sv=df[['DATE', 'MONTH', 'HOUR', 'CITY - VENUE', 'HOME', 'AWAY', 'COMPETITION',
       'ROUND', 'SYSTEM TYPE', 'SOFTWARE VERSION', 'SYSTEM N°', 'OPERATOR',
       'STABILIZATION RESS. NEEDS', 'WEATHER CONDITIONS', 'STADIUM CONDITIONS',
       'INTERNET CONDITIONS', 'ISSUE TYPE', 'LIVE TRACKING ISSUE TYPE',
       'LIVE TECH SUPPORT TIME', 'POST  ISSUE TYPE','PLAYER/LINESMAN ID  ACCURACY (%)',
       'PLAYER ONLY ID ACCURACY', 'COMPLETENESS(%)', 'LIVE SCOUT TIME (S)',
       'LIVE SCOUTS COUNT', 'LIVE CONFIRMATIONS (%)', 'TGV AUTO TRAJ COUNTS',
       'CONFIRMATIONS', 'NEW_DELTA']]

#from now on we will work with sv as our data frame
#Univariate Stats
sv.describe()
#              MONTH  ...                  NEW_DELTA
# count  1199.000000  ...                       1199
# mean      6.605505  ...  0 days 17:54:45.287739783
# std       3.712813  ...  0 days 04:25:40.627017549
# min       1.000000  ...            0 days 00:00:00
# 25%       3.000000  ...            0 days 15:30:00
# 50%       7.000000  ...            0 days 18:23:00
# 75%      10.000000  ...            0 days 20:52:30
# max      12.000000  ...            1 days 07:02:00
# [8 rows x 9 columns]
sv.shape
#(1199, 29)
sv.info()
sv['DATE'].value_counts()
sv['MONTH'].value_counts()
sv['HOUR'].value_counts()
sv['CITY - VENUE'].value_counts()
sv['HOME'].value_counts()
sv['AWAY'].value_counts()
sv['COMPETITION'].value_counts()
sv['ROUND'].value_counts()
sv['SYSTEM TYPE'].value_counts()
# SportVU 2.0    1199
# Name: SYSTEM TYPE, dtype: int64

sv['SYSTEM TYPE'].value_counts()
# SportVU 2.0    1199
# Name: SYSTEM TYPE, dtype: int64

sv['SOFTWARE VERSION'].value_counts()
sv['SYSTEM N°'].value_counts()
sv['OPERATOR'].value_counts()
sv['STABILIZATION RESS. NEEDS'].value_counts()
# Name: STABILIZATION RESS. NEEDS, dtype: int64
sv['WEATHER CONDITIONS'].value_counts()
# Good            805
# Sun/Shadow      279
# OK               64
# Fog/Mist         11
# Sun/shadow       11
# Smoke             7
# Heavy rain        5
# Fog               3
# Fireworks         3
# Raining           3
# Snow              2
# Mist/Fog          2
# Light change      1
# Heavy Fog         1
# Heavy snow        1
# Heavy fog         1
# Name: WEATHER CONDITIONS, dtype: int64
sv['STADIUM CONDITIONS'].value_counts()
sv['INTERNET CONDITIONS'].value_counts()
sv['ISSUE TYPE'].value_counts()
sv['LIVE TRACKING ISSUE TYPE'].value_counts()
sv['LIVE TECH SUPPORT TIME'].value_counts()
sv['POST  ISSUE TYPE'].value_counts()
sv['PLAYER/LINESMAN ID  ACCURACY (%)'].value_counts()
sv['PLAYER ONLY ID ACCURACY'].value_counts()
sv['COMPLETENESS(%)'].value_counts()
sv['LIVE SCOUT TIME (S)'].value_counts()
sv['LIVE SCOUTS COUNT'].value_counts()
sv['LIVE CONFIRMATIONS (%)'].value_counts()
sv['TGV AUTO TRAJ COUNTS'].value_counts()
sv['CONFIRMATIONS'].value_counts()
sv['NEW_DELTA'].value_counts()

print(f'DATE: {pd.api.types.is_numeric_dtype(sv.DATE)}')
print(f'MONTH: {pd.api.types.is_numeric_dtype(sv.MONTH)}')
print(f'HOUR: {pd.api.types.is_numeric_dtype(sv.HOUR)}')
# print(f'CITY - VENUE: {pd.api.types.is_numeric_dtype(sv.CITY - VENUE)}')
print(f'HOME: {pd.api.types.is_numeric_dtype(sv.HOME)}')
print(f'AWAY: {pd.api.types.is_numeric_dtype(sv.AWAY)}')
# print(f'COMPETITION: {pd.api.types.is_numeric_dtype(sv.COMPETITION)}')
print(f'ROUND: {pd.api.types.is_numeric_dtype(sv.ROUND)}')
# print(f'SOFTWARE VERSION: {pd.api.types.is_numeric_dtype(sv.SOFTWARE VERSION)}')
# print(f'SYSTEM N°: {pd.api.types.is_numeric_dtype(sv.SYSTEM N°)}')
print(f'OPERATOR: {pd.api.types.is_numeric_dtype(sv.OPERATOR)}')
# print(f'STABILIZATION RESS. NEEDS: {pd.api.types.is_numeric_dtype(sv.STABILIZATION RESS. NEEDS)}')
# print(f'WEATHER CONDITIONS: {pd.api.types.is_numeric_dtype(sv.WEATHER CONDITIONS)}')
# print(f'STADIUM CONDITIONS: {pd.api.types.is_numeric_dtype(sv.STADIUM CONDITIONS)}')
# print(f'INTERNET CONDITIONS: {pd.api.types.is_numeric_dtype(sv.INTERNET CONDITIONS)}')
# print(f'ISSUE TYPE: {pd.api.types.is_numeric_dtype(sv.ISSUE TYPE)}')
# print(f'LIVE TRACKING ISSUE TYPE: {pd.api.types.is_numeric_dtype(sv.LIVE TRACKING ISSUE TYPE)}')
# print(f'LIVE TECH SUPPORT TIME: {pd.api.types.is_numeric_dtype(sv.LIVE TECH SUPPORT TIME)}')
# print(f'POST  ISSUE TYPE: {pd.api.types.is_numeric_dtype(sv.POST  ISSUE TYPE)}')
# print(f'PLAYER/LINESMAN ID  ACCURACY (%): {pd.api.types.is_numeric_dtype(sv.PLAYER/LINESMAN ID  ACCURACY (%))}')
# print(f'PLAYER ONLY ID ACCURACY: {pd.api.types.is_numeric_dtype(sv.PLAYER ONLY ID ACCURACY)}')
# print(f'COMPLETENESS(%): {pd.api.types.is_numeric_dtype(sv.COMPLETENESS(%))}')
# print(f'LIVE SCOUT TIME (S): {pd.api.types.is_numeric_dtype(sv.LIVE SCOUT TIME (S))}')
# print(f'LIVE SCOUTS COUNT: {pd.api.types.is_numeric_dtype(sv.LIVE SCOUTS COUNT)}')
# print(f'LIVE CONFIRMATIONS (%): {pd.api.types.is_numeric_dtype(sv.LIVE CONFIRMATIONS (%))}')
# print(f'TGV AUTO TRAJ COUNTS: {pd.api.types.is_numeric_dtype(sv.TGV AUTO TRAJ COUNTS)}')
print(f'CONFIRMATIONS: {pd.api.types.is_numeric_dtype(sv.CONFIRMATIONS)}')
print(f'NEW_DELTA: {pd.api.types.is_numeric_dtype(sv.NEW_DELTA)}')

#More stats info
LND=(sv.NEW_DELTA .count(),
          sv.NEW_DELTA .min(),
          sv.NEW_DELTA .max(),
          sv.NEW_DELTA.quantile(.25),
          sv.NEW_DELTA .quantile(.50),
          sv.NEW_DELTA .quantile(.75),
          sv.NEW_DELTA .mean(),
          sv.NEW_DELTA .median(),
          sv.NEW_DELTA .mode()
      )
for i in range(8):
    print(LND[i],end='\n')

LCONF=(sv.NEW_DELTA .count(),
          sv.CONFIRMATIONS .min(),
          sv.CONFIRMATIONS .max(),
          sv.CONFIRMATIONS.quantile(.25),
          sv.CONFIRMATIONS .quantile(.50),
          sv.CONFIRMATIONS .quantile(.75),
          sv.CONFIRMATIONS .mean(),
          sv.CONFIRMATIONS .median(),
          sv.CONFIRMATIONS .mode()
      )
for i in range(8):
    print(LCONF[i],end='\n')



