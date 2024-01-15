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
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1199 entries, 0 to 1198
# Data columns (total 29 columns):
#  #   Column                            Non-Null Count  Dtype
# ---  ------                            --------------  -----
#  0   DATE                              1199 non-null   datetime64[ns]
#  1   MONTH                             1199 non-null   int64
#  2   HOUR                              1199 non-null   object
#  3   CITY - VENUE                      1199 non-null   object
#  4   HOME                              1199 non-null   object
#  5   AWAY                              1199 non-null   object
#  6   COMPETITION                       1199 non-null   object
#  7   ROUND                             1198 non-null   object
#  8   SYSTEM TYPE                       1199 non-null   object
#  9   SOFTWARE VERSION                  863 non-null    object
#  10  SYSTEM N°                         1197 non-null   object
#  11  OPERATOR                          1199 non-null   object
#  12  STABILIZATION RESS. NEEDS         1146 non-null   object
#  13  WEATHER CONDITIONS                1199 non-null   object
#  14  STADIUM CONDITIONS                1199 non-null   object
#  15  INTERNET CONDITIONS               1195 non-null   object
#  16  ISSUE TYPE                        1198 non-null   object
#  17  LIVE TRACKING ISSUE TYPE          1189 non-null   object
#  18  LIVE TECH SUPPORT TIME            142 non-null    object
#  19  POST  ISSUE TYPE                  1101 non-null   object
#  20  PLAYER/LINESMAN ID  ACCURACY (%)  1102 non-null   float64
#  21  PLAYER ONLY ID ACCURACY           1102 non-null   float64
#  22  COMPLETENESS(%)                   1199 non-null   float64
#  23  LIVE SCOUT TIME (S)               1185 non-null   float64
#  24  LIVE SCOUTS COUNT                 1189 non-null   float64
#  25  LIVE CONFIRMATIONS (%)            1188 non-null   float64
#  26  TGV AUTO TRAJ COUNTS              1062 non-null   float64
#  27  CONFIRMATIONS                     1151 non-null   object
#  28  NEW_DELTA                         1199 non-null   timedelta64[ns]
# dtypes: datetime64[ns](1), float64(7), int64(1), object(19), timedelta64[ns](1)
# memory usage: 271.8+ KB

#Checking for unique values

sv['DATE'].value_counts()
# 2022-05-14    22
# 2022-03-05    19
# 2021-08-28    17
# 2022-03-12    17
# 2021-11-06    16
#               ..
# 2022-03-01     1
# 2021-12-07     1
# 2021-10-18     1
# 2021-12-08     1
# 2021-07-23     1
# Name: DATE, Length: 215, dtype: int64

sv['MONTH'].value_counts()
# 2     140
# 12    128
# 8     125
# 9     123
# 10    123
# 3     116
# 4     106
# 11    100
# 1      98
# 5      78
# 7      35
# 6      27
# Name: MONTH, dtype: int64

sv['HOUR'].value_counts()
# 19:00    267
# 21:00    227
# 15:00    155
# 20:45    133
# 18:30    101
# 17:00     58
# 13:30     45
# 13:00     37
# 20:00     37
# 18:45     35
# 16:00     35
# 16:15     32
# 17:05      7
# 11:30      5
# 20:30      4
# 21:15      3
# 13:45      3
# 14:00      3
# 10:30      3
# 8:00       2
# 9:30       2
# 18:00      2
# 21:10      1
# 12:00      1
# 7:00       1
# Name: HOUR, dtype: int64

sv['CITY - VENUE'].value_counts()
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

sv['HOME'].value_counts()
# OGC Nice                  30
# AS Monaco                 27
# Club Brugge KV            25
# Royal Antwerp             25
# Olympique de Marseille    23
#                           ..
# Bergerac Perigord FC       1
# FC Barcelona               1
# Italy Women                1
# Sweden Women               1
# Guangzhou City             1
# Name: HOME, Length: 117, dtype: int64

sv['AWAY'].value_counts()
# Club Brugge KV            26
# AS Monaco                 26
# Olympique de Marseille    24
# KRC Genk                  23
# AS Saint-Etienne          22
#                           ..
# Montpellier HSC            1
# Real Madrid CF             1
# Qarabag                    1
# Norway Women               1
# Shenzhen FC                1
# Name: AWAY, Length: 122, dtype: int64

sv['COMPETITION'].value_counts()
# France Ligue 1                  372
# France Ligue 2                  365
# Belgium Jupiler Pro League      313
# China - Chinese Super League     27
# UEFA Champions League            27
# UEFA Europa League               25
# France - Coupe de France         18
# Belgium JPL PlayOff II           10
# Belgium JPL PlayOff I            10
# UEFA Conference League            7
# Belgium Croky Cup                 5
# Friendly game                     3
# Algarve Cup                       2
# France - Barrages L1/L2           2
# UEFA Women Champions League       2
# Netherlands - KNVB Beker          2
# Japan - J-League 1                2
# France - Play Off Ligue 2         2
# France - Barrages L2/Nat          2
# Belgium SuperCup                  1
# Belgium JPL Barrage               1
# Japan - J-League 2                1
# Name: COMPETITION, dtype: int64

sv['ROUND'].value_counts()
# 1                 50
# 3                 47
# 2                 46
# 5                 45
# 4                 41
# 6                 41
# 30                32
# 25                31
# 14                31
# 13                30
# 23                30
# 19                30
# 17                30
# 16                30
# 26                30
# 27                30
# 12                30
# 11                30
# 21                30
# 9                 30
# 28                30
# 10                30
# 29                29
# 24                29
# 31                29
# 18                29
# 20                29
# 15                29
# 22                28
# 8                 28
# 7                 28
# 33                24
# 38                22
# 34                22
# 32                20
# 37                19
# 36                16
# Round of 16       15
# 35                10
# -                 10
# Semi               6
# Round of 64        6
# Quarter            5
# Round of 32        3
# Final              3
# PlayOff            3
# KnockOut Round     2
# Name: ROUND, dtype: int64

sv['SYSTEM TYPE'].value_counts()
# SportVU 2.0    1199
# Name: SYSTEM TYPE, dtype: int64

sv['SYSTEM TYPE'].value_counts()
# SportVU 2.0    1199
# Name: SYSTEM TYPE, dtype: int64

sv['SOFTWARE VERSION'].value_counts()
# 109.21        161
# 2.0.108.6     136
# 111.4         120
# 2.0.107.10     67
# 2.0.108.4      59
# 2.0.109.8      47
# 2.0.108.5      41
# 2.0.109.5      35
# 2.0.105.8      30
# 2.0.109.21     29
# 109.18         25
# 109.8          20
# 112.6          18
#                17
# 1.12.6          9
# 2.0.109.18      9
# 2.0.109.4       8
# 2.0.109.10      8
# 112.3           4
# 112.5           3
# 111.2           3
# 111.1           2
# 2.0.109.3       2
# 111.12          1
# 112.2           1
# 2.0.107.9       1
# 2.0.109.22      1
# 2.0.108.9       1
# 108.6           1
# 2.0.107.11      1
# 2.0.108.3       1
# 2.0.107.8       1
# 2.0.109.12      1
# Name: SOFTWARE VERSION, dtype: int64

sv['SYSTEM N°'].value_counts()
# BELROOT202    92
# BELROOT206    72
# BELROOT207    70
# LFPROOT210    51
# BELROOT204    49
# LFPROOT202    45
# LFPROOT204    45
# LFPROOT211    44
# LFPROOT215    43
# BELROOT203    41
# BELROOT208    40
# LFPROOT208    40
# LFPROOT201    40
# LFPROOT214    38
# LFPROOT219    38
# LFPROOT206    37
# LFPROOT207    37
# LFPROOT216    37
# LFPROOT203    35
# LFPROOT220    35
# LFPROOT209    34
# LFPROOT217    34
# LFPROOT205    33
# LFPROOT212    32
# ROOT601       23
# BELROOT205    20
# LFPROOT218    19
# ROOT602       18
# LFPROOT213    18
# BELROOT209     6
# CSLROOT102     5
# CSLROOT106     4
# CSLROOT103     4
# CSLROOT109     3
# CSLROOT110     3
# CSLROOT107     3
# CSLROOT108     3
# CSLROOT105     2
# PXLC-NICE      1
# lFPROOT209     1
# PXLC0206       1
# PIX-ANDY       1
# Name: SYSTEM N°, dtype: int64

sv['OPERATOR'].value_counts()
# Tom Marant               90
# Thomas Hadjigeorgiou     85
# Hamza Jarbouh            64
# Othman Derdeb            64
# Thom van der Meij        49
# Saad Idrissi-Zouggari    49
# Maxence Beillevaire      40
# JF Carissan              39
# Steeve Carlin            39
# Alex Faubel              39
# Steven Collet            38
# Grégory Poulet           37
# Johann Buis              36
# Edouard Brunel           35
# Maxime Billiau           35
# Ilyass Qrefa             35
# Andria Chiari            35
# Loris Meucci             34
# Valentin Mathieu         34
# Théo Avry                34
# Ali Belgheddar           34
# Romain Huguet            31
# Julien Foubert           29
# Mehdi Bouazzaoui         27
# Quentin Pluquet          24
# Romain Bertrand          21
# Farid Benyahia           21
# Thom van der meij        18
# Erwan Henry              17
# Killian Picaud           11
# Alexandre Maries          8
# Rafik Abbes               7
# Client                    3
# Liu,Lei                   3
# Liu,Yuejin                3
# Cui,Qi                    3
# Wang,Xi                   3
# Tanguy Ngandu             3
# Tian,Mingwei              3
# Liu,Xin                   3
# Xu,Dongming               3
# Chen,Chen                 3
# Liu,Da                    2
# Chems Belgheddar          2
# Thom van der Meij         1
# Andy Calvert              1
# Maxence  Beillevaire      1
# Maxime Billau             1
# Maxime Billiau ?          1
#  Liu,Da                   1
# Name: OPERATOR, dtype: int64

sv['STABILIZATION RESS. NEEDS'].value_counts()
# No     817
# Yes    328
# yes      1
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
# Good           917
# Height <12m    137
# Height <8m      75
# OK              70
# Name: STADIUM CONDITIONS, dtype: int64

sv['INTERNET CONDITIONS'].value_counts()
# Good             675
# Operator 4g      292
# Operator 4G      138
# OK                58
# No internet       17
# Poor Internet      8
# 4g box             6
# Unstable           1
# Name: INTERNET CONDITIONS, dtype: int64

sv['ISSUE TYPE'].value_counts()
# No issue             1015
# Software issue         96
# Operation issue        40
# Camera issue           24
# Calibration issue       9
# Hardware issue          6
# Upload issue            5
# Detection issue         3
# Name: ISSUE TYPE, dtype: int64

sv['LIVE TRACKING ISSUE TYPE'].value_counts()
# No issue              992
# No live delivery      109
# Operation issue        33
# Software issue         25
# Internet issue          7
# Detection issue         4
# Camera issue            4
# Connection issue        3
# software issue          3
# No internet             3
# Weather issue           2
# Calibration issue       1
# Incomplete              1
# Live data impacted      1
# Upload issue            1
# Name: LIVE TRACKING ISSUE TYPE, dtype: int64

sv['LIVE TECH SUPPORT TIME'].value_counts()
# 0 min      102
# <5 min      27
# <15min       8
# <30 min      3
# <1h          1
# 1 min        1
# Name: LIVE TECH SUPPORT TIME, dtype: int64

sv['POST  ISSUE TYPE'].value_counts()
# No issue                 1043
# Software issue             15
# Operation issue            14
# Calibration issue           6
# Camera issue                5
# Data Collection issue       4
# Ops Upload issue            4
# Internal issue              3
# Ops camera issue            2
# Production issue            2
# Upload issue                1
# Late Upload client          1
# Incomplete                  1
# Name: POST  ISSUE TYPE, dtype: int64

sv['PLAYER/LINESMAN ID  ACCURACY (%)'].value_counts()
# 82.760    5
# 86.900    4
# 80.700    3
# 86.030    3
# 89.145    3
#          ..
# 79.650    1
# 83.390    1
# 81.140    1
# 84.460    1
# 82.490    1
# Name: PLAYER/LINESMAN ID  ACCURACY (%), Length: 948, dtype: int64

sv['PLAYER ONLY ID ACCURACY'].value_counts()
# 92.700    4
# 87.300    4
# 93.370    4
# 90.450    3
# 87.470    3
#          ..
# 71.265    1
# 92.965    1
# 80.830    1
# 92.240    1
# 81.630    1
# Name: PLAYER ONLY ID ACCURACY, Length: 915, dtype: int64

sv['COMPLETENESS(%)'].value_counts()
# 97.45    10
# 97.29    10
# 98.05     8
# 97.23     8
# 97.54     8
#          ..
# 95.00     1
# 91.88     1
# 86.72     1
# 93.86     1
# 89.26     1
# Name: COMPLETENESS(%), Length: 476, dtype: int64

sv['LIVE SCOUT TIME (S)'].value_counts()
# 3.36    15
# 3.28    13
# 3.06    13
# 3.70    10
# 3.78    10
#         ..
# 8.66     1
# 6.49     1
# 7.00     1
# 2.31     1
# 7.53     1
# Name: LIVE SCOUT TIME (S), Length: 366, dtype: int64

sv['LIVE SCOUTS COUNT'].value_counts()
# 1229.0    7
# 1169.0    5
# 746.0     5
# 739.0     4
# 1189.0    4
#          ..
# 1273.0    1
# 1088.0    1
# 1116.0    1
# 770.0     1
# 1648.0    1
# Name: LIVE SCOUTS COUNT, Length: 785, dtype: int64

sv['LIVE CONFIRMATIONS (%)'].value_counts()
# 0.00     38
# 60.95     4
# 72.25     3
# 70.19     3
# 71.95     3
#          ..
# 50.88     1
# 70.17     1
# 29.11     1
# 60.28     1
# 48.10     1
# Name: LIVE CONFIRMATIONS (%), Length: 948, dtype: int64

sv['TGV AUTO TRAJ COUNTS'].value_counts()
# 506.0     9
# 401.0     8
# 435.0     7
# 466.0     7
# 459.0     6
#          ..
# 609.0     1
# 247.0     1
# 746.0     1
# 264.0     1
# 1879.0    1
# Name: TGV AUTO TRAJ COUNTS, Length: 498, dtype: int64

sv['CONFIRMATIONS'].value_counts()
# 397     6
# 755     6
# 415     6
# 413     5
# 472     5
#        ..
# 731     1
# 290     1
# 1363    1
# 378     1
# 379     1
# Name: CONFIRMATIONS, Length: 701, dtype: int64

sv['NEW_DELTA'].value_counts()
# 1 days 00:00:00    37
# 0 days 17:48:00     7
# 0 days 17:58:00     6
# 0 days 16:51:00     6
# 0 days 17:20:00     6
#                    ..
# 0 days 15:09:00     1
# 0 days 10:40:00     1
# 0 days 17:29:00     1
# 0 days 16:08:00     1
# 0 days 23:58:00     1
# Name: NEW_DELTA, Length: 634, dtype: int64

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



