import pandas as pd
import datetime
from datetime import datetime
from datetime import timedelta
import math

df=pd.read_csv('/Users/mengwenwu/PycharmProjects/Stage22/SportVu2.0.csv',header=[0])
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
