# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 09:40:28 2022

@author: Featherine
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as md

df = pd.read_csv('features - Final.csv')
df = df.fillna(0)

# df = df[0:48]

df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')

fig, axs = plt.subplots(4, 2, figsize=(15,12))

# A random day
df_day = df[0:48]
axs[0, 0].plot('DateTime', '1006', data=df_day)
axs[0, 0].set_xlim(df_day['DateTime'].min()-pd.Timedelta(1,'h'),
                   df_day['DateTime'].max()+pd.Timedelta(1,'h'))
axs[0, 0].xaxis.set_major_locator(md.HourLocator(interval = 1))
axs[0, 0].xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()
axs[0, 0].set_title('Traffic in a Random Day')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Number of Cars')

axs[0, 1].plot('DateTime', 'Temp', data=df_day)
axs[0, 1].set_xlim(df_day['DateTime'].min()-pd.Timedelta(1,'h'),
                   df_day['DateTime'].max()+pd.Timedelta(1,'h'))
axs[0, 1].xaxis.set_major_locator(md.HourLocator(interval = 1))
axs[0, 1].xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()
axs[0, 1].set_title('Temperature in a Random Day')
axs[0, 1].set_ylabel('Temperature')



# Per over a year
axs[1, 0].plot('DateTime', '1006', data=df)
axs[1, 0].set_xlim(df['DateTime'].min()-pd.Timedelta(1,'h'),
                   df['DateTime'].max()+pd.Timedelta(1,'h'))
axs[1, 0].xaxis.set_major_locator(md.HourLocator(interval = 1))
axs[1, 0].xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()
axs[1, 0].set_title('Traffic in a Year')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Number of Cars')

axs[1, 1].plot('DateTime', 'Temp', data=df)
axs[1, 1].set_xlim(df['DateTime'].min()-pd.Timedelta(1,'h'),
                   df['DateTime'].max()+pd.Timedelta(1,'h'))
axs[1, 1].xaxis.set_major_locator(md.HourLocator(interval = 1))
axs[1, 1].xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()
axs[1, 1].set_title('Temperature in a Year')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Temperature')


# Get average per hour
df['hour'] = df['DateTime'].dt.hour

df_day = df.copy()

df_day[['1006', 'Temp']] = df_day[['1006', 'Temp']].groupby(df_day['hour']).transform('mean')

# Plot average per hour
df_day = df_day[0:48]

axs[2, 0].plot('DateTime', '1006', data=df_day)
axs[2, 0].set_xlim(df_day['DateTime'].min()-pd.Timedelta(1,'h'),
                   df_day['DateTime'].max()+pd.Timedelta(1,'h'))
axs[2, 0].xaxis.set_major_locator(md.HourLocator(interval = 1))
axs[2, 0].xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()
axs[2, 0].set_title('Average Traffic in a Day')
axs[2, 0].set_xlabel('Time')
axs[2, 0].set_ylabel('Number of Cars')

axs[2, 1].plot('DateTime', 'Temp', data=df_day)
axs[2, 1].set_xlim(df_day['DateTime'].min()-pd.Timedelta(1,'h'),
                   df_day['DateTime'].max()+pd.Timedelta(1,'h'))
axs[2, 1].xaxis.set_major_locator(md.HourLocator(interval = 1))
axs[2, 1].xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()
axs[2, 1].set_title('Average Temperature in a Day')
axs[2, 1].set_ylabel('Temperature')



# Get std per hour
df['hour'] = df['DateTime'].dt.hour
df_day = df.copy()
df_day = df_day.groupby('hour').agg(np.std, ddof=0)
df_day.reset_index(inplace=True)
df_day = df_day.drop('DateTime', axis=1)

axs[3, 0].plot('hour', '1006', data=df_day)
axs[3, 0].set_title('std Traffic in a Day')
axs[3, 0].set_xlabel('Time')
axs[3, 0].set_ylabel('Number of Cars')

axs[3, 1].plot('hour', 'Temp', data=df_day)
axs[3, 1].set_title('std Temperature in a Day')
axs[3, 1].set_xlabel('Time')
axs[3, 1].set_ylabel('Temperature')

plt.show()  

plt.savefig('visualization.png')

# Seasonality - Periodic
# Stationarity - constant mean and variance