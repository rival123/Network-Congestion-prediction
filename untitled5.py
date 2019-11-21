# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 02:56:42 2019

@author: ASUS
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns
%matplotlib inline

df_X = pd.read_csv("train_upd.csv")
df_X = df_X.drop(columns = ['Unnamed: 39','Unnamed: 40'])
df_X1 = df_X.drop(columns = ['par_year', 'par_month', 'par_hour','par_min','cell_range','tilt','ran_vendor','Congestion_Type'])
df_X1 = df_X1.drop(columns=[' Cell_name'])
cols = ['4G_rat', 'par_day', 'subscriber_count', 'web_browsing_total_bytes',
       'video_total_bytes', 'social_ntwrking_bytes',
       'cloud_computing_total_bytes', 'web_security_total_bytes',
       'gaming_total_bytes', 'health_total_bytes', 'communication_total_bytes',
       'file_sharing_total_bytes', 'remote_access_total_bytes',
       'photo_sharing_total_bytes', 'software_dwnld_total_bytes',
       'marketplace_total_bytes', 'storage_services_total_bytes',
       'audio_total_bytes', 'location_services_total_bytes',
       'presence_total_bytes', 'advertisement_total_bytes',
       'system_total_bytes', 'voip_total_bytes', 'speedtest_total_bytes',
       'email_total_bytes', 'weather_total_bytes', 'media_total_bytes',
       'mms_total_bytes', 'others_total_bytes', 'beam_direction',
       'Total_bytes', '1/cell_range_square', 'cell_range_square',
       'inner_radius', 'subscriber/r2', 'Total_bytes/r2', 'rin-rout',
       'subscriber/(rin-rout)^2', '(rin-rout)^2', 'total_byte/(rin-rout)^2']
subset_df = wines[cols]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=cols)
final_df = pd.concat([scaled_df, wines['wine_type']], axis=1)
final_df.head()

# plot parallel coordinates
from pandas.plotting import parallel_coordinates
pc = parallel_coordinates(final_df, 'wine_type', color=('#FFE888', '#FF9999'))