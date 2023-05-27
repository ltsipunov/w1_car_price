#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import timeit as ti
import datetime as dt
import numpy as np
np.random.seed(4999)


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle


# In[3]:


from catboost import CatBoostRegressor,cv,Pool
from category_encoders.target_encoder import TargetEncoder
from category_encoders.cat_boost import CatBoostEncoder


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# ####  Load and purge data

# In[ ]:


X = ['year','make','model','abbr_trim','body','transmission','state','condition','odometer','abbr_seller']
y = ['sellingprice']
cat_cols = ['make','model','abbr_trim','body','transmission','state','abbr_seller'] 


# In[ ]:


def fillna(df):
    df[['make','model','trim','body']] = df[['make','model','trim','body']].fillna('UNKNOWN')
    df[['color','interior']] = df[['color','interior']].fillna('—')
    df['transmission'] = df['transmission'].fillna('automatic')
    cond_mean = df.groupby('year').condition.mean()
    idx_na= df.condition.isna()
    df.loc[idx_na,'condition'] = df[idx_na].year.apply(lambda s: cond_mean[s])    
    df.loc[df.condition.isna(),'condition'] = df.condition.mean()
  
    run_mean = df.groupby('year').odometer.mean()
    idx_na=df.odometer.isna() 
    df.loc[idx_na,'odometer'] = df[idx_na].year.apply(lambda s: run_mean[s])
    df.loc[df.odometer.isna(),'odometer'] = df.odometer.mean()
    return(df)


# In[6]:


def normalize(df,rounding):
    start = ti.default_timer()
    cols_to_upper= ['make','model','trim','body']
    cols_dt = ['year','month','day','hour','minute','second','weekday','yearday','dl']
    cols_abbr = ['trim','seller']
    cols_trash = ['saledate','trim','abbr_seller','seller','second','yearday','dl']
    def transform_row(r,cols_to_upper,cols_dt,cols_abbr):
        def abbr(s,prefix_size=5 ):
            s = s.strip().upper()
            if len(s) <= prefix_size:
                return(s)
            s = s[:prefix_size].replace(' ','-')+s[prefix_size:]
            i = s.find(' ')
            if i > 0:
                s = s[:i]        
            return(s)

        t =  dt.datetime.strptime(r['saledate'].split('GMT')[0]  ,"%a %b %d %Y %H:%M:%S ").timetuple()
        dc = dict(zip(cols_dt ,t))
        for col in cols_abbr:
            dc['abbr_'+col] = abbr(r[col])

        for col in cols_to_upper:
            dc[col] = str(r[col]).upper()
 
        dc['odometer']= round(r['odometer']/rounding['odometer'])
        dc['condition']= round(r['condition'],rounding['condition'])
        return dc

    transformed = df.apply(transform_row, axis=1,result_type='expand',
                                         cols_to_upper=cols_to_upper,cols_dt=cols_dt,cols_abbr=cols_abbr)
    df[transformed.columns] = transformed   
    
#    print(f"normalize: {ti.default_timer()-start} s " )
    return df


# In[8]:


def skew(df,threshold,mult):
    df0 = df[df[y]>threshold]
    df1 = df[ds[y]<threshold]
    if mult >=1:
        df = pd.concat( [df0]+mult*[df1] ,axis=0).copy()
    else:
        idx = df0.sample(frac=mult, replace=True).index
        df = pd.concat( [df0[idx]]+[df1] ,axis=0).copy()
    df =  shuffle(df)
    return df


# In[9]:


def encode_transform(df,enc ):
    start= ti.default_timer()
    df_enc = pd.DataFrame( enc.transform(df[cat_cols]), columns = enc.get_feature_names_out() )
    df = df.drop(cat_cols,axis=1)
    df = pd.concat( [df,df_enc] , axis=1)

#    print(f"encode: {ti.default_timer()-start} s " )
    return df
