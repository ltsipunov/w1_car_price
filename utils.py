#!/usr/bin/env python
# coding: utf-8

# this module defines low-level procedures for dataset used in this task,
# uses specific of the task and data

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

# --- the global varibales must be set from calling process before run
X = ['year','make','model','trim','body','transmission','state','condition','odometer','seller']
cat_cols = ['make','model','trim','body','transmission','state','seller'] 
y = ['sellingprice']


# In[ ]:

# 
def fillna(df):
    df[['make','model','trim','body']] = df[['make','model','trim','body']].fillna('UNKNOWN')
    df[['color','interior']] = df[['color','interior']].fillna('—')       # as the synbol already used in this columns for unknown  
    df['transmission'] = df['transmission'].fillna('automatic')           # this feature has very little effect so used simple fill  
    
# -- for numeric columns, unknowns filled as average by year    
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
    prefix_size = rounding['prefix_size'] if ('prefix_size' in rounding) else 5 

    cols_to_upper= ['make','model','trim','body']                                     # columns to convert to standard uppercase
    cols_dt = ['yr','month','day','hour','minute','second','weekday','yearday','dl']  # columns to unpack selling time
    cols_abbr = ['trim','seller']                                                     # columns to reduce of cardinality    

# ------------ general procedure to transform each record     
    def transform_row(r,cols_to_upper,cols_dt,cols_abbr):              
        def abbr(s,prefix_size ):                          # spaces in prefix must be trimmed, then only first word left 
            s = s.strip().upper()
            if len(s) <= prefix_size:
                return(s)
            s = s[:prefix_size].replace(' ','-')+s[prefix_size:]
            i = s.find(' ')
            if i > 0:
                s = s[:i]        
            return(s)
        
    # selling time converted to tuple , however these fields have little effect, so they dropped in late versions
        t =  dt.datetime.strptime(r['saledate'].split('GMT')[0]  ,"%a %b %d %Y %H:%M:%S ").timetuple()
        
        dc = dict(zip(cols_dt ,t))
        for col in cols_abbr:
            dc[col] = abbr(r[col],prefix_size)

        for col in cols_to_upper:
            dc[col] = str(r[col]).upper()
    # to round numeric fileds to reduce cardinality (optionally possible coercion to string by setting cat_cols in params) 
        dc['odometer']= round(r['odometer']/rounding['odometer'])
        dc['condition']= round(r['condition'],rounding['condition'])
        for c in cat_cols:
            if(type(r[c]) != str):
                r[c] = str(r[c])
        return dc

    transformed = df.apply(transform_row, axis=1,result_type='expand',
                                         cols_to_upper=cols_to_upper,cols_dt=cols_dt,cols_abbr=cols_abbr)
    df[transformed.columns] = transformed   
    
    return df


# In[8]:


def skew(df,threshold,mult):                             # class to up/down-scale dataset by target variable
    df0 = df[df[y[0]]>threshold]
    df1 = df[df[y[0]]<threshold]
    
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

    return df


# In[10]:
def classify_sellers(ds,params):       
    # I tried to create additional feature to estimate tendency of a seller to buy trash cars - by .25 quantile of condition
    # but the idea proved useless with target encoder
    stages = params['stages']
    abbr_count = ds['seller'].value_counts()

    def abbr_power(s):
        for i in range(stages):
            if 2**i > abbr_count[s]:
                return('ABBR_T_'+str(i) )
        return(s)

    ds['seller']= ds['seller'].apply(abbr_power)

    g_abbr = ds.groupby('seller').condition
    q_abbr = pd.DataFrame( { 'q25': g_abbr.quantile(.25),'q75':g_abbr.quantile(.75)} )
    #q_abbr.loc['159191']
    ds[['q25','q75']] = ds.seller.apply(lambda s: q_abbr.loc[s] )

    return ds
