#!/usr/bin/env python
# coding: utf-8

# In[1]:

from dataset import * 
import logging
import datetime as dt


# In[2]:


# --- base class for all tasks : validation, cross-validation, submission

class Task:

    def __init__(self, dfs,params = {}):
        self.params= params
        params['handler'] = type(self).__name__
        self.log = ""
        tstamp = dt.datetime.strftime( dt.datetime.now(), '%y%m%d%H%M')
        logging.basicConfig( filename=f"logs/{self.params['handler']}_{tstamp}.log",
                format='%(asctime)s - %(levelname)s:  %(message)s', 
                level=logging.DEBUG )
        logging.info(f"init params: {self.params} ")

    def new_encoder(self):
        self.encoder = TargetEncoder(handle_unknown='value',cols = cat_cols )
        return self.encoder

    def new_model(self):
        self.model = RandomForestRegressor
        return self.model

    def process(self):
        return self