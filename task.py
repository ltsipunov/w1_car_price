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
        self.params['handler'] = type(self).__name__
        self.log = ""        
        tstamp = dt.datetime.strftime( dt.datetime.now(), '%y%m%d%H%M')
        filename = filename=f"logs/{self.params['handler']}_{tstamp}.log"
        logging.basicConfig( filename=filename,
                format='%(asctime)s - %(levelname)s:  %(message)s', 
                level=logging.DEBUG )
        logging.info(f"init params: {self.params} ")
        if 'slice' in self.params:
            self.set_slice(self.params['slice'])
            
# ---- set scopes of columns for features and target used in utils module for the instance  
    def set_slice(self,slice):
        global X,y,cat_cols
        X = slice['X'] if 'X' in slice else X
        X = slice['y'] if 'y' in slice else y
        X = slice['cat_cols'] if 'cat_cols' in slice else cat_cols


# ----- define class and parameters for encoder and regressor used for the instance 
    def new_encoder(self):
        enc_params =  { 'handle_unknown':'value','cols':cat_cols }        
        encoder_class= (self.params['encoder_class']) if ('encoder_class' in self.params) else TargetEncoder  
        if 'encoder' in self.params:
            enc_params.update( self.params['encoder'])    
        self.encoder = encoder_class( **enc_params )
        return self.encoder

    def new_model(self):
        self.model = (self.params['model_class']) if ('model_class' in self.params) else RandomForestRegressor
        return self.model
    
# --- abstract class for a main proceudre  , must be overridden
    def process(self):
        return self