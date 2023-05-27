#!/usr/bin/env python
# coding: utf-8

# In[1]:

from dataset import * 

# In[2]:


# --- base class for all tasks : validation, cross-validation, submission

class Task:

    def __init__(self, dfs,params = {}):
        self.params= params
        params['handler'] = type(self)
        self.log = ""
        logging.info(f"init params: {self.params} ")

    def new_encoder(self):
        self.encoder = TargetEncoder(handle_unknown='value')
        return self.encoder

    def new_model(self):
        self.model = RandomForestRegressor
        return self.model

    def process(self):
        return self