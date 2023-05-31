#!/usr/bin/env python
# coding: utf-8

# In[1]:

import logging
#import datetime as dt
#tstamp = dt.datetime.strftime( dt.datetime.now(), '%y%m%d%H%M')
#logging.basicConfig(format='%(asctime)s - %(levelname)s:  %(message)s', level=logging.DEBUG)

from utils import * #X,y,cat_cols,fillna,normalize,skew,encode

# In[2]:
#----- base class for typical teaching process with  2 dataframes , contains train and test frames 
#----- dataframes used as is, with standard names for columns slices X ( list of features columns ) and y( target column )
#----- these names and elementary operations with dataframes defined in utils.py file
#----- two small subclass extensions ScoreSet and SubmissionSet defined for estimation and submission tasks

class DataSet:
    def __init__(self,tr,te,params={}):
        self.tr= tr
        self.te= te
        self.log = ""
        self.params=params

    
    def set_encoder(self,encoder):
        self.encoder = encoder
        return(self)
    def set_model(self,model_class):
        self.model= model_class(**self.params['model'])
        return(self)
    
    def prepare(self):  
        global X,cat_cols

        logging.debug(f"prepare: started ")
        self.tr = fillna(self.tr)
        self.te = fillna(self.te)
        logging.debug(f"prepare: fillna finished ")
        self.tr = normalize(self.tr,self.params['rounding'])
        self.te = normalize(self.te,self.params['rounding'])
        logging.debug(f"prepare: normalize finished")
        if 'skew' in self.params:
            th = self.params['skew']['threshold']
            mult = self.params['skew']['mult']
            self.tr =  skew(self.tr,th,mult )
            logging.debug(f"prepare: skew finished")
        if 'sellers' in self.params:
            self.tr =  classify_sellers(self.tr,self.params['sellers'] )
            self.te =  classify_sellers(self.te,self.params['sellers'] )
            X = X + ['q25','q75'] 
            
            logging.debug(f"prepare: classify_sellers finished")
        self.encode()
        logging.debug(f"prepare: finished")
        return(self)
        
    def encode(self):
        self.encoder.fit(self.tr[cat_cols],self.tr[y])
        logging.debug('encode: fitted ')
        self.tr=encode_transform(self.tr,self.encoder)
        self.te=encode_transform(self.te,self.encoder)
        logging.debug('encode: test transformed ')
        return(self)
        
    def fit(self):
        global X,cat_cols
        self.model.fit(self.tr[X],self.tr[y])     
        logging.debug('model: learned')
        return(self)     

    def fix_result(self):
        self.log= '' 
    
    def predict(self):
        self.pr =  self.model.predict(self.te[X])
        logging.debug('model: predicted')
        self.fix_result()
        return(self)


# In[3]:


# ------------ class to use in validations, estimating a result of the pass and  fixing the estimation 
class ScoreSet(DataSet):

    def fix_result(self):
        self.score = mape(self.pr,self.te[y])
        logging.info(f"model: MAPE= {self.scores()} ")

    def scores(self):
        return self.score

# In[4]:

        
# ------------ class to use in the final run , writing a result to upload 
class SubmissionSet(DataSet):

    def fix_result(self):
        prv = pd.DataFrame({'vin': self.te['vin'],'sellingprice':self.pr} )
        prv.to_csv('datasets/result.csv',index=False)
        logging.debug('model: prediction saved')
