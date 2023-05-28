# w1_car_price
 Kaggle car price competition

This project predicts car selling price on standard Kaggle dataset.

This release uses only RandomForestRegressor model with Target encoder,
 but may be easily expanded to other models and encoders 
 that planned for future releases 

The project contains 3 notebooks for parameters picking and estimation: 
- validation - for primary evaluation 
- cross_validation - for cross validation
- submit - for providing finally results

All notebooks based on common library of three modules:
- utils.py - basic operations with dataframes
- dataset.py - operations on "train-test" pair of dataframes (class DataSet)
- task.py - base class for estimating/submitting process  

Every testrun writes result to separate logfile with timestamp in the name,
 these logs stored in "log" (git-ignored ) catalog,
 best results may be transferred to "selected" (git-supported ) catalog

In the research, next approaches were tried:
- tuning hyperparameters of models
- reduction of cardinality both in categorical and quntitative features 
- upscale records of cheap sales  (some effect with OrdinalEncoder  but useless with TargetEncoder) 

Best Kaggle score achieved on submission is 15.75,
 best score confirmed by crosvalidation is 17.1

Next steps planned in future:
- log parser notebook  to get easily estimations from logs
- separate notebook for feature analysis 
- more algorithms of  data preparation
- CatBoost model research ( probed ,but results were disappointing )