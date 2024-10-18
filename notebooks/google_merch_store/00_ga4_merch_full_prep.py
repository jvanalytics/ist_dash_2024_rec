import pandas as pd
import numpy as np
import scripts.dslabs_functions
import scripts.config

import scripts.data_functions

filepath=r'data/df_merch_2_encoded.csv'
file_tag = "df_merch_full_prep"



test_data=True
# test_data=False

if test_data==True:

    data=pd.read_csv(filepath)

    # 1% sample
    data=data.sample(frac=0.01, replace=False)
    

else:
    data=pd.read_csv(filepath)
    # 50% sample
    # data=data.sample(frac=0.5, replace=False)

 
target='returning_user'

# ensure sorting by day_of_year for correct splitting 
# "When in the presence of temporal data, data partition shall use older data to train and newer to test, in order to not use future data to classify past data. In any other case, partition shall be random."
data.sort_values(by='day_of_year', inplace=True)



# missing values

data=apply_missing_values_frequent(data)




# outliers

var='engagement_time_msec'
summary5=data[[var]].describe(include="all")

data=drop_outliers(data, summary5, var)


# scaling

data=apply_min_max_scaler(data,target)



# feature engineering



# balancing



print(data.info)