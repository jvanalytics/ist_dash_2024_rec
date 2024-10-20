import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os

# Append the path of the scripts folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))


import dslabs_functions
import config
import data_functions



# Append the path of the data folder to Python path
filepath = os.path.join(os.path.dirname(__file__), 'data', 'df_merch_2_encoded.csv')
# filepath=r'df_merch_2_encoded.csv'

file_tag = "df_merch_full_prep"



# test_data=True
test_data=False

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

print(f"Encoded Data Load completed at: {datetime.now()}")


# --------- missing values

data=data_functions.apply_missing_values_frequent(data)

# ----------------- outliers
# no improvement?

var='engagement_time_msec'
summary5=data[[var]].describe(include="all")
data=data_functions.drop_outliers(data, summary5, var)


# ------------ scaling

data=data_functions.apply_min_max_scaler(data,target)


# ---------------- feature engineering

# data=data_functions.apply_remove_low_variance_variables(data,max_threshold=0.08, target=target, min_features_to_keep=3, exclude=['day_of_year'])



print(f"Data Prep completed at: {datetime.now()}")
print(data.head())

# evaluation prep

# Call the function to split the data
y, X, labels, class_counts = data_functions.define_target_and_prepare_data(data, target=target)


train, test = data_functions.split_data_save_csv(pd.DataFrame(X, columns=data.columns), y, data_columns=data.columns, target_column=target)


# balancing training after splitting
train=data_functions.apply_balanced_smote(train)


# NB+KNN evaluate

# Print current date and time
print(f"Current time before final NB/KNN evaluation: {datetime.now()}")

eval_metric='f2'
eval_final: dict[str, list] = dslabs_functions.evaluate_approach(train, test, target='returning_user', metric=eval_metric)


print(f'final evaluation: {eval_final}')
print(f"Current time after final NB/KNN evaluation: {datetime.now()}")
