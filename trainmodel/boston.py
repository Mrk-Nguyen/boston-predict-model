# %%
from IPython import get_ipython

# %%
import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from lightgbm import LGBMRegressor

from basic_models import MeanRegressor, RandomRegressor

import pickle
import json

import matplotlib.pyplot as plt
import seaborn as sns

from collections.abc import Iterable

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
TARGETCOL = 'MEDVALUE'

# %%
def mean_average_percentage_error(y_true, y_pred, epsilon=0.01):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(abs(y_true - y_pred) / (y_true + epsilon))

def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_average_percentage_error(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


# %%
data_load = load_boston()
data_df = pd.DataFrame(data_load.get('data'),columns=data_load.get('feature_names'))

# %%
# Check for multicollinearity

fig,ax = plt.subplots(figsize=(12,8))
sns.heatmap(data_df.corr(), ax=ax, annot=True,fmt='0.2f',linewidths=0.5,cmap="RdBu_r")
fig.show()

# %% [markdown]
# ## Remove `RAD` column
# 

# %%
data_df = data_df.drop(columns=['RAD'])

# %%
# Look at correlation against target for each feature
target_df = pd.DataFrame([data_load.get('target') for col in data_df.columns]).T
target_df.columns = data_df.columns

data_df.corrwith(target_df,axis=0)

# %%
# Add target column to Dataframe
data_df[TARGETCOL] = data_load.get('target')

# %% [markdown]
# Assume outliers have been removed already. Ready to move on to model building
# %% [markdown]
# # Build potential models to predict Boston House Prices

# %%
# Split data into training and holdout 
train_df, holdout_df = train_test_split(data_df, test_size=0.20, shuffle=True, random_state=20)
train_df.reset_index(drop=True, inplace=True)
holdout_df.reset_index(drop=True, inplace=True)

print(train_df.shape[0],holdout_df.shape[0])


# %%
# Split for X and Y data
X_train_df = train_df.drop(columns=[TARGETCOL])
Y_train_df = train_df[TARGETCOL]

X_holdout_df = holdout_df.drop(columns=[TARGETCOL])
Y_holdout_df = holdout_df[TARGETCOL]


# %%
# Export exact order of features so web app can order the columns correctly for model scoring
with open('../model/feature_sequence.txt', 'w') as f:
    f.write(json.dumps(X_train_df.columns.tolist()))


# %%
def results(modelname: str, model: object , 
            X_train: Iterable ,Y_train: Iterable,
            X_holdout: Iterable,Y_holdout: Iterable,
            results_train_df: pd.DataFrame,results_holdout_df: pd.DataFrame) -> None:
    '''Add RMSE,MAE,MAPE results of given model to results dataframes

    Args:
        modelname (str): Name of the model to be added to results dataframes as an index
        X_train (Iterable): Input matrix from training data
        Y_train (Iterable): Target column from training data
        X_holdout (Iterable): Input matrix from holdout data
        Y_holdout (Iterable): Target column from holdout data
        results_train_df (pd.DataFrame): Trainng results dataframe with columns: RMSE,MAE,MAPE
        results_holdout_df (pd.DataFrame): Holdout results dataframe with columns: RMSE,MAE,MAPE

    Return:
        None
    '''

    #TODO
    pass
    

# %%
results_train_df = pd.DataFrame(columns=["RMSE", "MAE", "MAPE"])
results_holdout_df = pd.DataFrame(columns=["RMSE", "MAE", "MAPE"])


# %%
# Mean Model
mean_mdl = MeanRegressor()
mean_mdl.fit(X_train_df, Y_train_df)

results('Mean (Baseline)',mean_mdl,X_train_df,Y_train_df,X_holdout_df,Y_holdout_df,results_train_df,results_holdout_df)

print(results_train_df)
print(results_holdout_df)


# %%
# Random Model

random_mdl = RandomRegressor()
random_mdl.fit(X_train_df, Y_train_df)

results('Random',random_mdl,X_train_df,Y_train_df,X_holdout_df,Y_holdout_df,results_train_df,results_holdout_df)

print(results_train_df)
print(results_holdout_df)


# %%
# Linear Regression
linear_mdl = Pipeline([
    ("Scaler", StandardScaler()),
    ("Model", LinearRegression())
])

linear_mdl.fit(X_train_df, Y_train_df)

results('Linear Regression',linear_mdl,X_train_df,Y_train_df,X_holdout_df,Y_holdout_df,results_train_df,results_holdout_df)

print(results_train_df)
print(results_holdout_df)


# %%
# LGBMRegressor
lgbm_mdl = LGBMRegressor()

lgbm_mdl.fit(X_train_df, Y_train_df)

results('LightGBM',lgbm_mdl,X_train_df,Y_train_df,X_holdout_df,Y_holdout_df,results_train_df,results_holdout_df)

print(results_train_df)
print(results_holdout_df)

# %%
# Chosen model --------------------------------------------
# TODO
model = object

# Export chosen model to webapp




# Export data for endpoint testing
