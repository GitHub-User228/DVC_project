import json
import math
import os
import pickle
import sys

import pandas as pd
from sklearn.metrics import *
from dvclive import Live
import matplotlib.pyplot as plt

EVAL_PATH = "eval"

model_file = sys.argv[1]
train_file = os.path.join(sys.argv[2], "train.csv")
test_file = os.path.join(sys.argv[2], "val.csv")
TARGET = "median_house_value"

def evaluate(model, data, split):
    FEATURES = [col for col in data.columns if col != TARGET]
    X, y = data[FEATURES].values, data[TARGET].values
    y_pred = model.predict(X)
      
    # Use dvclive to log a few simple metrics...
    
    mse = mean_squared_error(y, y_pred)
    rmse = mse**(0.5)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    if not live.summary:
        live.summary = {"MSE": {}, "RMSE": {}, "MAE": {}, "R2": {}}
    live.summary["MSE"][split] = mse
    live.summary["RMSE"][split] = rmse 
    live.summary["MAE"][split] = mae 
    live.summary["R2"][split] = r2


# Load model and data.
with open(model_file, "rb") as fd:
    model = pickle.load(fd)

train = pd.read_csv(train_file, index_col=0)
test = pd.read_csv(test_file, index_col=0)

# Evaluate train and test datasets.
live = Live(os.path.join(EVAL_PATH, "live"), dvcyaml=False)
evaluate(model, train, "train")
evaluate(model, test, "test")
live.make_summary()
