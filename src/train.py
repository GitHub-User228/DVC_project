import os
import sys
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yaml


params = yaml.safe_load(open("params.yaml"))["train"]
SEED = params["seed"]
N_EST = params["n_est"]
MAX_DEPTH = params["max_depth"]
TARGET = "median_house_value"

input = os.path.join(sys.argv[1], 'train.csv') #####################
output_path = sys.argv[2]
output = os.path.join(output_path, 'model.pkl')

if not os.path.exists(output_path):
    os.makedirs(output_path)

X = pd.read_csv(input, index_col=0)
FEATURES = [col for col in X.columns if col != TARGET]
X_train, y_train = X[FEATURES].values, X[TARGET].values

def train(model, kwargs, X_train, y_train):
  model = model(**kwargs)
  model.fit(X_train, y_train)
  return model

model = RandomForestRegressor
kwargs = {"n_estimators": N_EST, "max_depth": MAX_DEPTH, "random_state": SEED}
model = train(model, kwargs, X_train, y_train)

# save
with open(output, 'wb') as fd:
    pickle.dump(model, fd)
