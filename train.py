import os
import sys
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import yaml


params = yaml.safe_load(open("params.yaml"))["train"]
C = params["C"]
TARGET = None

input = sys.argv[1]
output = sys.argv[2]

X = pd.read_csv(input)
FEATURES = [col for col in X.columns if col != TARGET]
X_train, y_train = X[FEATURES].values, X[TARGET].values

def train(model, kwargs, X_train, y_train):
  model = model(**kwargs)
  model.fit(X_train, y_train)
  return model

model = LogisticRegression
kwargs = {"C": C}
model = train(model, kwargs, X_train, y_train)

# save
with open(output, 'wb') as fd:
    pickle.dump(model, fd)
