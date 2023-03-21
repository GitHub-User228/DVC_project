import os
import sys
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import yaml


params = yaml.safe_load(open("params.yaml"))["prepare"]
C = params["C"]
METRIC = params["metric"]
TARGET = None

pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)

train_path = os.path.join(path, 'data', 'train')
validation_path = os.path.join(path, 'data', 'validation')

train = pd.read_csv(os.listdir(train_path)[0])
validation = pd.read_csv(os.listdir(validation_path)[0])

FEATURES = [col for col in train.columns if col != TARGET]
X_train, X_val, y_train, y_val = train[FEATURES].values, validation[FEATURES].values, train[TARGET].values, validation[TARGET].values

def train(model, kwargs, X_train, X_val, y_train, y_val, metric):
  model = model(**kwargs)
  model.fit(X_train, y_train)
  score = metric(y_val, model.predict(X_test))
  return model, score

model = LogisticRegression
kwargs = {"C": C}
model, score = train(model, kwargs, X_train, X_val, y_train, y_val, METRIC)

print("For C = {} the {} = {}".format(C, metric.__name__, score))

# save
with open('model.pkl','wb') as fd:
    pickle.dump(model, fd)
