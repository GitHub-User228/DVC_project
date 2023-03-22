import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml

def split_data(data, test_size, seed):
  train, val = train_test_split(data, test_size=test_size, seed=seed)
  return train, val

def preprocess_data(train, val):
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    return train, val

params = yaml.safe_load(open("params.yaml"))["prepare"]
  
SEED = params["seed"]
TEST_SIZE = params["split"]
  
pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)

input = sys.argv[1]
prepared_data_path = os.path.join(path, 'data', 'prepared')
if not os.path.exists(prepared_data_path):
    os.makedirs(prepared_data_path)

    
data = pd.read_csv(input)

train, val = split_data(data, TEST_SIZE, SEED)

train, val = preprocess_data(train, val)

train.to_csv(os.path.join(prepared_data_path, 'train.csv'))
val.to_csv(os.path.join(prepared_data_path, 'val.csv'))    