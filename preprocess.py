import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def split_data(data, test_size, seed):
  train, test = train_test_split(data, test_size=test_size, seed=seed)
  return train, test

def preprocess_text(input_text):
    input_text = input_text.lower()
    input_text = input_text.replace('\n',' ')
    input_text = input_text.split(' ')
    input_text = [word for word in input_text if not(word == '')]
    input_text = ' '.join(input_text)
    return input_text  

def preprocess_data(data, preprocess_func):
    '''
    Function to preprocess all texts in data using "preprocess_func"
    * Input:
        - data: data in DataFrame format with at least ['text','label'] columns
        - preprocess_func: function to preprocess a single string 
    * Output: data with preprocessed ['text'] column
    '''
    data['text'] = data['text'].apply(lambda x: preprocess_func(x))
    return data

params = yaml.safe_load(open("params.yaml"))["prepare"]
  
SEED = params["seed"]
TEST_SIZE = params["split"]
  
pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)

data_path = os.path.join(path, 'data', 'raw')
train_path = os.path.join(path, 'data', 'train')
validation_path = os.path.join(path, 'data', 'validation')
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(validation_path):
    os.makedirs(validation_path)

    
data = pd.read_json(os.listdir(data_path)[0])

train, test = split_data(data, TEST_SIZE, SEED)

train = preprocess_data(train, preprocess_text)
test = preprocess_data(test, preprocess_text)

train.to_json(os.path.join(train_path, 'train.json'))
test.to_json(os.path.join(test_path, 'test.json'))              



  
  
  
