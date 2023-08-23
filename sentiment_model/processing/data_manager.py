import sys
from pathlib import Path
import os
#sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from datetime import datetime
import typing as t
import pandas as pd
import json
from pathlib import Path
import string
from nltk.corpus import stopwords
import re
import nltk
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow import keras
#import sentiment_model.config
from sentiment_model.config.core import config
from sentiment_model import __version__ as _version
from sentiment_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config, PACKAGE_ROOT
#import DATASET_DIR, TRAINED_MODEL_DIR, config
from keras.preprocessing.text import Tokenizer, tokenizer_from_json

##  Pre-Pipeline Preparation
#DATASET_DIR = config.app_config.save_best_only,

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def preprocess_text(sen):

    sen = re.sub('<.*?>', ' ', sen)                        # remove html tags

    tokens = word_tokenize(sen)           # tokenize words

    tokens = [w.lower() for w in tokens]                   # convert to lower case
    table = str.maketrans('', '', string.punctuation)      # remove punctuations
    stripped = [w.translate(table) for w in tokens]

    words = [word for word in stripped if word.isalpha()]  # remove non-alphabet
    stop_words = set(stopwords.words('english'))

    words = [w for w in words if not w in stop_words]      # remove stop words

    words = [w for w in words if len(w) > 2]

    return words
 
 
def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    data_frame.dropna(subset = ['ProfileName', 'Summary'],inplace=True)
    
    sentiment = data_frame['Score'].apply(lambda x : 1 if(x > 3) else 0)
    
    data_frame.insert(1, "Sentiment", sentiment)
    
     
    # drop unnecessary variables
    data_frame.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)

    data_frame.drop_duplicates(subset=['Sentiment', 'Text'],inplace=True)
    
    data_frame['Time']=data_frame['Time'].apply(lambda x : datetime.fromtimestamp(x))
    
    data_frame['Text'] = data_frame['Text'].apply(preprocess_text)
    
    return data_frame

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)
    return transformed

# Define a function to return a commmonly used callback_list
def callbacks_and_save_model():
    callback_list = []
    
# Define a function to return a commmonly used callback_list
def callbacks_and_save_model():
    callback_list = []
    
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.model_save_file}{_version}"
  #  save_file_name = f"{config.app_config.model_save_file}{_version}"
    #save_path = "C:\Projects_py\Reviews.csv"
    save_path= TRAINED_MODEL_DIR / save_file_name

    remove_old_model(files_to_keep = ["Reviews.cvs"])

    

    if config.model_config.earlystop > 0:
        callback_list.append(keras.callbacks.EarlyStopping(patience = config.model_config.earlystop))
    
    # Default callback
    callback_list.append(keras.callbacks.ModelCheckpoint(filepath = str(save_path),
                                                         save_best_only = config.model_config.save_best_only,
                                                         monitor = config.model_config.monitor))
    return callback_list

def save_tokenizer(json_object: str)->None:
    # Writing to sample.json
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.tokenizer_filename}{_version}"
    save_path = TRAINED_MODEL_DIR/save_file_name
    with open(save_path, "w") as outfile:
      outfile.write(json_object)

def load_tokenizer(filename):
   with open(filename, 'r') as openfile:
     # Reading from json file
    json_object = json.load(openfile)
    return json_object

def load_model(*, file_name: str) -> keras.models.Model:
    """Load a persisted model."""

    file_path = TRAINED_MODEL_DIR / file_name
    #trained_model = keras.models.load_model(filepath = file_path)
    trained_model = keras.models.load_model(filepath = file_path)
    return trained_model


def remove_old_model(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old models.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    Remove old models.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

def getDataset(*,df: pd.DataFrame)->tf.data.Dataset:
      dataset_text = tf.data.Dataset.from_tensor_slices(df['Text'])
      dataset_sentiment = tf.data.Dataset.from_tensor_slices(df['Sentiment'])
      dataset = tf.data.Dataset.zip((dataset_text, dataset_sentiment))
      return dataset   

def getTokenizer()->tf.keras.preprocessing.text.Tokenizer:
    save_file_name = f"{config.app_config.tokenizer_filename}{_version}"
    save_path = TRAINED_MODEL_DIR / save_file_name
    
    if os.path.exists(save_path):
        tokenizer_json = load_tokenizer(save_path)
        tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

def getTokenizer(train_data_frame_text: pd.DataFrame)->tf.keras.preprocessing.text.Tokenizer:
    save_file_name = f"{config.app_config.tokenizer_filename}{_version}"
    save_path = TRAINED_MODEL_DIR / save_file_name
    
    if os.path.exists(save_path):
        tokenizer_json = load_tokenizer(save_path)
        tokenizer = tokenizer_from_json(tokenizer_json)
    else:
        tokenizer = Tokenizer(num_words=config.app_config.max_num_words)
        tokenizer.fit_on_texts(train_data_frame_text)
        json_object = json.dumps(tokenizer.to_json())
        save_tokenizer(json_object)
    
    return tokenizer
    

def load_pipeline(*, file_name: str):
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe
