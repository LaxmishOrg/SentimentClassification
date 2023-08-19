import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import typing as t
from pathlib import Path
from pathlib import Path
import re
import string
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize  # Import word_tokenize from nltk
from nltk.corpus import stopwords  # Import the stopwords module
import datetime
import os
import json 

import tensorflow as tf
from tensorflow import keras
from sentiment_model.config.core import config
from sentiment_model import __version__ as _version
from sentiment_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config, PACKAGE_ROOT
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

import nltk
nltk.download('punkt')
nltk.data.path.append(PACKAGE_ROOT / "nltk_data")
nltk.download('stopwords')



##  Pre-Pipeline Preparation

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
    
    sentiment = data_frame['Score'].apply(lambda x : 'positive' if(x > 3) else 'negative')
    
    data_frame.insert(1, "Sentiment", sentiment)
    
    #Convert Sentiment to numerical
    data_frame['Sentiment'] = data_frame['Sentiment'].apply(lambda x: 1 if x=="positive" else 0)
    
    # drop unnecessary variables
    data_frame.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)
    
    duplicated = data_frame.duplicated(subset=['Sentiment','Text'])
    #print("Are there  duplicates",duplicated.value_counts())
    
    data_frame.drop_duplicates(subset=['Sentiment', 'Text'],inplace=True)
    
    data_frame['Time']=data_frame['Time'].apply(lambda x : datetime.datetime.fromtimestamp(x))
    
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
    save_file_name = f"{config.app_config.model_save_file}{_version}"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_model(files_to_keep = [save_file_name])

    # Default callback
    callback_list.append(keras.callbacks.ModelCheckpoint(filepath = save_path,
                                                         save_best_only = config.model_config.save_best_only,
                                                         monitor = config.model_config.monitor))

    if config.model_config.earlystop > 0:
        callback_list.append(keras.callbacks.EarlyStopping(patience = config.model_config.earlystop))

    return callback_list

def save_tokenizer(*, json_object: str)->None:
    # Writing to sample.json
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.tokenizer_filename}{_version}"
    save_path = TRAINED_MODEL_DIR / save_file_name
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
    trained_model = keras.models.load_model(filepath = file_path)
    trained_model = keras.models.load_model(filepath = file_path)
    return trained_model

def remove_old_model(*, files_to_keep: t.List[str]) -> None:
    """
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
        save_tokenizer(json_object=json_object)
    
    return tokenizer
    

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model