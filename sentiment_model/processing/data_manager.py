import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import typing as t
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from sentiment_model.config.core import config
from sentiment_model import __version__ as _version
from sentiment_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


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
    
# 2. processing cabin

f1=lambda x: 0 if type(x) == float else 1  ## Ternary Expression
  
def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    data_frame.dropna(subset = ['ProfileName', 'Summary'],inplace=True)
    
    sentiment = data_frame['Score'].apply(lambda x : 'positive' if(x > 3) else 'negative')
    
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
    
    # Prepare versioned save file name
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


def load_model(*, file_name: str) -> keras.models.Model:
    """Load a persisted model."""

    file_path = TRAINED_MODEL_DIR / file_name
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
