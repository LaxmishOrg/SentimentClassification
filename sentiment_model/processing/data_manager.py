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

# 1. Extracts the title (Mr, Ms, etc) from the name variable
def get_title(passenger:str) -> str:
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
    
# 2. processing cabin

f1=lambda x: 0 if type(x) == float else 1  ## Ternary Expression
  
  def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    data_frame["Title"] = data_frame["Name"].apply(get_title)       # Fetching title

    data_frame['FamilySize'] = data_frame['SibSp'] + data_frame['Parch'] + 1  # Family size

    data_frame['Has_cabin']=data_frame['Cabin'].apply(f1)               #  processing cabin 

    # drop unnecessary variables
    data_frame.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)

    return data_frame

def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed

def load_train_dataset():
    train_dataset = image_dataset_from_directory(directory = DATASET_DIR / config.app_config.train_path,
                                                image_size = config.model_config.image_size,
                                                batch_size = config.model_config.batch_size)    
    return train_dataset


def load_validation_dataset():
    validation_dataset = image_dataset_from_directory(directory = DATASET_DIR / config.app_config.validation_path,
                                                    image_size = config.model_config.image_size,
                                                    batch_size = config.model_config.batch_size)
    return validation_dataset


def load_test_dataset():
    test_dataset = image_dataset_from_directory(directory = DATASET_DIR / config.app_config.test_path,
                                                image_size = config.model_config.image_size,
                                                batch_size = config.model_config.batch_size)
    return test_dataset


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
