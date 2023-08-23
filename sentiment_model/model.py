import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers

from sentiment_model.config.core import config

# Create a function that returns a model
def create_model(*,input_dim:int,output_dim:int, p_optimizer:str, p_loss:str, 
                 metrics: int,p_dropout:float,
                 r_dropout:float,p_units:int)-> keras.models.Model:
    
    model_lstm = Sequential()
    model_lstm.add(Embedding(input_dim=input_dim, output_dim=output_dim))
    model_lstm.add(LSTM(units=p_units,  dropout=dropout, recurrent_dropout=r_dropout))
    model_lstm.add(Dense(1, activation='sigmoid'))
    model_lstm.compile(loss=p_loss, optimizer=p_optimizer, metrics=metrics)
    return model_lstm


# Create model
classifier = create_model(input_dim = config.model_config.input_dim,
                          output_dim = config.model_config.output_dim, 
                          p_optimizer = config.model_config.optimizer, 
                          p_loss = config.model_config.loss, 
                          metrics = [config.model_config.accuracy_metric],
                          p_dropout = config.model_config.dropout,
                          r_dropout = config.model_config.rdropout,
                          p_units = config.model_config.units)
