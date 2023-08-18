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
from sentiment_model.processing.features import data_augmentation


# Create a function that returns a model
def create_model(input_dim,output_dim, optimizer, loss, metrics,dropout,rdropout,units):
    
    model_lstm = Sequential()
    model_lstm.add(Embedding(input_dim=input_dim, output_dim=output_dim))
    model_lstm.add(LSTM(units=units,  dropout=dropout, recurrent_dropout=rdropout))
    model_lstm.add(Dense(1, activation='sigmoid'))
    model_lstm.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model_lstm


# Create model
classifier = create_model(input_dim = config.model_config.input_dim,output_dim = config.model_config.output_dim, 
                          optimizer = config.model_config.optimizer, 
                          loss = config.model_config.loss, 
                          metrics = [config.model_config.accuracy_metric],
                          dropout = config.model_config.dropout,
                          rdropout = config.model_config.rdropout,
                          units = config.model_config.units)
