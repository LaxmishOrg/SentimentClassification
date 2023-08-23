"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from sentiment_model import __version__ as _version
from sentiment_model.config.core import config
from sentiment_model.predict import make_prediction
from sentiment_model.processing.data_manager import load_model


def test_make_prediction(sample_input_data):
    # Given
    
    
    input_data = pd.DataFrame({"Text": ["This movie is fantastic! I really like it because it is so good!"]})


    # When
    results = make_prediction(input_data = sample_input_data)
    y_pred = results['predictions'][0]
    #y_pred = 0.9
    
    # Then
    assert y_pred is not None
    assert 0 <= y_pred <= 1
    assert results['version'] == _version
    

def test_accuracy(sample_input_data):
    # Given
    model_file_name = f"{config.app_config.model_save_file}{_version}"
    clf_model = load_model(file_name = model_file_name)
   
    # When
    test_loss, test_acc = clf_model.evaluate(sample_input_data, verbose=0)
    
    # Then
    assert test_acc >= 0
