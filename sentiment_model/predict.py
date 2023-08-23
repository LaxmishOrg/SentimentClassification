import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Union
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentiment_model import __version__ as _version
from sentiment_model.config.core import config
from sentiment_model.processing.data_manager import load_model, getTokenizer, load_dataset


model_file_name = f"{config.app_config.model_save_file}{_version}"
clf_model = load_model(file_name = model_file_name)

tokenizer = getTokenizer('')
def make_prediction(*, input_data: Union[pd.DataFrame, dict, tf.Tensor]) -> dict:
    """Make a prediction using a saved model """
    
    tokenized_input = tokenizer.texts_to_sequences([input_data])
    padded_input = pad_sequences(tokenized_input, maxlen=config.app_config.max_sequence_length)
  #  reshaped_input = tf.reshape(padded_input, (1, config.app_config.max_sequence_length))

    results = {"predictions": None, "version": _version}
    
    predictions = clf_model.predict(reshaped_input, verbose = 0)
    pred_labels = []
    for i in predictions:
        pred_labels.append(predictions)
        
    results = {"predictions": pred_labels, "version": _version}
    pred = results.get('predictions')
    val = pred[0]
    print(val[0])
    #print(type(results))

    return results


if __name__ == "__main__":

    
    test_sample_1 = "This movie is not good! I really did not like it because it is so good!"
   # test_sample_2 = "Good movie!"
   # test_sample_3 = "Maybe I like this movie."
   # test_sample_4 = "Not to my taste, will skip and watch another movie"
   # test_sample_5 = "if you like action, then this movie might be good for you."
   # test_sample_6 = "Bad movie!"
   # test_sample_7 = "Not a good movie!"
   # test_sample_8 = "This movie really sucks! Can I get my money back please?"
   # test_samples = [test_sample_1, test_sample_2, test_sample_3, test_sample_4, test_sample_5, test_sample_6, test_sample_7, test_sample_8]
    
    #data_in={'Text':['This movie is fantastic! I really like it because it is so good!']}
   
    input_text = "This movie is not good! I really did not like it because it is bad!"
    input_data = pd.DataFrame({"Text": [input_text]})

    
    make_prediction(input_data = input_data)