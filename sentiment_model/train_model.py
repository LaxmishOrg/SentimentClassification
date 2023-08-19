import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences


from sentiment_model.config.core import config
from sentiment_model.model import classifier
from sentiment_model.processing.data_manager import getDataset, load_dataset, callbacks_and_save_model, getTokenizer




def run_training_dataset() -> None:
    """
    Split the dataset
    """
    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    tokenizer = getTokenizer(data)
    xtrain, x_test, ytrain, y_test = train_test_split(
        data['Text'], #data[0],  # predictors
        data['Sentiment'], #data[1],
        test_size=config.model_config.test_size, random_state=config.model_config.random_state)
    
    X_train, X_val, y_train, y_val = train_test_split(
        xtrain, 
        ytrain, 
        test_size=config.model_config.test_size2, random_state=config.model_config.random_state)
     
    X_train = tokenize_and_pad(X_train, tokenizer)
    X_test = tokenize_and_pad(x_test, tokenizer)
    X_val = tokenize_and_pad(X_val, tokenizer)
     
    classifier.fit(X_train, y_train,
                   epochs=config.model_config.epochs,
                   validation_data=(X_val, y_val),
                   callbacks=callbacks_and_save_model(),
                   verbose=config.model_config.verbose)

    # Calculate the score/error
    # test_loss, test_acc = classifier.evaluate(test_data)
    print("Accuracy(in %):", accuracy_score(x_test, y_test) * 100)

    # persist trained model
    save_pipeline(pipeline_to_persist=sentiment_pipe)
    # printing the score

    


def tokenize_and_pad(df: pd.DataFrame , tokenizer :tf.keras.preprocessing.text.Tokenizer)->pd.DataFrame:
    df = tokenizer.texts_to_sequences(df)
    df = pad_sequences(df, maxlen=config.app_config.max_sequence_length) 
    return df

    
def run_training() -> None:
    
    run_training_dataset()
    # Model fitting
    #print("Loss:", test_loss)
    #print("Accuracy:", test_acc)

'''
# Ajay (aug 18)
def run_training() -> None:
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting
    titanic_pipe.fit(X_train, y_train)
    # y_pred = titanic_pipe.predict(X_test)
    # print("Accuracy(in %):", accuracy_score(y_test, y_pred) * 100)

    # persist trained model
    save_pipeline(pipeline_to_persist=titanic_pipe)
    # printing the score
'''
    
if __name__ == "__main__":
    run_training()