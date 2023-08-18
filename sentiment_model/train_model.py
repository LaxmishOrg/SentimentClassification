import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from sentiment_model.config.core import config
from sentiment_model.model import classifier
from sentiment_model.processing.data_manager import getDataset, load_dataset,callbacks_and_save_model,getTokenizer


def load_dataset() -> None:
    
    """
    Split the dataset
    """
    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    tf.keras.preprocessing.text.Tokenizer tokenizer = getTokenizer()
    xtrain, x_test, ytrain, y_test = train_test_split(
        data[0],  # predictors
        data[1],
        test_size=config.model_config.test_size_1,random_state = 0)
    
    X_train, X_val, y_train, y_val = train_test_split(
        xtrain, 
        ytrain, 
        test_size = config.model_config.test_size_2, random_state = 0)
     
    X_train = tokenize_and_pad(X_train,tokenizer)
    X_test = tokenize_and_pad(X_test,tokenizer)
    X_val = tokenize_and_pad(X_val,tokenizer)
     
     classifier.fit(X_train, y_train,
                   epochs = config.model_config.epochs,
                   validation_data = (X_val,y_val),
                   callbacks = callbacks_and_save_model(),
                   verbose = config.model_config.verbose)

    # Calculate the score/error
    #test_loss, test_acc = classifier.evaluate(test_data)
    print("Accuracy(in %):", accuracy_score(x_test, y_test)*100)

    # persist trained model
    save_pipeline(pipeline_to_persist= sentiment_pipe)
    # printing the score
    
if __name__ == "__main__":
    run_training()

    



def tokenize_and_pad(*,df: pd.DataFrame , tokenizer :tf.keras.preprocessing.text.Tokenizer)->pd.DataFrame:
    df = tokenizer.texts_to_sequences(df)
    df = pad_sequences(df, maxlen=config.app_config.max_sequence_length) 
    return df

    
def run_training() -> None:
    
    load_dataset()
    # Model fitting
    #print("Loss:", test_loss)
    #print("Accuracy:", test_acc)

    
if __name__ == "__main__":
    run_training()