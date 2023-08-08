import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Union
import pandas as pd
import numpy as np

from titanic_model import __version__ as _version
from titanic_model.config.core import config
from titanic_model.pipeline import titanic_pipe
from titanic_model.processing.data_manager import load_pipeline
from titanic_model.processing.data_manager import pre_pipeline_preparation
from titanic_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
titanic_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    #print(validated_data)
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
  
    results = {"predictions": None, "version": _version, "errors": errors}
    
    #predictions = titanic_pipe.predict(validated_data)

    #print(results)
    if not errors:
        
        predictions = titanic_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={'PassengerId':[60],'Pclass':[2],'Name':["Caldwell, Master. Alden Gates"],'Sex':['male'],'Age':[0.83],
                'SibSp':[2],'Parch':[3],'Ticket':[48738],'Cabin':['ramen'],'Embarked':['S'],'Fare':[45]}
    
    print(make_prediction(input_data=data_in))
