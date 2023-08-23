import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sentiment_model.config.core import config
from sentiment_model.processing.data_manager import _load_raw_dataset


@pytest.fixture
def sample_input_data():
    #data = _load_raw_dataset(file_name=config.app_config.training_data_file)
    input_text = "This movie is fantastic! I really like it because it is so good!"
    data = input_text
    
    return data


'''
@pytest.fixture
def sample_input_data():
    test_data = load_test_dataset()

    return test_data
'''