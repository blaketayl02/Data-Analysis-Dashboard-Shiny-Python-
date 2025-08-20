import pandas as pd
import json
import pytz
import nltk
import numpy as np
import string
nltk.data.path.append("nltk_data")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



def load_and_clean_data() -> dict:
    final_combined_df = pd.read_csv("Data Test/NewKato/ShinyFinalDataFeb11-June30.csv")
    old_kato_df = pd.read_csv("Data Test/NewKato/oldKatoFinalDataframe.csv")
    july_kato_df = pd.read_csv("Data Test/NewKato/ShinyFinalDataJuly25.csv")

    
    data_map_24 = {
        "Old Kato": old_kato_df,
        "New Kato": final_combined_df,
        "New Kato July": july_kato_df
    }


    return data_map_24
