import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import *

import random
seed =0
random.seed(seed)
np.random.seed(seed)

"""
    CODE START
"""

def get_tfidf_embd(df:pd.DataFrame, *args):
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    X = tfidfconverter.fit_transform(data).toarray()

    for arg in args:
        X = np.concatenate((X, df[arg].values.reshape(-1, 1)), axis=1)

    return X
"""
    CODE END
"""


def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)

