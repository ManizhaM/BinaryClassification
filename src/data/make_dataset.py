import pandas as pd
import numpy as np
from collections.abc import Iterable
import datasets
import math
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def data_preprocessing():
    data = datasets.load_dataset('ucirvine/sms_spam')
    data.set_format(type='pandas')
    df = data['train'].to_pandas()
    df['sms'] = df['sms'].apply(lambda x: x.lower())
    df['sms'] = df['sms'].apply(word_tokenize)

    df.to_csv('data/preprocessed_data.csv', index=False)
    return df


def data_split(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df.sms.tolist(), df.label.tolist(), test_size=0.3, random_state=42
    )

    train_df = pd.DataFrame({'sms': X_train, 'label': y_train})
    test_df = pd.DataFrame({'sms': X_test, 'label': y_test})
    
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    return X_train, X_test, y_train, y_test
