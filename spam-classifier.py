# -*- coding: utf-8 -*-
"""
Implements a simple spam classifier based on the SpamAssassin Public Corpus.
TODO:
    - Clean the data
    - Convert the text strings into a list of tokens
    - Understand the stopwords
    - Fit the SVM
"""
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import os
import email
import pandas as pd
from sklearn.utils import shuffle


def read_data(path):
    data = []
    for filename in os.listdir(path):
        f = open(os.path.join(path, filename), 'r', encoding='latin-1')
        message = email.message_from_string(','.join(f.readlines()).replace(',', ''))
        if message.is_multipart():
            for payload in message.get_payload():
                body = payload.get_payload()
        else:
            body = message.get_payload()
        data.append(body)
        f.close()

    return data


def clean_data(data):


    return data


def create_data_set(spam, ham, save=False):
    spam_val = [1] * len(spam)
    ham_val = [0] * len(ham)
    df1 = pd.DataFrame(list(zip(spam, spam_val)), columns=['Mail', 'Class'])
    df0 = pd.DataFrame(list(zip(ham, ham_val)), columns=['Mail', 'Class'])
    df = pd.concat([df1, df0], ignore_index=True)
    if save:
        df.to_csv('train_data.csv', index=True)

    return df


raw_spam = read_data('spam/')
raw_ham = read_data('ham/')
spam = clean_data(raw_spam)
ham = clean_data(raw_ham)
train_data = shuffle(create_data_set(spam, ham, False)).reset_index(drop=True)
