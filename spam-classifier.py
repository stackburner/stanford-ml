# -*- coding: utf-8 -*-
"""
Implements a spam classifier based on the SpamAssassin Public Corpus. Only the email body
is used for classification. The data is cleaned, vectorized and split into a training- and
test-dataset. Stemming is not implemented yet.
"""
import nltk
from nltk.corpus import stopwords
import os
import email
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import re


def read_data(path):
    data = []
    for filename in os.listdir(path):
        f = open(os.path.join(path, filename), 'r', encoding='latin-1')
        message = email.message_from_string(','.join(f.readlines()).replace(',', ''))
        if not message.is_multipart():
            body = message.get_payload()
            data.append(body)
        f.close()

    return data


def clean_data(data, stops):
    for i, row in enumerate(data):
        row = strip_html(row)
        row = strip_emoticon(row)
        row = strip_number(row)
        row = [w for w in row.split() if w not in stops]
        data[i] = row

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


def strip_html(data):
    data = re.sub('<[^>]*>', '', data)

    return data


def strip_emoticon(data):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', data.lower())
    data = re.sub('[\W]+', ' ', data.lower()) + ' '.join(emoticons).replace('-', '')

    return data


def strip_number(data):
    data = re.sub('^\d+\s|\s\d+\s|\s\d+$', 'number', data)

    return data


nltk.download('stopwords')
stops = stopwords.words('english')
raw_spam = read_data('spam/')
raw_ham = read_data('ham/')
spam = clean_data(raw_spam, stops)
ham = clean_data(raw_ham, stops)
train_data = shuffle(create_data_set(spam, ham, False)).reset_index(drop=True)
X_train = train_data["Mail"][:500]
Y_train = train_data["Class"][:500]
X_test = train_data["Mail"][501:1000]
Y_test = train_data["Class"][501:1000]
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=1000)
train_data_features = vectorizer.fit_transform(X_train.map(' '.join))
train_data_features = train_data_features.toarray()
test_data_features = vectorizer.transform(X_test.map(' '.join))
test_data_features = test_data_features.toarray()
clf = svm.SVC(kernel='linear', C=1)
clf.fit(train_data_features, Y_train)
predicted = clf.predict(test_data_features)
accuracy = np.mean(predicted == Y_test)
print("Accuracy: ", accuracy)
X = train_data["Mail"][1001:1002]
validation_data = vectorizer.transform(X.map(' '.join))
validation_data = validation_data.toarray()
print("Mail: ", X)
classification = clf.predict(validation_data)
print("Classification: ", classification)
