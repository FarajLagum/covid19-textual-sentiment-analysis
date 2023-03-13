# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: ta-sa
#     language: python
#     name: ta-sa
# ---

# %% [markdown] id="30f7cc46"
# # Machine Learning For Sentiment Analysis
#
# This notebook provides instructions for preparing data and constructing a machine learning model to perform sentiment analysis. The model architecture consists of an RNN-CNN-GRU joint architecture with a pre-trained/customized embedding utilizing word2vec. The code is suitable for execution on Jupyter Notebook or Google Colab.

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 5329, "status": "ok", "timestamp": 1678558191031, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="214210b4" outputId="355d1559-e7c6-4549-f03d-9c744050172a"
# pandas version 1.1.4
from time import strftime
from pathlib import Path
import shutil
import tensorflow.keras.metrics as ms
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, concatenate, Bidirectional, GRU, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import tensorflow
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Conv1D
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, GRU, Bidirectional, Dropout, Flatten
from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
import pickle
import numpy as np
from sklearn import utils
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
from gensim.models.word2vec import Word2Vec
import gensim
import os
import sys
from tqdm import tqdm
import re
import pandas as pd

pd.__version__

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1678558191032, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="ebcfc09f" outputId="06dab3a0-2d19-477a-e276-fd4e1f840ce7"
print(gensim.__version__)

if gensim.__version__ < '4.3.1':
    # !pip install --upgrade gensim

    # gensim-4.3.1

    # %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1678558191034, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="a599b9b5" outputId="99a09e73-097f-4f8f-f1f5-425c61363c04"
    print(tf.__version__)

# %% [markdown] id="e972fd89"
# ## Mount Google Drive
# To Mount your Google Drive, follow the steps. If you are running it on local machine, mounting will be skipped.
#
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1336, "status": "ok", "timestamp": 1678558192360, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="e7741d9d" outputId="3ca133c8-2085-4640-a13f-fc17bdc7dbf2"
# Location of the files


is_colab = "google.colab" in sys.modules

if is_colab:
    from google.colab import drive
    drive.mount('/content/drive/')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1678558192361, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="fdfb5fc6" outputId="fc99fc74-b6d2-401e-b785-5e29bd61742b"
cwd = os.getcwd()
print(cwd)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1678558192362, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="e9130d41" outputId="bab657b5-e8e1-4d71-d70e-a6b21402086b"
is_colab

# %% executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1678558192362, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="00684b3e"
NUM_CLASSES = 2
num_classes = NUM_CLASSES

# %% [markdown] id="93ee6883"
# ## Setup the data, models and notebooks paths
#
#
#

# %% executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1678558192362, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="80354260"

if is_colab:
    data_file_location = '/content/drive/My Drive/Colab Notebooks/ta-sa/sa/data/'
    model_file_location = '/content/drive/My Drive/Colab Notebooks/ta-sa/sa/models/'
else:
    data_file_location = './data/'
    model_file_location = './models/'


tag = 'March-2023'

# Trianing data
training_data_covid_file = 'COVID19_Tweet_Main_annotated_V2_clean_training.csv'
training_data_covid_path = data_file_location + training_data_covid_file

training_data_oc_file = 'OCTranspo-all-Data_5classes_clean.csv'
training_data_oc_path = data_file_location + training_data_oc_file


training_data_quayside_file = 'Quayside-all-Data_5classes_clean.csv'
training_data_quayside_path = data_file_location + training_data_quayside_file


# Testing data
testing_data_file = 'COVID19_Tweet_Main_annotated_V2_clean_testing.csv'
testing_data_path = data_file_location + testing_data_file


# Embedding data
embedding_data_file = 'COVID19DATASET_APRIL_10_Embedding_clean.csv'
embedding_data_path = data_file_location + embedding_data_file


COIVD19_Best_Model_file = 'COVID19-Best-model-' + \
    str(NUM_CLASSES) + 'classes_' + tag + '.hdf5'
COIVD19_Best_Model_path = model_file_location + COIVD19_Best_Model_file


tokenizer_file = 'COVID19-tokenizer-' + \
    str(NUM_CLASSES) + 'classes_' + tag + '.pickle'
tokenizer_path = model_file_location + tokenizer_file

# %% executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1678558192363, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="a83cb304"

# Define the mapping dictionary
mapping_dict = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}

# if NUM_CLASSES == 3:
#     # Apply the mapping dictionary to the column
#     training_data_covid['Score 5-Classe'] = training_data_covid['Score 5-Classe'].map(
#         mapping_dict)


def classes_mapping(NUM_CLASSES, mapping_dict, df):
    if NUM_CLASSES == 3:
        # Apply the mapping dictionary to the column
        df['Score 5-Classe'] = df['Score 5-Classe'].map(mapping_dict)
    if NUM_CLASSES == 2:
        # Remove rows where col_mapped is 1
        # df['Score 5-Classe'] = df['Score 5-Classe'].astype(str)
        df = df[df['Score 5-Classe'] != 2]
        mapping_dict = {0: 0, 1: 0, 3: 1, 4: 1}
        df['Score 5-Classe'] = df['Score 5-Classe'].map(mapping_dict)

    return df


# %% [markdown] id="d3adbc14"
# ## Loading data
#
# Change the paths to reflect the dataset locations
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1678558192363, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="efb171c0" outputId="afb9f62b-0c70-45d9-c1fc-84793d1aac8c"
# Load the training data 1

training_data_covid = pd.read_csv(
    training_data_covid_path, encoding="ISO-8859-1", on_bad_lines='skip', usecols=[1, 5])
training_data_covid.head(10)

training_data_covid = classes_mapping(
    NUM_CLASSES, mapping_dict, training_data_covid)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1678558192363, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="93f228fa" outputId="97776686-9005-4846-e3ac-bd9ced3679cf"
training_data_covid.info()
training_data_covid.columns

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 310, "status": "ok", "timestamp": 1678558192667, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="b09d298d" outputId="64094822-f406-46c2-9feb-662364c1029d"
training_data_covid["Score 5-Classe"].value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1678558192667, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="af2b20c0" outputId="77c856cf-04b3-4526-bdb7-9f30c900500b"
training_data_covid["Score 5-Classe"].value_counts()

# %% executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1678558192668, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="451c6076"
text_col = "Text"

sentiment_col = 'Sentiment'

# %% colab={"base_uri": "https://localhost:8080/", "height": 238} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1678558192668, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="5ff408df" outputId="8c1e761a-23ae-42fd-e42c-4347ac26605d"
training_data_covid.columns = [text_col, sentiment_col]
training_data_covid.head(6)

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1678558192668, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="b41e8873" outputId="be69db94-677a-45b1-c3fa-6bb238f4aaef"
# Load the training data 2

training_data_oc = pd.read_csv(
    training_data_oc_path, encoding="ISO-8859-1", on_bad_lines='skip', usecols=[1, 2])
training_data_oc.head(2)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1678558192669, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="d2bcbe7c" outputId="7e7567ac-3924-4ad1-96ad-9f42ea5b1beb"


training_data_oc = classes_mapping(NUM_CLASSES, mapping_dict, training_data_oc)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1678558192669, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="92e82c14" outputId="d65113cb-6d8e-439f-832d-79863e9c09ec"
training_data_oc["Score 5-Classe"].value_counts()

# %% executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1678558192670, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="9e4b92e6"
# change the order of the coloumns
training_data_oc = training_data_oc[['clean_text', 'Score 5-Classe']]

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1678558192670, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="916621ad" outputId="d6744586-e651-4d47-ce62-9c2d5b306c2a"


training_data_oc.columns = [text_col, sentiment_col]
training_data_oc.head(2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1678558192671, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="47e03c8f" outputId="287964d6-0e73-48fe-a7cf-0ce3e3315124"
# Load the training data 3

training_data_quayside = pd.read_csv(training_data_quayside_path, encoding="ISO-8859-1",
                                     on_bad_lines='skip', usecols=[1, 2])  # error_bad_lines=False


training_data_quayside.head(2)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 21, "status": "ok", "timestamp": 1678558192671, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="e70fce4f" outputId="0499ff5c-9cc8-4793-94a6-dd14410f7998"
# if NUM_CLASSES == 3:
#     # Apply the mapping dictionary to the column
#     training_data_quayside['Score 5-Classe'] = training_data_quayside['Score 5-Classe'].map(
#         mapping_dict)


training_data_quayside = classes_mapping(
    NUM_CLASSES, mapping_dict, training_data_quayside)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1678558192672, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="c05ee605" outputId="5bec100f-bc7c-4f58-ec50-731d4b9d1100"
training_data_quayside["Score 5-Classe"].value_counts()

# %% executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1678558192673, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="0277201e"

# change the order of the coloumns
training_data_quayside = training_data_quayside[[
    'clean_text', 'Score 5-Classe']]

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1678558192906, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="c8c5d46d" outputId="bc056eb1-8f90-4e8b-859b-b60f12cf02c7"

training_data_quayside.columns = [text_col, sentiment_col]
training_data_quayside.head(10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1678558192906, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="92511e33" outputId="7f7f180c-568f-46fe-f340-f4cb99ff427d"
# Load Testing Data

testing_data = pd.read_csv(testing_data_path, encoding="ISO-8859-1",
                           on_bad_lines='skip', usecols=[1, 5])  # error_bad_lines=False
testing_data.head(2)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1678558192907, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="840cee44" outputId="cb08c601-79b3-4994-86f9-47787fb97e6b"
# if NUM_CLASSES == 3:
#     # Apply the mapping dictionary to the column
#     testing_data['Score 5-Classe'] = testing_data['Score 5-Classe'].map(
#         mapping_dict)

testing_data = classes_mapping(NUM_CLASSES, mapping_dict, testing_data)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1678558192907, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="21327aa0" outputId="13927a05-52b2-493c-d6e5-56cd98c8d862"
testing_data["Score 5-Classe"].value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1678558192907, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="ec9a9cdc" outputId="1aa1e8e2-566e-4c83-9c0a-909856799a00"
testing_data.columns = [text_col, sentiment_col]
testing_data.head(2)

# %% executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1678558192907, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="99e63607"
testing_data.dropna(inplace=True)

# %% executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1678558192908, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="524b48bc"
# testing_data[text_col][100:]

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} executionInfo={"elapsed": 244, "status": "ok", "timestamp": 1678558193139, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="e87b2c01" outputId="8378544b-5c7d-4a6a-ad82-03ac488c5706"
# Load Embedding Data

embedding_data = pd.read_csv(
    embedding_data_path, encoding="ISO-8859-1", on_bad_lines='skip', usecols=[4])
embedding_data.head(2)

# %% executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1678558193140, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="e2e6de50"
# import numpy as np
# embedding_data['clean_text'][1] = np.nan

# %% executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1678558193140, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="f7933759"
embedding_data.dropna(inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1678558193140, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="f732dc9f" outputId="289eaaf0-fd1e-4452-92fd-9a5dc0357bcd"
embedding_data.columns = [text_col]
embedding_data.head()

# %% [markdown] id="6741ce06"
# ## Data Preparation
#
# We already did data cleaning in a separate file which include several cleaning operations.

# %% executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1678558193140, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="a0276008"
# Concatenate all training datasets

training_data = pd.concat(
    [training_data_oc, training_data_quayside, training_data_covid])  # training_data_covid,

# training_data = pd.concat([training_data1, training_data2])

training_data.dropna(inplace=True)

# X_train
x_train = training_data[text_col]
x_train_covid = training_data_covid[text_col]

# y_train
y_train = training_data[sentiment_col]
y_train_covid = training_data_covid[sentiment_col]

# %% executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1678558193140, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="84fdcb81"
testing_data.dropna(inplace=True)

x_test = testing_data[text_col]

y_test = testing_data[sentiment_col]

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 206, "status": "ok", "timestamp": 1678558193337, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="17fec1ec" outputId="c6175574-063f-4e1d-ff2a-addc89028cd0"
# Concatenate the text of all datasets so we can use them for Word embedding (word2vec) and tokenization. 
# The testing data will not be used in training


x = embedding_data[text_col]

word_embedding_text = pd.concat(
    [x,  x_train, x_test], ignore_index=True)  # x_train_covid,

type(word_embedding_text)
word_embedding_text[0]

# Word_embedding_text = Word_embedding_text.drop([3240, 9224])

# Word_embedding_text.to_csv(data_file_location + 'Word_embedding_text.csv')

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 16, "status": "ok", "timestamp": 1678558193337, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="2b386882" outputId="cb7908b9-dfc1-4492-d0e8-0c2bb95873f3"
# Word_embedding_text.drop([0, 1])

word_embedding_text[3240]

# %% executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1678558193338, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="94802335"
# Word_embedding_text.head()

# %% [markdown] id="b4ec257b"
# ### Word2Vec
# We will train two Word2Vec models using Continuous Bag Of Words (CBOW) and Skip Gram models. For each model, each word has a vector size of 100. So, the vector representation of each word has a dimension of 200.

# %% executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1678558193338, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="bc1b1056"
# #!pip install -U gensim
# super_data.CleanText=super_data.CleanText.astype(str) # Prevent pandas form automatically converts input to float

# %% executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1678558193339, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="988406f7"


def labelize_documents(docs):
    labelized_docs = []
    for ind, doc in zip(docs.index, docs):
        labelized_docs.append(TaggedDocument(
            doc.split(), ["doc_id" + '_%s' % ind]))
    return labelized_docs


# all_x = pd.concat([x_train, x_test])
text_doc_id = labelize_documents(word_embedding_text)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1678558193339, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="3154b7ac" outputId="1a9ccc8c-17da-48ff-8bc6-d8a27e225469"
text_doc_id[0][0]

# %% [markdown] id="b0ee3a10"
# ### CBOW

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 460, "status": "ok", "timestamp": 1678558193790, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="39faf5b2" outputId="ffc52ebe-80f8-49a2-aa34-c7a4d9a177d4"
# CBOW
cores = multiprocessing.cpu_count()
# Continuous Bag Of Words
model_cbow = Word2Vec(sg=0, vector_size=100, negative=20, window=5,
                      min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
# sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
# size/vector_size (int, optional) – Dimensionality of the word vectors.
# window (int, optional) – Maximum distance between the current and predicted word within a sentence.
# min_count (int, optional) – Ignores all words with total frequency lower than this.
# workers (int, optional) – Use these many worker threads to train the model (=faster training with multicore machines).
# negative (int, optional) – If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used..The paper says that selecting 5-20 words works well for smaller datasets, and you can get away with only 2-5 words for large datasets.
# alpha (float, optional) – The initial learning rate.
# min_alpha (float, optional) – Learning rate will linearly drop to min_alpha as training progresses.

# build_vocab: Build vocabulary from a sequence of sentences
model_cbow.build_vocab([x.words for x in tqdm(text_doc_id)])

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1678558193791, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="4ac3817e" outputId="1d47e995-b570-4eb6-f2cc-0a22b568f55c"
cores

# %% [markdown]
# `train`: Update the model’s neural weights from a sequence of sentences.
#  total_examples (int) – Count of sentences.
# `total_words` (int) – Count of raw words in sentences.
# `epochs` (int) – Number of iterations (epochs) over the corpus.
# `start_alpha` (float, optional) – Initial learning rate. If supplied, replaces the starting alpha from the constructor, for this one call to`train()`. Use only if making multiple calls to train(), when you want to manage the alpha learning-rate yourself (not recommended).
# `end_alpha` (float, optional) – Final learning rate. Drops linearly from start_alpha. If supplied, this replaces the final `min_alpha` from the constructor, for this one call to train(). Use only if making multiple calls to train(), when you want to manage the alpha learning-rate yourself (not recommended).
# `word_count` (int, optional) – Count of words already trained. Set this to 0 for the usual case of training on all words in sentences.
# `queue_factor` (int, optional) – Multiplier for size of queue (number of workers * queue_factor).
# `report_delay` (float, optional) – Seconds to wait before reporting progress.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 21234, "status": "ok", "timestamp": 1678558215019, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="6677e365" outputId="4346b21d-5d7a-440f-ff70-0a92ae6eac16"
# %%time
for iteration in range(15):
    model_cbow.train(utils.shuffle([x.words for x in text_doc_id]), total_examples=len(
        text_doc_id), epochs=1)  # tqdm(text_doc_id)
    model_cbow.alpha -= 0.002
    model_cbow.min_alpha = model_cbow.alpha

# %% [markdown] id="8e8dc333"
# ### Skip Gram

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 265, "status": "ok", "timestamp": 1678558215263, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="c3061562" outputId="9f1fa781-4975-4ea6-8d08-f0d71bed9dc8"
# Skip Gram
model_skip_gram = Word2Vec(sg=1, vector_size=100, negative=20,
                           window=5, min_count=2, workers=cores, alpha=0.02)
model_skip_gram.build_vocab([x.words for x in tqdm(text_doc_id)])

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 39117, "status": "ok", "timestamp": 1678558254375, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="6f8895e9" outputId="d0607ded-7961-4679-9cea-808b841fcd9d"
# %%time
for epoch in range(15):
    model_skip_gram.train(utils.shuffle([x.words for x in
                                         text_doc_id]), total_examples=len(text_doc_id), epochs=1)  # tqdm(text_doc_id)
    model_skip_gram.alpha -= 0.002
    # print(model_ug_sg.alpha)
    model_skip_gram.min_alpha = model_skip_gram.alpha

# %% [markdown] id="2a588c95"
# ### Embeddings with CBOW and Skip-Gram

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1678558254376, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="3f57b28e" outputId="577cb74f-f398-41b9-f80e-e0617db2963d"

embeddings_index = {}
for w in model_cbow.wv.key_to_index.keys():
    embeddings_index[w] = np.append(model_cbow.wv[w], model_skip_gram.wv[w])
print('Found %s word vectors.' % len(embeddings_index))
# embeddings_index

# %% [markdown] id="ddd0ba26"
# ### Tokenizer
#
# We build the the tokenizer and save it to be used later.
#

# %% executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1678558254376, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="c564bce3"
# #!pip install tensorflow
# #!pip install keras


MAX_WORDS = 60000
# num_words = MAX_WORDS ; num_words = None
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(word_embedding_text)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_train_covid_seq = tokenizer.texts_to_sequences(x_train_covid)

x_test_seq = tokenizer.texts_to_sequences(x_test)

# saving tokenizer
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% [markdown] id="3b43bbc5"
# ### Encoding the data
#
# We use one-hot encoding, also called categorical encoding. It is widely used for categorical data.
#

# %% executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1678558254376, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="8b432faf"

y_train_labels = to_categorical(y_train)
y_train_covid_labels = to_categorical(y_train_covid)

y_test_labels = to_categorical(y_test)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1678558254377, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="694d79d4" outputId="5b529a07-0249-47ab-8525-6f60ae0ea406"
y_train_labels[2]

# %% [markdown] id="8241f009"
# ### Sequence Padding
# ``pad_sequences`` is used to ensure that all sequences in a list have the same length. By default (``maxlen=None``) this is done by padding 0 in the beginning of each sequence until each sequence has the same length as the longest sequence.

# %% executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1678558254377, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="1cf1cc72"
# from keras.preprocessing.sequence import pad_sequences

# pad_sequences = ""
# from keras.utils import Sequence
MAX_LENGTH = 40   # the maximum length of words per tweet.

x_train_padded = pad_sequences(x_train_seq, maxlen=MAX_LENGTH)  # maxlen=MAXLEN
y_train_covid_padded = pad_sequences(
    x_train_covid_seq, maxlen=MAX_LENGTH)  # maxlen=MAXLEN

x_test_padded = pad_sequences(x_test_seq, maxlen=MAX_LENGTH)


# len(padded_test[0])
# padded_test[0]

# %% executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1678558254377, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="c71ccf9c"
embed_size = 200

# maximum number of words kept after tokenization based on their word frequency
embed_matrix = np.zeros((MAX_WORDS, embed_size))
for word, i in tokenizer.word_index.items():
    if i >= MAX_WORDS:
        continue
    embed_vector = embeddings_index.get(word)
    if embed_vector is not None:
        embed_matrix[i] = embed_vector

# %% executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1678558254378, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="5eaf6bbb"


my_metrics = [ms.Precision(thresholds=0.4),
              ms.Recall(thresholds=0.4), ms.CategoricalAccuracy(), ms.AUC()]


def get_model(my_metrics):
    # Input layer:
    inpt = Input(shape=(MAX_LENGTH, ))
    # Emnedding Layer:
    layer = Embedding(MAX_WORDS, 200, weights=[
                      embed_matrix], input_length=MAX_LENGTH, trainable=True)(inpt)
    # Spatial dropout layer I:
    layer = SpatialDropout1D(0.3)(layer)
    # Bidirectional RNN (GRU) layer:
    layer = Bidirectional(GRU(100, return_sequences=True))(layer)
    # Convolutional layer:
    layer = Conv1D(128, kernel_size=2, padding="valid",
                   kernel_initializer="he_uniform")(layer)
    #  Pooling layers:
    avg_pool = GlobalAveragePooling1D()(layer)
    max_pool = GlobalMaxPooling1D()(layer)
    conc = concatenate([avg_pool, max_pool])
    # Dropout layer II:
    conc = Dropout(0.3)(conc)
    # DNN layer:
    den_layer = Dense(64, activation='relu')(conc)
    outp = Dense(num_classes, activation="softmax")(
        den_layer)  # sigmoid; softmax
    # Complete model
    model = Model(inputs=inpt, outputs=outp)
    # Defining the loss function, optimozer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(
        learning_rate=0.001), metrics=my_metrics)  # Adam, Nadam, SGD, Adamax
    return model

# %% executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1678558254378, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="4fa6ff42"


def get_model_b():
    # Input layer:
    inpt = Input(shape=(MAX_LENGTH, ))
    # Emnedding Layer:
    layer = Embedding(MAX_WORDS, 200, weights=[
                      embed_matrix], input_length=MAX_LENGTH, trainable=True)(inpt)

    # Convolutional layer:
    layer = Conv1D(128, kernel_size=2, padding="valid",
                   kernel_initializer="he_uniform")(layer)
    # layer = Flatten()(layer)
    # Spatial dropout layer I:
    layer = Dropout(0.3)(layer)

    # Bidirectional RNN (GRU) layer:
    layer = Bidirectional(GRU(100, return_sequences=True))(layer)
    #  Pooling layers:
    avg_pool = GlobalAveragePooling1D()(layer)
    max_pool = GlobalMaxPooling1D()(layer)
    conc = concatenate([avg_pool, max_pool])
    # Dropout layer II:
    conc = Dropout(0.3)(conc)
    # DNN layer:
    den_layer = Dense(64, activation='relu')(conc)
    outp = Dense(num_classes, activation="softmax")(
        den_layer)  # sigmoid; softmax
    # Complete model
    model = Model(inputs=inpt, outputs=outp)
    # Defining the loss function, optimozer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(
        learning_rate=0.001), metrics=[tf.keras.metrics.Precision(thresholds=0.4),
                                       tf.keras.metrics.Recall(thresholds=0.4), tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])  # Adam, Nadam, SGD,  ['categorical_accuracy']
    return model

# %% executionInfo={"elapsed": 19, "status": "ok", "timestamp": 1678558254379, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="d5a0b506"


def get_model4():
    # Input layer:
    inpt = Input(shape=(MAX_LENGTH, ))
    # Emnedding Layer:
    layer = Embedding(MAX_WORDS, 200, weights=[
                      embed_matrix], input_length=MAX_LENGTH, trainable=True)(inpt)

    # Bidirectional RNN (GRU) layer:
    layer = Bidirectional(GRU(100, return_sequences=True))(layer)
    # Convolutional layer:
    layer = Conv1D(128, kernel_size=2, padding="valid",
                   kernel_initializer="he_uniform")(layer)
    #  Pooling layers:
    avg_pool = GlobalAveragePooling1D()(layer)
    # max_pool = GlobalMaxPooling1D()(layer)
   # conc = concatenate([avg_pool, max_pool])

    # DNN layer:
    # den_layer = Dense(64, activation='relu')(conc)
    outp = Dense(num_classes, activation="softmax")(
        avg_pool)  # sigmoid; softmax
    # Complete model
    model = Model(inputs=inpt, outputs=outp)
    # Defining the loss function, optimozer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer=Adam(
        lr=0.001), metrics=['categorical_accuracy'])  # categorical_crossentropy
    return model


# %% executionInfo={"elapsed": 19, "status": "ok", "timestamp": 1678558254379, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="92d51901"
def get_model1():
    # Input layer:
    inpt = Input(shape=(MAX_LENGTH, ))
    # Emnedding Layer:
    # layer = Embedding(MAX_WORDS, 200, weights=[embed_matrix], input_length=MAXLEN, trainable=True)(inpt)

    outp = Dense(num_classes, activation="softmax")(inpt)  # sigmoid; softmax
    # Complete model
    model = Model(inputs=inpt, outputs=outp)
    # Defining the loss function, optimozer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer=Adam(
        lr=0.001), metrics=['categorical_accuracy'])  # categorical_crossentropy
    return model


# %% executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1678558254379, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="616d546a"
def get_model2():
    # Input layer:
    inpt = Input(shape=(MAX_LENGTH, ))
    # Emnedding Layer:
    layer = Embedding(MAX_WORDS, 200, weights=[
                      embed_matrix], input_length=MAX_LENGTH, trainable=True)(inpt)
    # Spatial dropout layer I:
    # layer = SpatialDropout1D(0.3)(layer)
    # Bidirectional RNN (GRU) layer:
    # layer = Bidirectional(GRU(100, return_sequences=True))(layer)
    # Convolutional layer:
    layer = Conv1D(128, kernel_size=2, padding="valid",
                   kernel_initializer="he_uniform")(layer)
    #  Pooling layers:
    avg_pool = GlobalAveragePooling1D()(layer)
    # max_pool = GlobalMaxPooling1D()(layer)
    # conc = concatenate([avg_pool, max_pool])
    # Spatial dropout layer II:
    # conc = Dropout(0.3)(conc)
    # DNN layer:
    # den_layer = Dense(64, activation='relu')(conc)
    outp = Dense(num_classes, activation="softmax")(
        avg_pool)  # sigmoid; softmax
    # Complete model
    model = Model(inputs=inpt, outputs=outp)
    # Defining the loss function, optimozer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
        lr=0.001), metrics=['categorical_accuracy'])  # categorical_crossentropy
    return model


# %% executionInfo={"elapsed": 2196, "status": "ok", "timestamp": 1678558256557, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="63d58b70"

# used_model = get_model_b()
# used_model = get_LSTM_model()
# used_model = get_CNN_LSTM_model()
filepath = COIVD19_Best_Model_path


# /usr/local/lib/python3.8/dist-packages/keras/optimizers/optimizer_v2/adam.py:117: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.

# %% executionInfo={"elapsed": 4510, "status": "ok", "timestamp": 1678558261063, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="CqHDX2YW2E39"
if "google.colab" in sys.modules:
  # !pip install -q -U tensorboard-plugin-profile

    # %% executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1678558261064, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="9y4_m_952Q8P"
    shutil.rmtree("my_logs", ignore_errors=True)


# %% executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1678558261065, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="8xePA3xQ2izk"


def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

# run_logdir = get_run_logdir()


# %% executionInfo={"elapsed": 266, "status": "ok", "timestamp": 1678558261321, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="bf94c3b2"

# tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir, profile_batch=(100, 200))
checkpoint_path = os.path.join(model_file_location, 'checkpoints',
                               f'COVID19-best-model-{num_classes}-classes'+'.weights.epoch-{epoch:02d}-val_loss-{val_loss:.2f}-val_categorical_accuracy-{val_categorical_accuracy:.2f}.hdf5')

# I want to stop using checkpoint_path to save space in gDrive
checkpoint_path = COIVD19_Best_Model_path

model_checkpoint = ModelCheckpoint(
    checkpoint_path, monitor='val_recall', verbose=1, save_best_only=True, mode='max')


early_stop_callback_1 = EarlyStopping(
    monitor='val_loss', mode='min', min_delta=0.001, patience=5, verbose=1)

early_stop_callback_2 = EarlyStopping(
    monitor='val_categorical_accuracy', mode='max', patience=5, verbose=1)


my_callbacks = [model_checkpoint, early_stop_callback_1,
                early_stop_callback_2]  # , tensorboard_cb]


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 43453, "status": "ok", "timestamp": 1678558304771, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="FmR5Sohi25Nc" outputId="c50af260-5789-402b-cc65-6fcf17332ef2"

used_model = get_model(my_metrics)
history = used_model.fit(x=x_train_padded, y=y_train_labels, validation_data=(x_test_padded, y_test_labels), batch_size=128,
                         callbacks=my_callbacks, epochs=20, verbose=1)

# %% id="93Y4a7VX3EGt"
# # %load_ext tensorboard
# # %tensorboard --logdir=./my_logs

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"elapsed": 167, "status": "ok", "timestamp": 1678558350804, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="oE1Wj1g_340W" outputId="fb686eae-ec00-4809-eefd-98fe61af5d98"
# extra code

if "google.colab" in sys.modules:
    from google.colab import output

    output.serve_kernel_port_as_window(6006)
else:
    from IPython.display import display, HTML

    display(HTML('<a href="http://localhost:6006/">http://localhost:6006/</a>'))

# %% executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1678558309480, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="geAmI4_1z_PV"
# freez layers
for layer in used_model.layers[0:-1]:
    layer.trainable = False

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10976, "status": "ok", "timestamp": 1678558320449, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="bfb1ea86" outputId="304e4a2f-83d7-440f-c41f-8db2afaf36e6"

for layer in used_model.layers[0:-1]:
    layer.trainable = False

used_model.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(
                       learning_rate=0.001), metrics=my_metrics)  # Adam, Nadam, SGD, Adamax

history = used_model.fit(x=y_train_covid_padded, y=y_train_covid_labels,
                         validation_data=(
                             x_test_padded, y_test_labels), batch_size=128,
                         callbacks=my_callbacks, epochs=20, verbose=1)

# %% executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1678558320450, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="85d8f970"
# history = used_model.fit(x=padded_train, y=y_train_labels, validation_data=(padded_test, y_test_labels), batch_size=128,
#     callbacks=my_callbacks, epochs=2, verbose=1)

# %% executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1678558320450, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="90009656"
# from sklearn.metrics import accuracy_score

# # Predict class probabilities for test set
# y_prob = used_model.predict(padded_test)

# # Convert probabilities to class labels
# y_pred = np.argmax(y_prob, axis=1)

# # Calculate accuracy for each class
# class_acc = []
# for c in range(num_classes):
#     idx = np.where(y_test_labels.argmax(axis=1) == c)[0]
#     acc = accuracy_score(y_test_labels[idx].argmax(axis=1), y_pred[idx])
#     class_acc.append(acc)

# # Calculate weighted accuracy
# weighted_acc = np.average(class_acc, weights=np.bincount(y_test_labels.argmax(axis=1)))

# weighted_acc

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 298, "status": "ok", "timestamp": 1678558320740, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="54dc9ed9" outputId="a05201bd-b285-4d43-8557-bfaf2722a299"
used_model.summary()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 706, "status": "ok", "timestamp": 1678558321444, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="0812b74a" outputId="a2522172-694c-4d3e-cbd2-0825aed6e3ef"

plot_model(used_model, to_file='model_plot.png',
           show_shapes=True, show_layer_names=True)

# %% executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1678558321446, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="06f6735d"
# #!pip install pydot

# %% executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1678558321447, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="a19fda35"
# used_model = get_model_s()


# history = used_model.fit(x=padded_train, y=y_train_labels, validation_data=(padded_test, y_test_labels), batch_size=128,
#     callbacks=my_callbacks, epochs=20, verbose=1)

# %% executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1678558321447, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="9265394f"
# used_model.add(Dense(3, activation="softmax"))

# last_layer = used_model.get_layer(index=-1)
# last_layer.units = 1

# last_layer.output_shape

# %% executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1678558321448, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="7343e38d"
def create_model(num_layers=1, num_units=100, activation='relu', optimizer='nadam', kernel_initializer='glorot_uniform'):
    # Input layer:
    inpt = Input(shape=(MAX_LENGTH, ))
    # Emnedding Layer:
    layer = Embedding(MAX_WORDS, 200, weights=[
                      embed_matrix], input_length=MAX_LENGTH, trainable=True)(inpt)
    # Spatial dropout layer I:
    layer = SpatialDropout1D(0.3)(layer)
    # Bidirectional RNN (GRU) layer:
    for i in range(num_layers):
        layer = Bidirectional(GRU(num_units, return_sequences=True))(layer)
    # Convolutional layer:
    layer = Conv1D(128, kernel_size=2, padding="valid",
                   kernel_initializer=kernel_initializer)(layer)
    #  Pooling layers:
    avg_pool = GlobalAveragePooling1D()(layer)
    max_pool = GlobalMaxPooling1D()(layer)
    conc = concatenate([avg_pool, max_pool])
    # Dropout layer II:
    conc = Dropout(0.3)(conc)
    # DNN layer:
    den_layer = Dense(64, activation=activation)(conc)
    outp = Dense(num_classes, activation="softmax")(den_layer)
    # Complete model
    model = Model(inputs=inpt, outputs=outp)
    # Defining the loss function, optimizer, and metrics
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer == 'nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    elif optimizer == 'adamax':
        opt = tf.keras.optimizers.Adamax(learning_rate=0.001)
    else:
        raise ValueError('Invalid optimizer')
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['categorical_accuracy'])
    return model

# %% executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1678558321449, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="374b6efd"
# from scikeras.wrappers import KerasClassifier
# # !pip install scikeras[tensorflow]

# %% executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1678558321450, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="168154a2"


# Define the parameter grid

def hyperparameters_tuning():

    param_grid = {
        'num_layers': [1],
        'num_units': [100],
        'activation': ['relu', 'selu', 'elu'],
        'optimizer': ['adam', 'nadam', 'adamax'],
        'kernel_initializer': ['he_uniform', 'glorot_uniform', 'uniform']
    }

    # Create the Keras model
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32)

    # Create the grid search object
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

    # Run the grid search
    grid_result = grid_search.fit(
        x_train_padded, y_train_labels, validation_data=(x_test_padded, y_test_labels))

    # Print the best results
    print('Best score:', grid_result.best_score_)
    print('Best parameters:', grid_result.best_params_)

    # Save the best model
    best_model = grid_result.best_estimator_.model
    best_model.save('best_model_GridSearchCV.h5')
    # Best parameters:  {'activation': 'relu', 'kernel_initializer': 'he_uniform', 'num_layers': 1, 'num_units': 100, 'optimizer': 'adam'}

    return best_model, grid_result


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1678558321450, "user": {"displayName": "Faraj Lagum", "userId": "01113283037435065577"}, "user_tz": 300} id="49c8f407" outputId="9d823380-a1c3-407d-a1b8-0ed4c681c155"
# Print the best parameters and score

if __name__ == '__main__':
    print("hyperparameters tuning ... ")
    # best_model, grid_result = hyperparameters_tuning()
    # print("Best parameters: ", grid_result.best_params_)
    # print("Best score: ", grid_result.best_score_)
    # with open('best_model_hyperparameters.txt', 'w') as f:
    #     f.write(str(grid_result.best_params_))
