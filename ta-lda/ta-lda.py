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

# %%
import warnings
import pyLDAvis.gensim
import gensim
import time
import re
import string
import nltk
import pandas as pd
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os
import glob


# %%
warnings.filterwarnings(
    "ignore", message="In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only.")

# %%
results_dir = './results/'
data_dir = './data'

# %%
# from get_csv_files import get_csv_files

# folders_and_files = get_csv_files(data_dir)

# print(folders_and_files.keys())

# for val in folders_and_files.values():
#     for f in val:
#         print(f,)

# %%


# demo_file_name = demo_data_dir + "05_Week4_May_2021_COVID19_Ottawa.csv"
demo_file_name = './data/months/01_January_2021_COVID19_Ottawa.csv'

file_names = glob.glob(data_dir + "/*/*.csv")
# print(file_names)

# %%


def get_data_df(demo_file_name):
    documents_df = pd.read_csv(demo_file_name, on_bad_lines='skip')
# documents_df['text_clean_textual'] = documents_df['text_clean_textual'].astype(str)
# print(documents_df.head(3))
# print(documents_df['Article'])

    return documents_df


documents_df = get_data_df(demo_file_name)
documents_df.head(3)

# %%


# text processing
# def initial_clean(text):
#     text = re.sub("[^a-zA-Z ]", "", text)
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#     return text

def initial_clean(text):
    text = str(text)
    if isinstance(text, str) and text.strip():
        text = re.sub("[^a-zA-Z ]", "", text)
        text = text.lower()
    text = nltk.word_tokenize(text)
    return text


stop_words = stopwords.words('english')
stop_words.extend(['say', 'use', 'not', 'would', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'took', 'per', 'cent', 'done', 'try', 'many', 'some', 'see', 'rather',
                   'lot', 'lack', 'make', 'want', 'seem', 'even', 'also', 'may', 'take',
                   'come', 'new', 'said', 'like', 'de'])


def remove_stop_words(text):
    return [word for word in text if word not in stop_words]


stemmer = PorterStemmer()


def stem_words(text):
    try:
        text = [stemmer.stem(word) for word in text]
        # no single letter words
        text = [word for word in text if len(word) > 1]
    except IndexError:
        pass
    return text


def pre_processing(text):
    return stem_words(remove_stop_words(initial_clean(text)))


# %%
# clean documents and create new column "tokenized"
t1 = time.time()
documents_df['tokenized_documents'] = documents_df['text_clean_textual'].apply(
    pre_processing)
t2 = time.time()
# Time to clean and tokenize 3209 documents: 0.21254388093948365 min
print("Time to clean and tokenize", len(
    documents_df), "documents:", (t2-t1)/60, "min")

# LDA


# %%
# documents_df


# %%

tokenized = documents_df['tokenized_documents']

# Create a Gensim dictionary from the tokenized corpus
dictionary = corpora.Dictionary(tokenized)

# Filter out terms that occur in less than 5 document and more than 80% of the documents.
dictionary.filter_extremes(no_below=5, no_above=0.8)

# Greate a bag of words
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]

# print(corpus[:1])

# %%
# Create LDA model

NUM_TOPICS = 5

lda_model = gensim.models.ldamodel.LdaModel(
    corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15, update_every=2,  iterations=5)

# Save model
lda_model.save('model_combined.gensim')


# %%
topics = lda_model.print_topics(num_words=6)
for topic in topics:
    print(topic)

# %%
get_document_topics = lda_model.get_document_topics(corpus[1])

print(get_document_topics)


# %%
for i, topic in lda_model.show_topics(num_topics=NUM_TOPICS, num_words=10, formatted=False):
    print('Topic {}:'.format(i))
    for word, prob in topic:
        print('\t{}'.format(word))

# %%
# visualizing topics
lda_viz = gensim.models.ldamodel.LdaModel.load('model_combined.gensim')

lda_display = pyLDAvis.gensim.prepare(
    lda_viz, corpus, dictionary, sort_topics=True)


# replace
demo_file_name = demo_file_name.replace("data", "results")


pyLDAvis.save_html(lda_display, demo_file_name + '.html')


# %%
# get one function


def topic_modelling(file_name, num_topics=5):

    documents_df = get_data_df(file_name)
    documents_df['tokenized_documents'] = documents_df['text_clean_textual'].apply(
        pre_processing)
    documents_df.dropna(inplace=True)
    tokenized = documents_df['tokenized_documents']
    # Create a Gensim dictionary from the tokenized corpus
    dictionary = corpora.Dictionary(tokenized)
    # Filter out terms that appear in less than 5 document
    # and more than 80% of the documents.
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    # convert the dictionary to a bag of words corpus
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
    # Create LDA model
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    # change the directory name to save the results in the results dir
    file_name = file_name.replace("data", "results")
    file_name = file_name.replace(".csv", "_")
    # Save model
    lda_model.save(file_name + 'model_combined.gensim')
    # visualizing topics
    lda_viz = gensim.models.ldamodel.LdaModel.load(
        file_name + 'model_combined.gensim')
    lda_display = pyLDAvis.gensim.prepare(
        lda_viz, corpus, dictionary, sort_topics=True)
    # Save visualizations
    pyLDAvis.save_html(lda_display, file_name + '.html')


# %%

# apply to all files
NUM_TOPICS = 5


for file_name in file_names:
    print(file_name)

    topic_modelling(file_name, NUM_TOPICS)

# %%
