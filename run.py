import pandas as pd
from os import path as osp
import re
import heapq
from math import log
import tqdm
import numpy as np
import argparse


# Argument parsing
# Note: -r selects preprocessed rows, after processing, available rows will decrease
parser = argparse.ArgumentParser('Simple Movie Recommendation')
parser.add_argument('-q', type=str, required=True, help='Query')
parser.add_argument('-r', type=int, default=4000, help='How many rows to read from the movie database (some will be dropped out during preprocessing)')
parser.add_argument('-n', type=int, default=10, help='Number of top recommendations')
args = parser.parse_args()

rows = args.r
query = args.q
n_results = args.n

# Paths
data = './MovieSummaries'
movie_metadata_path = osp.join(data, 'movie.metadata.tsv')
plot_summaries_path = osp.join(data, 'plot_summaries.txt')

# Data Preprocessing
# Extract Files -> Drop Useless data -> Reorganize Data -> Clean Text-Based Data -> merge
# movie_metadata_df: contains movie meta data (name, release data, etc.)
movie_metadata_df = pd.read_csv(movie_metadata_path, sep='\t', header=None, nrows=rows)
movie_metadata_df.drop([1, 4, 7], axis=1, inplace= True)
movie_metadata_df.set_index(0, inplace=True, drop=True)
movie_metadata_df.rename_axis(None, axis=0, inplace=True)
movie_metadata_df.rename(columns={2: 'Name', 3: 'Date', 5: 'Runtime', 6: 'Languages', 8: 'Genre'}, inplace=True)

# plot_summaries_df: contains movie summaries
plot_summaries_df = pd.read_csv(plot_summaries_path, sep='\t', header=None)
plot_summaries_df.set_index(0, inplace=True, drop=True)
plot_summaries_df.rename_axis(None, axis=0, inplace=True)
plot_summaries_df.rename(columns={1: 'Summary'}, inplace=True)

movie_metadata_df = movie_metadata_df.join(plot_summaries_df)
movie_metadata_df.dropna(inplace=True, subset=['Summary'])

n_docs = len(movie_metadata_df) # documents left after drop

def clean(s: str):
    '''
    Extract & Clean data from database
    '''
    ret = []
    for c in s.split('"'):
        if c not in '{}: , ' and '/' not in c:
            ret.append(c)
    return ', '.join(ret)

movie_metadata_df[['Languages', 'Genre']] = movie_metadata_df[['Languages', 'Genre']].map(clean)


def build_BoW(summary: str):
    '''
    Cleans summary and create Bag of Words/Term Frequency
    '''
    bow = {}
    summary = summary.lower()
    words = re.split(r'[ ,{}.]+', summary)
    length = len(words)
    for w in words:
        if w not in bow:
            bow[w] = 0
        bow[w] += 1/length
    if '' in bow:
        bow.pop('')
    return bow

movie_metadata_df['BoW'] = movie_metadata_df.apply(lambda summary: build_BoW(summary['Summary']), axis=1)

def build_df(BoWs: pd.Series):
    '''
    Builds document frequency 
    '''
    
    df = {}
    for bow in BoWs:
        for k in bow.keys():
            if k not in df:
                df[k] = 0
            df[k] += 1
    return df

df = build_df(movie_metadata_df['BoW'])

# char2idx: maps words to the index of final tf-idf vector
char2idx = {}
for idx, word in enumerate(df.keys()):
    char2idx[word] = idx

char_size = len(df) # number of unique words entire corpus

def build_corpus_vector(BoW: dict):
    '''
    Builds a normalized vector for every movie based on tf-idf
    '''
    vec = np.zeros(char_size)
    for word in BoW:
        vec[char2idx[word]] = BoW[word] * np.log(n_docs/(1+df[word]))
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

movie_metadata_df['Vector'] = movie_metadata_df.apply(lambda bow: build_corpus_vector(bow['BoW']), axis=1)


# Terms that doesn't provide any meaning and messes with results
noisy_terms = ['i', 'movies', 'movie', 'film', 'films']

def build_query_vector(BoW: dict):
    '''
    Builds a normalized vector for query
    '''
    vec = np.zeros(char_size)
    for word in BoW.keys():
        if word in df and word not in noisy_terms:
            vec[char2idx[word]] = BoW[word] * log(n_docs/(1+df[word]))
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

# Basic query cleaning
query = query.lower()
query_bow = build_BoW(query)
query_vec = build_query_vector(query_bow)


# Calculates and give results
if np.linalg.norm(query_vec) == 0:
    print("No matches!")
else:
    difference_heap = []
    for i, series in movie_metadata_df.iterrows():
        similarity = np.dot(query_vec, series['Vector'])
        difference_heap.append((-similarity, i))
    heapq.heapify(difference_heap)
    
    for rec in range(n_results):
        movie_id = heapq.heappop(difference_heap)[1]
        movie = movie_metadata_df.loc[movie_id]
        print(f'{rec + 1}. {movie['Name']} \t {'Unknown' if pd.isna(movie['Date']) else movie['Date']} \t {movie['Genre']}')
        