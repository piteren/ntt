import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from typing import Dict, Optional
from tqdm import tqdm

from read_stanford_imdb import read_stanford_imdb
from defaults import CACHE_FD

from ptools.lipytools.little_methods import w_pickle


USE = {
    'U0':   'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3',
    'U1':   'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
}



def prep_use_data(dataD: Optional[Dict[int,Dict[str,str]]]=None):

    if not dataD: dataD = read_stanford_imdb()
    print(len(dataD))

    embed = hub.load(USE['U0'])

    sentences = [dataD[k]['text'] for k in range(len(dataD))]

    sj = []
    pack = 100
    sp = []
    for s in sentences:
        sj.append(s)
        if len(sj) == pack:
            sp.append(sj)
            sj = []
    if sj: sp.append(sj)

    npa = []
    for p in tqdm(sp):
        npa.append(np.asarray(embed(p)))

    return {
        'embeddings':   np.concatenate(npa, axis=0),
        'label':        np.asarray([0 if dataD[ix]['sentiment'] == 'negative' else 1 for ix in range(len(dataD))])}


if __name__ == '__main__':
    print(tf.executing_eagerly())
    data = prep_use_data()

    cache_FN = 'use_data.cache'
    cache_path = f'{CACHE_FD}/{cache_FN}'
    w_pickle(data, cache_path)


