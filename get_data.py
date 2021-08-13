import bpemb
import numpy as np
import random
from tqdm import tqdm
from typing import Dict, Tuple

from ptools.lipytools.little_methods import r_csv
from ptools.lipytools.plots import histogram


# reads csv data
def read_data(data_file='data/IMDB Dataset.csv') -> Dict[int,Dict[str,str]]:
    rows = r_csv(data_file)[1:]
    rows = rows[:2000] # to limit data size
    dataD = {ix: {
        'text':         d[0],
        'sentiment':    d[1]} for ix,d in enumerate(rows)}
    return dataD

# prepares NN train/test tokenized data
def prep_NN_data(
        dataD: Dict[int,Dict[str,str]],
        vocab_size= 100000,
        emb_width=  100,
        test_split= 0.2,
        seed=       123,
        verb=       1) -> Tuple[Dict[str,list], np.ndarray]:

    # https://github.com/bheinzerling/bpemb
    bpemb_en = bpemb.BPEmb(
        lang=   "en",
        vs=     vocab_size,
        dim=    emb_width)
    if verb>0: print('bpe_tokenization ..')
    data_tokenized = [{
        'tokens':   bpemb_en.encode_ids(dataD[ix]['text']),
        'label':    0 if dataD[ix]['sentiment'] == 'negative' else 1} for ix in tqdm(range(len(dataD)))]
    if verb>1: histogram([len(dt['tokens']) for dt in data_tokenized])

    random.seed(seed)

    random.shuffle(data_tokenized)

    data_split = {'train': [], 'test': []}
    for dt in data_tokenized:
        dd = data_split['test'] if random.random() < test_split else data_split['train']
        dd.append(dt)

    return data_split, bpemb_en.vectors


if __name__ == '__main__':
    dataD = read_data()
    data_split, embeddings = prep_NN_data(dataD, verb=1)
    print(embeddings.shape)
    print(embeddings[0])