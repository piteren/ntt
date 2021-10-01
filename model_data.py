import bpemb
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

from read_stanford_imdb import read_stanford_imdb, SAMPLE, SPLITS, report_about_data
from defaults import EMB_SHAPE, MAX_SEQ_LEN, CACHE_FD

from ptools.lipytools.little_methods import prep_folder, w_pickle
# from ptools.lipytools.plots import histogram - crashes TF to not execute eagerly (something from imports of that file)


NN_DATA = Dict[str,Dict[str,np.array]]

USE_MODELS = {
    'U0':   'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3',
    'U1':   'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3',
    'U2':   'https://tfhub.dev/google/universal-sentence-encoder-large/5',
}


# prepares USE embedded data
def prep_USE_data(
        data: Optional[Dict[str,List[SAMPLE]]]= None,
        use_model: str=                         'U0',
        pack_size=                              'auto') -> NN_DATA:

    if not data: data = read_stanford_imdb()
    report_about_data(data)

    if pack_size == 'auto':
        if use_model == 'U0': pack_size = 100
        if use_model == 'U1': pack_size = 10
        if use_model == 'U2': pack_size = 10

    embed = hub.load(USE_MODELS[use_model])

    npa_splits = {}
    for split in SPLITS:
        sentences = [d[0] for d in data[split]]

        # pack sentences
        sj = []
        sp = []
        for s in sentences:
            sj.append(s)
            if len(sj) == pack_size:
                sp.append(sj)
                sj = []
        if sj: sp.append(sj)

        npa = []
        for p in tqdm(sp):
            emb = embed(p)
            npa.append(np.asarray(emb))
        npa_splits[split] = np.concatenate(npa, axis=0)

    return {s: {
        'embeddings':   npa_splits[s],
        'labels':       [0 if d[1] == 'negative' else 1 for d in data[s]]} for s in SPLITS}

# writes USE embedded data to cache
def build_USE_cache():
    print(tf.executing_eagerly())
    for use_model in USE_MODELS:
        data = prep_USE_data()
        cache_FN = f'data_USE_{use_model}.cache'
        cache_path = f'{CACHE_FD}/{cache_FN}'
        w_pickle(data, cache_path)

# prepares BPE tokenized data with BPE embeddings
def prep_BPE_data(
        data: Optional[Dict[str,List[SAMPLE]]]= None,
        vocab_size=                             EMB_SHAPE[0],
        emb_width=                              EMB_SHAPE[1],
        max_seq_len=                            MAX_SEQ_LEN,
        verb=                                   1) -> Tuple[NN_DATA, np.array]:

    if not data: data = read_stanford_imdb()
    report_about_data(data)

    # https://github.com/bheinzerling/bpemb
    bpemb_en = bpemb.BPEmb(
        lang=   "en",
        vs=     vocab_size,
        dim=    emb_width)

    if verb>1:
        print('bpe samples:')
        for s in data[SPLITS[0]][:20]:
            print(s[0])
            print(bpemb_en.encode(s[0]))
            ids = bpemb_en.encode_ids(s[0])
            print(ids)
            print(bpemb_en.decode_ids(ids))
            print()

    data_tokenized = {}
    for split in SPLITS:
        if verb > 0: print(f'bpe tokenization {split} ..')
        data_tokenized[split] = [(
            bpemb_en.encode_ids(s[0]),
            0 if s[1] == 'negative' else 1) for s in tqdm(data[split])]

    if verb>1:
        #histogram([len(dt[0])for dt in data_tokenized[SPLITS[0]]], name='tokens_len',)
        print(f'STD of bpe vectors: {np.std(bpemb_en.vectors):.2f}')
        for split in SPLITS:
            print(f'labels average {split}: {np.mean([s[1] for s in data_tokenized[split]]):.2f}')


    # cut or pad (while reformat)
    pad_id = vocab_size
    data_splits = {s: {'tokens':[], 'labels':[]} for s in SPLITS}
    for split in SPLITS:
        for dt in data_tokenized[split]:
            tokens = dt[0]
            if len(tokens) > max_seq_len: tokens = tokens[:max_seq_len]
            else: tokens += [pad_id] * (max_seq_len-len(tokens))
            data_splits[split]['tokens'].append(tokens)
            data_splits[split]['labels'].append(dt[1])

    for s in data_splits:
        for d in data_splits[s]:
            data_splits[s][d] = np.asarray(data_splits[s][d])
            if verb>0: print(f'data {s} {d}: {data_splits[s][d].shape}')

    return data_splits, bpemb_en.vectors

# writes USE embedded data to cache
def build_BPE_cache():
    data_splits, embs = prep_BPE_data(verb=2)
    cache_FN = f'data_BPE.cache'
    cache_path = f'{CACHE_FD}/{cache_FN}'
    prep_folder(CACHE_FD)
    w_pickle((data_splits, embs), cache_path)


if __name__ == '__main__':
    build_USE_cache()
    #build_BPE_cache()