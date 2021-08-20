import bpemb
import numpy as np
import random
from tqdm import tqdm
from typing import Dict, Tuple, Optional

from defaults import EMB_SHAPE, MAX_SEQ_LEN, CACHE_FD

from ptools.lipytools.little_methods import r_csv, r_pickle, w_pickle, prep_folder
from ptools.lipytools.plots import histogram

DO_NOT_READ_CACHE = False#True


# reads csv data
def read_data(data_file='data/IMDB Dataset.csv') -> Dict[int,Dict[str,str]]:
    rows = r_csv(data_file)[1:]
    #rows = rows[:2000] # to limit data size
    dataD = {ix: {
        'text':         d[0],
        'sentiment':    d[1]} for ix,d in enumerate(rows)}
    return dataD

# prepares NN train/test tokenized data
def prep_NN_data(
        dataD: Optional[Dict[int,Dict[str,str]]]=   None,
        vocab_size=                                 EMB_SHAPE[0],
        emb_width=                                  EMB_SHAPE[1],
        max_seq_len=                                MAX_SEQ_LEN,
        valid_split=                                0.0,
        test_split=                                 0.2,
        seed=                                       123,
        verb=                                       1) -> Tuple[Dict[str,dict], np.ndarray]:

    if not dataD: dataD = read_data()

    # https://github.com/bheinzerling/bpemb
    bpemb_en = bpemb.BPEmb(
        lang=   "en",
        vs=     vocab_size,
        dim=    emb_width)
    if verb>0: print('bpe_tokenization ..')
    data_tokenized = [{
        'tokens':   bpemb_en.encode_ids(dataD[ix]['text']),
        'label':    0 if dataD[ix]['sentiment'] == 'negative' else 1} for ix in tqdm(range(len(dataD)))]
    if verb>1:
        histogram([len(dt['tokens']) for dt in data_tokenized])
        print(f'STD of bpe vectors: {np.std(bpemb_en.vectors):.2f}')

    random.seed(seed)

    random.shuffle(data_tokenized)

    # cut or pad
    pad_id = vocab_size
    for dt in data_tokenized:
        if len(dt['tokens']) > max_seq_len: dt['tokens'] = dt['tokens'][:max_seq_len]
        else: dt['tokens'] += [pad_id] * (max_seq_len-len(dt['tokens']))

    data_split = {s: {'tokens': [], 'label': []} for s in ['train','test','valid']}
    for dt in data_tokenized:
        if test_split+valid_split and random.random() < test_split + valid_split:
            if random.random() < test_split / (test_split + valid_split): dd = data_split['test']
            else: dd = data_split['valid']
        else: dd = data_split['train']
        dd['tokens'].append(dt['tokens'])
        dd['label'].append(dt['label'])
    if verb>1: print(f'labels average: {np.mean(data_split["train"]["label"]):.2f}')

    for ka in data_split:
        for kb in data_split[ka]:
            data_split[ka][kb] = np.asarray(data_split[ka][kb])

    return data_split, bpemb_en.vectors

#reads NN data from cache, if not found: prepares and saves
def get_NN_data(cache_FN='ntt_data.cache'):
    cache_path = f'{CACHE_FD}/{cache_FN}'
    nnd = r_pickle(cache_path)
    if not nnd or DO_NOT_READ_CACHE:
        nnd = prep_NN_data()
        prep_folder(CACHE_FD)
        w_pickle(nnd, cache_path)
    return nnd


if __name__ == '__main__':
    data_split, embeddings = prep_NN_data(verb=2)
    print(data_split['train']['tokens'].shape)
    print(data_split['train']['label'].shape)
    print(embeddings.shape)
    print(embeddings[0])