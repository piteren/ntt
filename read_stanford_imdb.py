import os
import random
from typing import Dict, List, Tuple

from defaults import SEED

SAMPLE = Tuple[str,'positive' or 'negative']
SPLITS = ['train', 'test']


# reads stanford imdb data, shuffles
def read_stanford_imdb(
        imbd_dir=   'data/aclImdb',
        seed=       SEED) -> Dict[str,List[SAMPLE]]:
    data = {s: [] for s in SPLITS }
    for split in SPLITS:
        for sent in ['pos', 'neg']:
            c_dir = f'{imbd_dir}/{split}/{sent}'
            files = os.listdir(c_dir)
            for file in files:
                with open(f'{c_dir}/{file}') as f:
                    data[split].append((
                        [line for line in f][0],
                        'positive' if sent == 'pos' else 'negative'))
    random.seed(seed)
    for split in data.keys(): random.shuffle(data[split])
    return data


def report_about_data(data: Dict[str,List[SAMPLE]]):
    print('Data report:')
    for k in data:
        print(f' > got {len(data[k])} samples in {k}')


if __name__ == '__main__':
    data = read_stanford_imdb()
    report_about_data(data)