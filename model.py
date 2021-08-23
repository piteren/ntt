from defaults import CACHE_FD

from ptools.lipytools.little_methods import r_pickle
from ptools.neuralmess.get_tf import tf
from ptools.neuralmess.nemodel import NEModel


# vector based model (e.g. USE embeddings)
class VecModel(NEModel):

    def load_model_data(self) -> dict:
        cache_FN = f'data_USE_{self["use_model"]}.cache'
        return r_pickle(f'{CACHE_FD}/{cache_FN}', raise_exception=True)

    def build_feed(self, batch:dict, train=True) -> dict:
        return {
            self['embeddings_PH']:  batch['embeddings'],
            self['train_flag_PH']:  train,
            self['labels_PH']:      batch['labels']}

# tokens sequence model (e.g. BPE tokenized)
class SeqModel(NEModel):

    def load_model_data(self) -> dict:
        cache_FN = f'data_BPE.cache'
        data_splits, embs = r_pickle(f'{CACHE_FD}/{cache_FN}', raise_exception=True)
        md = data_splits
        md['embeddings'] = embs
        return md

    def pre_train(self):
        super(SeqModel, self).pre_train()
        if self.verb>0: print(' > loading embeddings var.. ', end='')
        self.session.run(tf.assign(ref=self['emb_var'], value=self.model_data['embeddings']))
        if self.verb>0: print('done!')

    def build_feed(self, batch:dict, train=True) -> dict:
        return {
            self['tokens_PH']:      batch['tokens'],
            self['train_flag_PH']:  train,
            self['labels_PH']:      batch['labels']}