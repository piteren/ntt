from abc import ABC, abstractmethod

from defaults import CACHE_FD

from ptools.lipytools.little_methods import r_pickle
from ptools.neuralmess.get_tf import tf
from ptools.neuralmess.nemodel import NEModel
from ptools.neuralmess.batcher import Batcher


class NNTModel(ABC, NEModel):

    def __init__(
            self,
            **kwargs):

        ABC.__init__(self)
        NEModel.__init__(
            self,
            load_saver= False,
            **kwargs)

        self.model_data = self.load_model_data()
        self.batcher = Batcher(
            data_TR=    self.model_data['train'],
            data_TS=    self.model_data['test'],
            batch_size= self['batch_size'],
            btype=      'random_cov',
            verb=       self.verb)
        self.test_counter = 0

    @abstractmethod
    def load_model_data(self) -> dict: pass

    # before training method
    @ abstractmethod
    def pre_train(self): pass

    # builds feed dict from given batch of data
    @abstractmethod
    def build_feed(self, batch: dict) -> dict: pass

    def train(
            self,
            test_freq=  100):

        self.load()
        self.pre_train()

        if self.verb>0: print(f'{self["name"]} - training starts')
        batch_IX = 0
        acc_loss = []
        acc_acc = []
        while batch_IX < self['n_batches']:
            batch_IX += 1
            batch = self.batcher.get_batch()
            feed = self.build_feed(batch)
            fetches = [self['optimizer'], self['loss'], self['acc'], self['gg_norm'], self['avt_gg_norm']]
            _, loss, acc, gg_norm, avt_gg_norm = self.session.run(fetches, feed)
            self.log_TB(value=loss,         tag='tr/loss',      step=batch_IX)
            self.log_TB(value=acc,          tag='tr/acc',       step=batch_IX)
            self.log_TB(value=gg_norm,      tag='tr/gn',        step=batch_IX)
            self.log_TB(value=avt_gg_norm,  tag='tr/gn_avt',    step=batch_IX)
            acc_loss.append(loss)
            acc_acc.append(acc)
            if len(acc_loss) == test_freq:
                ts_acc, ts_loss = self.test()
                print(f'{batch_IX:5d} TR: {100*sum(acc_acc)/test_freq:.1f} / {sum(acc_loss)/test_freq:.3f} -- TS: {100*ts_acc:.1f} / {ts_loss:.3f}')
                acc_loss = []
                acc_acc = []


        self.save()

    def test(self):
        batches = self.batcher.get_TS_batches()
        acc_loss = []
        acc_acc = []
        for batch in batches:
            feed = self.build_feed(batch)
            fetches = [self['loss'], self['acc']]
            loss, acc = self.session.run(fetches, feed)
            acc_loss.append(loss)
            acc_acc.append(acc)
        loss = sum(acc_loss)/len(acc_loss)
        acc = sum(acc_acc) /len(acc_acc)
        self.log_TB(value=loss, tag='ts/loss', step=self.test_counter)
        self.log_TB(value=acc,  tag='ts/acc',  step=self.test_counter)
        self.test_counter += 1
        return acc, loss

# vector based model (e.g. USE embeddings)
class VecModel(NNTModel):

    def load_model_data(self) -> dict:
        cache_FN = f'data_USE_{self["use_model"]}.cache'
        return r_pickle(f'{CACHE_FD}/{cache_FN}', raise_exception=True)

    def pre_train(self): pass

    def build_feed(self, batch: dict) -> dict:
        return {
            self['embeddings_PH']:  batch['embeddings'],
            self['train_flag_PH']:  True,
            self['labels_PH']:      batch['labels']}

# tokens sequence model (e.g. BPE tokenized)
class SeqModel(NNTModel):

    def load_model_data(self) -> dict:
        cache_FN = f'data_BPE.cache'
        data_splits, embs = r_pickle(f'{CACHE_FD}/{cache_FN}', raise_exception=True)
        md = data_splits
        md['embeddings'] = embs
        return md

    def pre_train(self):
        if self.verb>0: print(' > loading embeddings var.. ', end='')
        self.session.run(tf.assign(ref=self['emb_var'], value=self.model_data['embeddings']))
        if self.verb>0: print('done!')

    def build_feed(self, batch: dict) -> dict:
        return {
            self['tokens_PH']:      batch['tokens'],
            self['train_flag_PH']:  True,
            self['labels_PH']:      batch['labels']}