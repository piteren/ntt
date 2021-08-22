from get_use_data import prep_use_data
from defaults import CACHE_FD

from ptools.neuralmess.nemodel import NEModel
from ptools.neuralmess.batcher import Batcher
from ptools.lipytools.little_methods import r_pickle, w_pickle, prep_folder


class NNTModel(NEModel):

    def __init__(
            self,
            **kwargs):

        NEModel.__init__(
            self,
            load_saver= False,
            **kwargs)

    def train(self):

        cache_FN = 'use_data.cache'
        data = r_pickle(f'{CACHE_FD}/{cache_FN}')
        batcher = Batcher(
                data=       data,
                batch_size= self['batch_size'],
                btype=      'random',
                verb=       self.verb)

        self.load()

        if self.verb>0: print('training starts')
        batch_IX = 0
        acc_loss = []
        acc_acc = []
        while batch_IX < self['n_batches']:
            batch = batcher.get_batch()
            feed = {
                self['embeddings_PH']:  batch['embeddings'],
                self['train_flag_PH']:  True,
                self['labels_PH']:      batch['label']}
            fetches = [self['optimizer'], self['loss'], self['acc'], self['gg_norm'], self['avt_gg_norm']]
            _, loss, acc, gg_norm, avt_gg_norm = self.session.run(fetches, feed)
            self.log_TB(value=loss,         tag='tr/loss',      step=batch_IX)
            self.log_TB(value=acc,          tag='tr/acc',       step=batch_IX)
            self.log_TB(value=gg_norm,      tag='tr/gn',        step=batch_IX)
            self.log_TB(value=avt_gg_norm,  tag='tr/gn_avt',    step=batch_IX)
            acc_loss.append(loss)
            acc_acc.append(acc)
            if len(acc_loss) == 100:
                print(f'{sum(acc_loss)/100:.4f} {sum(acc_acc)/100:.4f} {gg_norm:.4f} {avt_gg_norm:.4f}')
                acc_loss = []
                acc_acc = []
            batch_IX += 1

        self.save()


