from get_data import get_NN_data

from ptools.neuralmess.get_tf import tf
from ptools.neuralmess.nemodel import NEModel
from ptools.neuralmess.batcher import Batcher


class NNTModel(NEModel):

    def __init__(
            self,
            **kwargs):

        NEModel.__init__(
            self,
            load_saver= False,
            **kwargs)

        self.data_split, self.embeddings = None, None
        self.batcher = None

    def load_NN_data(self):
        self.data_split, self.embeddings = get_NN_data()

    def train(self):

        if not self.data_split or not self.embeddings:
            self.load_NN_data()
            self.batcher = Batcher(
                data=       self.data_split['train'],
                batch_size= self['batch_size'],
                verb=       self.verb)

        self.load()
        #self.session.run(tf.assign(ref=self['emb_var'], value=self.embeddings))

        if self.verb>0: print('training starts')
        batch_IX = 0
        acc_loss = []
        acc_acc = []
        while batch_IX < self['n_batches']:
            batch = self.batcher.get_batch()
            feed = {
                self['tokens_PH']:      batch['tokens'],
                self['train_flag_PH']:  True,
                self['labels_PH']:      batch['label']}
            fetches = [self['loss'], self['acc'], self['gg_norm'], self['avt_gg_norm']]
            loss, acc, gg_norm, avt_gg_norm = self.session.run(fetches, feed)
            acc_loss.append(loss)
            acc_acc.append(acc)
            if len(acc_loss) == 100:
                print(f'{sum(acc_loss)/100:.4f} {sum(acc_acc)/100:.4f} {gg_norm:.4f} {avt_gg_norm:.4f}')
                acc_loss = []
                acc_acc = []
            batch_IX += 1

        self.save()


