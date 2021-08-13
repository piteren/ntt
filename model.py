from ptools.neuralmess.nemodel import NEModel


class NNTModel(NEModel):

    def __init__(
            self,
            **kwargs):

        NEModel.__init__(
            self,
            load_saver= False,
            **kwargs)

    def train(self):
        pass

