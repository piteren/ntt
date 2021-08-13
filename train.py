from nn_graphs import avg
from model import NNTModel


if __name__ == '__main__':
    model = NNTModel(
        fwd_func=       avg,
        name_timestamp= True,
        verb=           1)
    model.train()