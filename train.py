from presets import get_preset
from model import NNTModel

if __name__ == '__main__':

    preset_name = 'avg'
    preset = get_preset(preset_name)
    fwd_func = preset.pop('nn_graph')

    model = NNTModel(
        fwd_func=       fwd_func,
        mdict=          preset,
        name_timestamp= True,
        verb=           1)
    model.train()