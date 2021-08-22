from model_graphs import seq, use
from model import VecModel, SeqModel

presets = {
    'seq': {
        'nn_graph':     seq,
        'model_type':   SeqModel,
        'n_batches':    10000,
        'batch_size':   128,
        'do_clip':      True,
        'iLR':          1e-3#3e-4
    },
    'use': {
        'nn_graph':     use,
        'model_type':   VecModel,
        'use_model':    'U0',
        'n_batches':    10000,
        'batch_size':   128,
        'do_clip':      False,
        'iLR':          3e-4
    }
}


def get_preset(preset_name: str) -> dict:

    defaults = {
        'name':         preset_name}

    preset = presets[preset_name]
    for k in defaults:
        if k not in preset: preset[k] = defaults[k]
    return preset