from defaults import EMB_SHAPE
from nn_graphs import avg


presets = {
    'avg': {
        'nn_graph':     avg,
        'n_batches':    10000,
        'batch_size':   128,
        'do_clip':      True,
        #'iLR':          0.0001,
    }
}


def get_preset(preset_name: str) -> dict:

    defaults = {
        'name':         preset_name}

    preset = presets[preset_name]
    for k in defaults:
        if k not in preset: preset[k] = defaults[k]
    return preset