from model_graphs import seq, use
from model import VecModel, SeqModel

presets = {
    'seq': {
        'fwd_func':     seq,
        'model_type':   SeqModel,
        #'make_cnn':     True,
        #'make_tns':     True,
        #'make_tat':     True,
        #'make_avg':     False,
        #'make_max':     False,
    },

    # ************************************************************************** USE models
    'use_base': {
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U1',
        'iLR':          4e-3,
        'psdd': { # done
            'do_clip':      (True, False),
            'iLR':          [1e-5, 3e-2]}
    },
    'use_hidden': {
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U1',
        'make_hidden':  True,
        'hid_width':    768,
        'iLR':          1e-4,
        'do_clip':      True,
        'psdd': { # done
            'hid_width':    [12,1024],
            'do_clip':      (True,False),
            'iLR':          (1e-5,3e-5,
                             1e-4,3e-4,
                             1e-3,3e-3,
                             1e-2,3e-2)}
    },
    'use_drt': {
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U1',

        'make_drt':     True,

        'psdd': {
            'drt_shared':   (True,False),
            'drt_layers':   (2,4,6,8,10,12),
            'drt_lay_width':[12,256],
            'drt_dns_scale':[2,7],
            'drt_drop':     [0.0,0.99],
            'do_clip':      (True,False),
            'iLR':          (1e-6,3e-6,
                             1e-5,3e-5,
                             1e-4,3e-4,
                             1e-3,3e-3,
                             1e-2,3e-2)}
    }
}


def get_preset(preset_name: str) -> dict:

    defaults = {
        'name':         preset_name,
        'n_batches':    10000,
        'batch_size':   128,
        'do_clip':      False,
        'iLR':          3e-4}

    preset = presets[preset_name]
    for k in defaults:
        if k not in preset: preset[k] = defaults[k]
    return preset