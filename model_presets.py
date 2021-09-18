from copy import deepcopy

from model_graphs import seq, use, use_more
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
    'use_base_U0': { # USE U0 to logits
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U0',
        'iLR':          3.83e-3,
        'do_clip':      True,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # 0.8406

    'use_base_U1': { # USE U1 to logits
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U1',
        'iLR':          3.81e-3,
        'do_clip':      True,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # 0.8406

    'use_one_hidden': { # USE with one hidden layer (search for width, no drop)
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U1',
        'make_hidden':  True,
        'hid_layers':   1,
        'hid_width':    561,
        'hid_dropout':  0,
        'iLR':          1.3e-4,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'hid_width':    [12, 1024],
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # 0.8428

    'use_hidden_stack': { # USE with MORE hidden layers, allowed dropout
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U1',
        'batch_size':   256,
        'make_hidden':  True,
        'hid_layers':   1,
        'hid_width':    630,
        'hid_dropout':  0.917,
        'iLR':          4.3e-4,
        'do_clip':      True,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'hid_layers':   [1, 12],
            'hid_width':    [12, 1024],
            'hid_dropout':  [0.0, 0.99],
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # 0.8472

    'use_drt': { # USE with DRT encoder
        'fwd_func':         use,
        'model_type':       VecModel,
        'use_model':        'U1',
        'make_drt':         True,
        'drt_shared':       True,
        'drt_layers':       1,
        'drt_lay_width':    574,
        'drt_dns_scale':    4,
        'drt_in_dropout':   0.388,
        'drt_res_dropout':  0.485,
        'drt_lay_dropout':  0.982,
        'iLR':              2.1e-2,
        'psdd': {
            'batch_size':       (16,32,64,128,256),
            'drt_shared':       (True, False),
            'drt_layers':       [1, 6],
            'drt_lay_width':    [12, 768],
            'drt_dns_scale':    [2, 6],
            'drt_in_dropout':   [0.0, 0.99],
            'drt_res_dropout':  [0.0, 0.99],
            'drt_lay_dropout':  [0.0, 0.99],
            'iLR':              [1e-7, 1e-1],
            'do_clip':          (True, False)}}, # 0.8317

    'use_more': { # experiments for DRT replacement
        'fwd_func':         use_more,
        'model_type':       VecModel,
        'use_model':        'U1',
        'do_projection':    False,
        'n_layers':         1,
        'shared_lays':      False,
        'do_norm':          False,
        'play_dropout':     0.655,
        'alay_dropout':     0.334,
        'res_dropout':      0.01,
        'iLR':              1.7e-3,
        'do_clip':          False,
        'psdd': {
            'batch_size':       (16,32,64,128,256),
            'do_projection':    (True, False),
            'proj_width':       [12, 1024],
            'n_layers':         [1, 12],
            'shared_lays':      (True, False),
            'do_norm':          (True, False),
            'play_dropout':     [0.0, 0.99],
            'alay_dropout':     [0.0, 0.99],
            'res_dropout':      [0.0, 0.99],
            'iLR':              [1e-7, 1e-1],
            'do_clip':          (True, False)}}, # ???
}

def get_preset(preset_name: str) -> dict:

    # DO NOT ever change defaults (eventually may add new)
    defaults = {
        'name':         preset_name,
        'n_batches':    10000,
        'batch_size':   128,
        'do_clip':      False,
        'iLR':          3e-4,
        'seed':         123,
        'verb':         0}

    preset = presets[preset_name]
    for k in defaults:
        if k not in preset: preset[k] = defaults[k]
    return deepcopy(preset)