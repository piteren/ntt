from copy import deepcopy

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
    'use_base_U0': {
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U0',
        'iLR':          3.83e-3,
        'do_clip':      True,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # 0.8406

    'use_base_U1': {
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U1',
        'iLR':          3.81e-3,
        'do_clip':      True,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # 0.8406

    'use_one_hidden': {
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
            'do_clip':      (True, False),}}, # 0.8423

    'use_hidden_stack': {
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U1',
        'make_hidden':  True,
        'hid_layers':   1,
        'hid_width':    577,
        'hid_dropout':  0.584,
        'iLR':          1.5e-4,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'hid_layers':   [1, 12],
            'hid_width':    [12, 1024],
            'hid_dropout':  [0.0, 0.99],
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # ???

    'use_drt': {
        'fwd_func':         use,
        'model_type':       VecModel,
        'use_model':        'U1',
        'make_drt':         True,
        'drt_shared':       False,
        'drt_layers':       1,
        'drt_lay_width':    55,
        'drt_dns_scale':    4,
        'drt_in_dropout':   0.2,
        'drt_res_dropout':  0.2,
        'drt_lay_dropout':  0.2,
        'iLR':              0.043,
        'do_clip':          True,
        'psdd': {
            'drt_shared':       (True, False),
            'drt_layers':       [1, 6],
            'drt_lay_width':    [12, 768],
            'drt_dns_scale':    [2, 6],
            'drt_in_dropout':   [0.0, 0.99],
            'drt_res_dropout':  [0.0, 0.99],
            'drt_lay_dropout':  [0.0, 0.99],
            'iLR':              [1e-7, 1e-1],
            'do_clip':          (True, False),}}, # 0.8307
}


def get_preset(preset_name: str) -> dict:

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