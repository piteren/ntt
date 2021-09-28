from copy import deepcopy

from model_graphs import seq, use, use_more
from model import VecModel, SeqModel


presets = {
    'seq_reduced': {
        'fwd_func':     seq,
        'model_type':   SeqModel,
        'reduce':       'avg_max',
        'iLR':          1.5e-2,
        'do_clip':      True,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'reduce':       ('avg','max','avg_max'),
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # 0.6884(100) ... 0.7294(500) 0.7259(200) 0.6539(50)

    'seq_cnn': {
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_cnn':         True,
        'batch_size':       128,
        'cnn_shared_lays':  False,
        'cnn_n_layers':     3,
        'cnn_n_filters':    49,
        'cnn_lay_drop':     0.46,
        'iLR':              2.1e-3,
        'do_clip':          True,
        'reduce':           'avg_max',
        'psdd': {
            'batch_size':           (64,128),
            'cnn_shared_lays':      (True, False),
            'cnn_n_layers':         [1, 6],
            'cnn_n_filters':        [12, 96],
            'cnn_lay_drop':         [0.0, 0.99],
            'iLR':                  [1e-7, 1e-1],
            'do_clip':              (True, False)}}, #

    'seq_tns': {
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_tns':         True,
        'batch_size':       128,    # 128-64
        'tns_shared_lays':  False,
        'tns_n_blocks':     4,      # 2-4
        'tns_n_heads':      5,      # all
        'tns_dense_mul':    4,      # 4-5
        'tns_dropout':      0.81,   # >0.5
        #'tns_dropout_att':  0.02,   # 0
        'iLR':              8.8e-5, # e-4 - e-5
        'do_clip':          False,
        'reduce':           'avg_max',
        'psdd': {
            'batch_size':       (64,128,256),#(16,32,64,128,256),
            'tns_shared_lays':  (True, False),
            'tns_n_blocks':     [2, 6],
            'tns_n_heads':      (1, 2, 5),
            'tns_dense_mul':    [2, 6],
            'tns_dropout':      [0.5, 0.99],
            #'tns_dropout_att':  [0.0, 0.99],
            'iLR':              [1e-6, 1e-4],
            'do_clip':          (True, False)}}, # 0.7877(seq 100 with about 400 runs)

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