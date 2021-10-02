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
        'cnn_shared_lays':  False,
        'cnn_n_layers':     5,
        'cnn_n_filters':    152,
        'cnn_lay_drop':     0.42,
        'iLR':              8e-5,
        'do_clip':          False,
        'reduce':           'avg_max',
        'psdd': {
            'batch_size':           (64,128),
            'time_drop':            [0.0, 0.99],
            'feat_drop':            [0.0, 0.99],
            'cnn_shared_lays':      (True, False),
            'cnn_n_layers':         [2, 7],
            'cnn_n_filters':        [64, 186],
            'cnn_lay_drop':         [0.0, 0.99],
            'iLR':                  [1e-7, 1e-1],
            'do_clip':              (True, False)}}, # 0.8054

    'seq_cnn_tf': { # CNN with added TF dropout
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_cnn':         True,
        'time_drop':        0.30,
        'feat_drop':        0.01,
        'cnn_shared_lays':  False,
        'cnn_n_layers':     4,
        'cnn_n_filters':    170,
        'cnn_lay_drop':     0.29,
        'iLR':              4.6e-4,
        'do_clip':          True,
        'reduce':           'avg_max',
        'psdd': {
            'batch_size':           (64,128),
            'time_drop':            [0.0, 0.99],
            'feat_drop':            [0.0, 0.99],
            'cnn_shared_lays':      (True, False),
            'cnn_n_layers':         [2, 7],
            'cnn_n_filters':        [64, 186],
            'cnn_lay_drop':         [0.0, 0.99],
            'iLR':                  [1e-7, 1e-1],
            'do_clip':              (True, False)}}, # 0.8157

    'seq_cnn_ind': { # CNN with added input dropout
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_cnn':         True,
        'input_drop':       0.0,
        'cnn_shared_lays':  False,
        'cnn_n_layers':     4,
        'cnn_n_filters':    170,
        'cnn_lay_drop':     0.29,
        'iLR':              4.6e-4,
        'do_clip':          True,
        'reduce':           'avg_max',
        'psdd': {
            'batch_size':           (64,128),
            'input_drop':           [0.0, 0.99],
            'cnn_shared_lays':      (True, False),
            'cnn_n_layers':         [2, 7],
            'cnn_n_filters':        [64, 186],
            'cnn_lay_drop':         [0.0, 0.99],
            'iLR':                  [1e-7, 1e-1],
            'do_clip':              (True, False)}}, # ???

    'seq_tns': {
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_tns':         True,
        'tns_shared_lays':  False,
        'tns_n_blocks':     6,
        'tns_n_heads':      5,
        'tns_dense_mul':    6,
        'tns_dropout':      0.52,
        'iLR':              8.3e-5,
        'do_clip':          False,
        'reduce':           'avg_max',
        'psdd': {
            'tns_shared_lays':  (True, False),
            'tns_n_blocks':     [2, 10],
            'tns_n_heads':      (1, 2, 5, 10),
            'tns_dense_mul':    [2, 10],
            'tns_dropout':      [0.3, 0.8],
            'tns_dropout_att':  [0.0, 0.2],
            'iLR':              [1e-7, 1e-3],
            'do_clip':          (True, False)}}, # 0.7869

    'seq_tns_tf': { # TNS with added TF dropout
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_tns':         True,
        'tns_shared_lays':  False,
        'tns_n_blocks':     6,
        'tns_n_heads':      5,
        'tns_dense_mul':    6,
        'tns_dropout':      0.52,
        'iLR':              8.3e-5,
        'do_clip':          False,
        'reduce':           'avg_max',
        'psdd': {
            'time_drop':        [0.0, 0.99],
            'feat_drop':        [0.0, 0.99],
            'tns_shared_lays':  (True, False),
            'tns_n_blocks':     [2, 10],
            'tns_n_heads':      (1, 2, 5, 10),
            'tns_dense_mul':    [2, 10],
            'tns_dropout':      [0.3, 0.8],
            'tns_dropout_att':  [0.0, 0.2],
            'iLR':              [1e-7, 1e-3],
            'do_clip':          (True, False)}}, # ???

    'seq_tat': {
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_tat':         True,
        'tat_shared_lays':  True,
        'tat_n_blocks':     3,
        'tat_n_heads':      5,
        'tat_dense_mul':    3,
        'tat_dropout':      0.75,
        'tat_dropout_att':  0.27,
        'iLR':              3.9e-4,
        'do_clip':          True,
        'reduce':           None,
        'psdd': {
            'batch_size':       (16,32,64,128),
            'tat_shared_lays':  (True, False),
            'tat_n_blocks':     [1, 7],
            'tat_n_heads':      (1, 2, 5, 10),
            'tat_dense_mul':    [2, 6],
            'tat_dropout':      [0.0, 0.99],
            'tat_dropout_att':  [0.0, 0.99],
            'iLR':              [1e-7, 1e-1],
            'do_clip':          (True, False)}}, # 0.7796

    # ************************************************************************** USE based models
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

    'use_base_U2': { # USE U2 to logits
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U2',
        'iLR':          4.3e-3,
        'do_clip':      False,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # 0.8396

    'use_one_hidden': { # USE with one hidden layer (search for width, no drop)
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U1',
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

    'use_more': { # experiments for DRT replacement with something better
        'fwd_func':         use_more,
        'model_type':       VecModel,
        'use_model':        'U1',
        'do_projection':    False,
        'n_layers':         1,
        'shared_lays':      False,
        'do_scaled_dns':    False,
        'lay_dropout':      0.84,
        'res_dropout':      0.02,
        'iLR':              8.3e-5,
        'do_clip':          False,
        'psdd': {
            'batch_size':       (16,32,64,128,256),
            'do_projection':    (True, False),
            'proj_width':       [12, 1024],
            'proj_drop':        [0.0, 0.99],
            'n_layers':         [1, 12],
            'shared_lays':      (True, False),
            'do_scaled_dns':    (True, False),
            'dns_scale':        [1, 6],
            'lay_dropout':      [0.0, 0.99],
            'res_dropout':      [0.0, 0.99],
            'iLR':              [1e-7, 1e-1],
            'do_clip':          (True, False)}}, # 0.8455
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