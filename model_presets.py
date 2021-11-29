from copy import deepcopy

from model_graphs import seq, use
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
            'do_clip':      (True, False)}}, # 0.6862(100) ... 0.7294(500) 0.7259(200) 0.6539(50)

    'seq_cnn': {
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_cnn':         True,
        'cnn_shared_lays':  False,
        'cnn_n_layers':     7,
        'cnn_n_filters':    140,
        'cnn_lay_drop':     0.49,
        'iLR':              2.3e-4,
        'do_clip':          False,
        'reduce':           'avg_max',
        'psdd': {
            'cnn_shared_lays':      (True, False),
            'cnn_n_layers':         [2, 9],
            'cnn_n_filters':        [64, 196],
            'cnn_lay_drop':         [0.0, 0.99],
            'iLR':                  [1e-6, 1e-2],
            'do_clip':              (True, False)}}, # 0.8057 #

    'seq_cnn_tf': { # CNN with added TF dropout
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_cnn':         True,
        'time_drop':        0.33,
        'feat_drop':        0.02,
        'cnn_shared_lays':  False,
        'cnn_n_layers':     7,
        'cnn_n_filters':    196,
        'cnn_lay_drop':     0.27,
        'iLR':              2.7e-4,
        'do_clip':          False,
        'reduce':           'avg_max',
        'psdd': {
            'time_drop':            [0.0, 0.9],
            'feat_drop':            [0.0, 0.2],
            'cnn_shared_lays':      (True, False),
            'cnn_n_layers':         [2, 9],
            'cnn_n_filters':        [64, 196],
            'cnn_lay_drop':         [0.0, 0.9],
            'iLR':                  [1e-6, 1e-2],
            'do_clip':              (True, False)}}, # 0.815 #

    'seq_cnn_ind': { # CNN with added input dropout
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_cnn':         True,
        'input_drop':       0.18,
        'cnn_shared_lays':  False,
        'cnn_n_layers':     7,
        'cnn_n_filters':    196,
        'cnn_lay_drop':     0.28,
        'iLR':              2.2e-4,
        'do_clip':          False,
        'reduce':           'avg_max',
        'psdd': {
            'input_drop':           [0.0, 0.5],
            'cnn_shared_lays':      (True, False),
            'cnn_n_layers':         [2, 9],
            'cnn_n_filters':        [64, 196],
            'cnn_lay_drop':         [0.0, 0.9],
            'iLR':                  [1e-6, 1e-2],
            'do_clip':              (True, False)}}, # ??? 0.8104

    'seq_cnn_tf_lay_DRT': { # CNN with added TF dropout and lay_DRT_EX
        'fwd_func':             seq,
        'model_type':           SeqModel,
        'make_cnn':             True,
        'time_drop':            0.17,
        'feat_drop':            0.13,
        'cnn_shared_lays':      False,
        'cnn_n_layers':         5,
        'cnn_n_filters':        186,
        'cnn_lay_drop':         0.28,
        'cnn_do_ldrt':          True,
        'cnn_ldrt_scaled_dns':  True,
        'cnn_ldrt_scale':       4,
        'cnn_ldrt_drop':        0.16,
        'cnn_ldrt_res':         True,
        'cnn_ldrt_res_drop':    0.0,
        'iLR':                  2.7e-4,
        'do_clip':              True,
        'reduce':               'avg_max',
        'psdd': {
            'batch_size':           (64,128),
            'time_drop':            [0.0, 0.99],
            'feat_drop':            [0.0, 0.99],
            'cnn_shared_lays':      (True, False),
            'cnn_n_layers':         [2, 7],
            'cnn_n_filters':        [64, 186],
            'cnn_lay_drop':         [0.0, 0.99],
            'cnn_ldrt_scaled_dns':  (True, False),
            'cnn_ldrt_scale':       [2, 6],
            'cnn_ldrt_drop':        [0.0, 0.99],
            'cnn_ldrt_res':         (True, False),
            'cnn_ldrt_res_drop':    [0.0, 0.99],
            'iLR':                  [1e-7, 1e-1],
            'do_clip':              (True, False)}}, # 0.8104

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
            'do_clip':          (True, False)}}, # 0.7930

    'seq_tns_tf': { # TNS with added TF dropout
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_tns':         True,
        'time_drop':        0.19,
        'feat_drop':        0.12,
        'tns_shared_lays':  True,
        'tns_n_blocks':     7,
        'tns_n_heads':      1,
        'tns_dense_mul':    10,
        'tns_dropout':      0.34,
        'tns_dropout_att':  0.19,
        'iLR':              1.1e-4,
        'do_clip':          False,
        'reduce':           'avg_max',
        'psdd': {
            'time_drop':        [0.1, 0.3], # <<
            'feat_drop':        [0.0, 0.2],
            'tns_shared_lays':  (True, False),
            'tns_n_blocks':     [2, 10],
            'tns_n_heads':      (1, 2, 5, 10),
            'tns_dense_mul':    [2, 10],
            'tns_dropout':      [0.2, 0.5], # <<
            'tns_dropout_att':  [0.1, 0.5],
            'iLR':              [1e-5, 1e-3],
            'do_clip':          (True, False)}}, # 0.7898

    'seq_tns_ind': { # TNS with added input dropout
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_tns':         True,
        'input_drop':       0.14,
        'tns_shared_lays':  False,
        'tns_n_blocks':     8,
        'tns_n_heads':      1,
        'tns_dense_mul':    8,
        'tns_dropout':      0.38,
        'tns_dropout_att':  0.07,
        'iLR':              1.0e-4,
        'do_clip':          True,
        'reduce':           'avg_max',
        'psdd': {
            'input_drop':       [0.1, 0.3],
            'tns_shared_lays':  (True, False),
            'tns_n_blocks':     [2, 10],
            'tns_n_heads':      (1, 2, 5, 10),
            'tns_dense_mul':    [2, 10],
            'tns_dropout':      [0.25, 0.6],
            'tns_dropout_att':  [0.0, 0.2],
            'iLR':              [1e-5, 1e-3],
            'do_clip':          (True, False)}}, # ??? 0.7952

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
            'do_clip':          (True, False)}}, # 0.7800

    'seq_tat_tf': {
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_tat':         True,
        'time_drop':        0.47,
        'feat_drop':        0.015,
        'tat_shared_lays':  True,
        'tat_n_blocks':     13,
        'tat_n_heads':      10,
        'tat_dense_mul':    5,
        'tat_dropout':      0.47,
        'tat_dropout_att':  0.55,
        'iLR':              2.9e-4,
        'do_clip':          True,
        'reduce':           None,
        'psdd': {
            'time_drop':        [0.0, 0.9],
            'feat_drop':        [0.0, 0.2],
            'tat_shared_lays':  (True, False),
            'tat_n_blocks':     [5, 13],
            'tat_n_heads':      (1, 2, 5, 10),
            'tat_dense_mul':    [4, 6],
            'tat_dropout':      [0.0, 0.9],
            'tat_dropout_att':  [0.0, 0.9],
            'iLR':              [1e-5, 1e-3],
            'do_clip':          (True, False)}}, # 0.7993 #

    'seq_tat_ind': {
        'fwd_func':         seq,
        'model_type':       SeqModel,
        'make_tat':         True,
        'input_drop':       0.14,
        'tat_shared_lays':  True,
        'tat_n_blocks':     13,
        'tat_n_heads':      10,
        'tat_dense_mul':    4,
        'tat_dropout':      0.72,
        'tat_dropout_att':  0.55,
        'iLR':              3.6e-4,
        'do_clip':          True,
        'reduce':           None,
        'psdd': {
            'input_drop':       [0.0, 0.9],
            'tat_shared_lays':  (True, False),
            'tat_n_blocks':     [5, 13],
            'tat_n_heads':      (1, 2, 5, 10),
            'tat_dense_mul':    [2, 6],
            'tat_dropout':      [0.0, 0.9],
            'tat_dropout_att':  [0.0, 0.9],
            'iLR':              [1e-5, 1e-3],
            'do_clip':          (True, False)}}, # 0.7977 #

    'use_base_U0': { # USE U0 to logits
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U0',
        'iLR':          3.83e-3,
        'do_clip':      True,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # 0.8395

    'use_base_U1': { # USE U1 to logits
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U1',
        'iLR':          3.81e-3,
        'do_clip':      True,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # 0.8395

    'use_base_U2': { # USE U2 to logits
        'fwd_func':     use,
        'model_type':   VecModel,
        'use_model':    'U2',
        'iLR':          4.3e-3,
        'do_clip':      False,
        'psdd': {
            'batch_size':   (16,32,64,128,256),
            'iLR':          [1e-7, 1e-1],
            'do_clip':      (True, False)}}, # 0.8394

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
            'do_clip':      (True, False)}}, # 0.8416

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
            'do_clip':      (True, False)}}, # 0.8455
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