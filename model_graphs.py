from defaults import EMB_SHAPE, BPE_STD, MAX_SEQ_LEN

from ptools.neuralmess.get_tf import tf
from ptools.neuralmess.base_elements import my_initializer
from ptools.neuralmess.layers import lay_dense, tf_drop
from ptools.neuralmess.encoders import enc_CNN, enc_TNS, enc_DRT
from ptools.neuralmess.encoders_EX import lay_DRT_EX


# sequence graph
def seq(name: str=              'seq',
        emb_shape=              EMB_SHAPE,
        trainable_emb: bool=    False,

        time_drop=              0.0,
        feat_drop=              0.0,

        make_cnn=               False,
        cnn_shared_lays=        False,
        cnn_n_layers :int=      12,
        cnn_kernel :int=        3,
        cnn_n_filters :int=     128,
        cnn_lay_drop=           0.0,
        cnn_ldrt_scale=         0,
        cnn_ldrt_res_drop=      0.0,
        cnn_ldrt_lay_drop=      0.0,

        make_tns=               False,
        tns_shared_lays=        False,
        tns_n_blocks=           6,
        tns_n_heads=            1,
        tns_dense_mul=          4,
        tns_dropout=            0.0,
        tns_dropout_att=        0.0,

        make_tat=               False,
        tat_n_blocks=           6,
        tat_n_heads=            1,
        tat_dense_mul=          4,
        tat_dropout=            0.0,
        tat_dropout_att=        0.0,

        reduce=                 'avg_max', # valid are: None,'avg','max','avg_max'
        seed=                   123,
        verb=                   1):

    tf_reduce = reduce is not None
    assert (make_tat or tf_reduce) and not (make_tat and tf_reduce), 'ERR: seq reduction configuration not valid!'

    tokens_PH = tf.compat.v1.placeholder( # tokens placeholder (seq input - IDs)
        name=   'tokens_PH',
        dtype=  tf.int32,
        shape=  (None,None)) # (batch_size,seq_len)
    train_flag_PH = tf.compat.v1.placeholder(
        name=   'train_flag_PH',
        dtype=  tf.bool)
    labels_PH = tf.compat.v1.placeholder(
        name=   'labels_PH',
        dtype=  tf.int32)

    with tf.variable_scope(name):

        emb_var = tf.compat.v1.get_variable(
            name=           'emb_var',
            shape=          emb_shape,
            initializer=    my_initializer(stddev=BPE_STD),
            trainable=      trainable_emb)
        pad_var = tf.compat.v1.get_variable(
            name=           'pad_var',
            shape=          (1,emb_shape[1]),
            initializer=    my_initializer(stddev=BPE_STD),
            trainable=      True)
        emb_var_conc = tf.concat([emb_var,pad_var], axis=0)
        if verb>0: print(f' > emb_var_conc: {emb_var_conc}')

        feats_lookup = tf.nn.embedding_lookup(
            params=         emb_var_conc,
            ids=            tokens_PH)
        if verb>0: print(f' > feats_lookup: {feats_lookup}')
        feats = feats_lookup

        if time_drop or feat_drop:
            feats = tf_drop(
                input=      feats,
                time_drop=  time_drop,
                feat_drop=  feat_drop,
                train_flag= train_flag_PH,
                seed=       seed)

        if make_cnn:
            enc_out = enc_CNN(
                input=          feats,
                shared_lays=    cnn_shared_lays,
                n_layers=       cnn_n_layers,
                kernel=         cnn_kernel,
                n_filters=      cnn_n_filters,
                lay_drop=       cnn_lay_drop,
                ldrt_scale=     cnn_ldrt_scale,
                ldrt_res_drop=  cnn_ldrt_res_drop,
                ldrt_lay_drop=  cnn_ldrt_lay_drop,
                training_flag=  train_flag_PH,
                seed=           seed,
                n_hist=         0)
            if verb>0: print(f' > enc_cnn_out: {enc_out}')
            feats = enc_out['output']

        if make_tns:
            enc_out = enc_TNS(
                in_seq=         feats,
                seq_out=        True,
                shared_lays=    tns_shared_lays,
                n_blocks=       tns_n_blocks,
                n_heads=        tns_n_heads,
                dense_mul=      tns_dense_mul,
                dropout=        tns_dropout,
                dropout_att=    tns_dropout_att,
                max_seq_len=    MAX_SEQ_LEN,
                training_flag=  train_flag_PH,
                seed=           seed,
                n_hist=         0)
            if verb>0: print(f' > enc_tns_out: {enc_out}')
            feats = enc_out['output']

        if make_tat:
            enc_out = enc_TNS(
                name=           'enc_TAT',
                in_seq=         feats,
                seq_out=        False,
                n_blocks=       tat_n_blocks,
                n_heads=        tat_n_heads,
                dense_mul=      tat_dense_mul,
                dropout=        tat_dropout,
                dropout_att=    tat_dropout_att,
                max_seq_len=    MAX_SEQ_LEN,
                training_flag=  train_flag_PH,
                seed=           seed,
                n_hist=         0)
            if verb>0: print(f' > enc_tns_out: {enc_out}')
            feats = enc_out['output']

        sum_feats = []
        if reduce in ['avg','avg_max']:
            avg = tf.math.reduce_mean(
                input_tensor=   feats,
                axis=           -2)
            if verb>0: print(f' > avg: {avg}')
            sum_feats.append(avg)
        if reduce in ['max','avg_max']:
            max = tf.math.reduce_max(
                input_tensor=   feats,
                axis=           -2)
            if verb>0: print(f' > max: {max}')
            sum_feats.append(max)
        feats_mm = tf.concat(sum_feats, axis=-1) if sum_feats else feats
        if verb>0: print(f' > feats_mm: {feats_mm}')

        logits = lay_dense(input=feats_mm, units=2)
        if verb>0: print(f' > logits: {logits}')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_PH)
        loss = tf.reduce_mean(loss)

        preds = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(labels_PH,preds), dtype=tf.float32))

    oth_vars = tf.compat.v1.trainable_variables(scope=name)
    if emb_var in oth_vars: oth_vars.remove(emb_var)

    return {
        'tokens_PH':        tokens_PH,
        'train_flag_PH':    train_flag_PH,
        'labels_PH':        labels_PH,
        'oth_vars':         oth_vars,
        'emb_var':          emb_var,
        'logits':           logits,
        'loss':             loss,
        'acc':              acc}


def use(name: str=          'use',
            # hidden
        hid_layers=         0,
        hid_width=          100,
        hid_dropout=        0.2,
            # drt
        drt_layers=         0,
        drt_shared=         False,
        drt_lay_width=      32,
        drt_dns_scale=      6,
        drt_in_dropout=     0.2,
        drt_res_dropout=    0.2,
        drt_lay_dropout=    0.2,
        seed=               123,
        verb=               1):

    embeddings_PH = tf.placeholder( # use embeddings placeholder
        name=   'embeddings_PH',
        dtype=  tf.float32,
        shape=  (None,512)) # (batch_size,512)
    if verb>0: print(f' > embeddings_PH: {embeddings_PH}')
    train_flag_PH = tf.placeholder(
        name=   'train_flag_PH',
        dtype=  tf.bool)
    labels_PH = tf.placeholder(
        name=   'labels_PH',
        dtype=  tf.int32,
        shape=  None)
    if verb>0: print(f' > labels_PH: {labels_PH}')
    feats = embeddings_PH

    with tf.variable_scope(name):

        for lay in range(hid_layers):

            feats = lay_dense(
                input=          feats,
                units=          hid_width,
                activation=     tf.nn.relu,
                seed=           seed)
            if verb>0: print(f' > {lay} hidden: {feats}')

            if hid_dropout:
                feats = tf.layers.dropout(
                    inputs=     feats,
                    rate=       hid_dropout,
                    training=   train_flag_PH,
                    seed=       seed)

        if drt_layers:
            drt_out = enc_DRT(
                input=          feats,
                shared_lays=    drt_shared,
                n_layers=       drt_layers,
                lay_width=      drt_lay_width,
                dns_scale=      drt_dns_scale,
                in_dropout=     drt_in_dropout,
                res_dropout=    drt_res_dropout,
                lay_dropout=    drt_lay_dropout,
                training_flag=  train_flag_PH,
                seed=           seed)
            if verb>0: print(f' > enc_drt_out: {drt_out}')
            feats = drt_out['output']

        logits = lay_dense(
            input=  feats,
            units=  2,
            seed=   seed)
        if verb>0: print(f' > logits: {logits}')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_PH)
        loss = tf.reduce_mean(loss)

        preds = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(labels_PH,preds), dtype=tf.float32))

    return {
        'embeddings_PH':    embeddings_PH,
        'train_flag_PH':    train_flag_PH,
        'labels_PH':        labels_PH,
        'logits':           logits,
        'loss':             loss,
        'acc':              acc}


def use_more(
        name: str=      'use_more',
        do_projection=  False,
        proj_width=     256,
        proj_drop=      0.0,
        n_layers=       1,
        shared_lays=    False,
        do_scaled_dns=  False,
        dns_scale=      4,
        lay_dropout=    0.84,
        res_dropout=    0.02,
        seed=           123,
        verb=           1):

    embeddings_PH = tf.placeholder( # use embeddings placeholder
        name=   'embeddings_PH',
        dtype=  tf.float32,
        shape=  (None,512)) # (batch_size,512)
    if verb>0: print(f' > embeddings_PH: {embeddings_PH}')
    train_flag_PH = tf.placeholder(
        name=   'train_flag_PH',
        dtype=  tf.bool)
    labels_PH = tf.placeholder(
        name=   'labels_PH',
        dtype=  tf.int32,
        shape=  None)
    if verb>0: print(f' > labels_PH: {labels_PH}')
    feats = embeddings_PH

    output = feats
    with tf.variable_scope(name):

        if do_projection:
            output = tf.keras.layers.LayerNormalization(axis=-1)(output)
            output = lay_dense(
                input=          output,
                units=          proj_width,
                seed=           seed)
            if proj_drop:
                output = tf.layers.dropout(
                    inputs=     output,
                    rate=       proj_drop,
                    training=   train_flag_PH,
                    seed=       seed)

        for lay in range(n_layers):
            lay_name = f'lay_shared' if shared_lays else f'lay_{lay}'
            with tf.variable_scope(name_or_scope=lay_name, reuse=tf.AUTO_REUSE):
                drt_out = lay_DRT_EX(
                    input=          output,
                    name=           lay_name,
                    hist_name=      'drt_EX',
                    do_scaled_dns=  do_scaled_dns,
                    dns_scale=      dns_scale,
                    lay_dropout=    lay_dropout,
                    res_dropout=    res_dropout,
                    training_flag=  train_flag_PH,
                    seed=           seed)
                output = drt_out['output']

        logits = lay_dense(
            input=  output,
            units=  2,
            seed=   seed)
        if verb>0: print(f' > logits: {logits}')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_PH)
        loss = tf.reduce_mean(loss)

        preds = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(labels_PH,preds), dtype=tf.float32))

    return {
        'embeddings_PH':    embeddings_PH,
        'train_flag_PH':    train_flag_PH,
        'labels_PH':        labels_PH,
        'logits':           logits,
        'loss':             loss,
        'acc':              acc}