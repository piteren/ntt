from defaults import EMB_SHAPE, BPE_STD, MAX_SEQ_LEN

from ptools.neuralmess.get_tf import tf
from ptools.neuralmess.base_elements import my_initializer
from ptools.neuralmess.layers import lay_dense
from ptools.neuralmess.encoders import enc_CNN, enc_TNS, enc_DRT


# sequence graph
def seq(name: str=              'seq',
        emb_shape=              EMB_SHAPE,
        trainable_emb: bool=    False,
        make_cnn=               False,
        make_tns=               False,
        make_tat=               False,
        make_avg=               True,
        make_max=               True,
        verb=                   1,
        **kwargs):

    tf_reduce = make_avg or make_max
    assert (make_tat or tf_reduce) and not (make_tat and tf_reduce), 'ERR: seq reduction configuration not valid!'

    if verb>0: print(f'nn_graph {name} got under kwargs: {kwargs}')

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

        if make_cnn:
            enc_out = enc_CNN(
                input=          feats,
                n_layers=       6,
                n_filters=      50,
                training_flag=  train_flag_PH)
            if verb>0: print(f' > enc_cnn_out: {enc_out}')
            feats = enc_out['output']

        if make_tns:
            enc_out = enc_TNS(
                in_seq=         feats,
                seq_out=        True,
                n_blocks=       6,
                n_heads=        1,
                max_seq_len=    MAX_SEQ_LEN,
                training_flag=  train_flag_PH)
            if verb>0: print(f' > enc_tns_out: {enc_out}')
            feats = enc_out['output']

        if make_tat:
            enc_out = enc_TNS(
                in_seq=         feats,
                seq_out=        False,
                n_blocks=       6,
                n_heads=        1,
                max_seq_len=    MAX_SEQ_LEN,
                training_flag=  train_flag_PH)
            if verb>0: print(f' > enc_tns_out: {enc_out}')
            feats = enc_out['output']

        sum_feats = []
        if make_avg:
            avg = tf.math.reduce_mean(
                input_tensor=   feats,
                axis=           -2)
            if verb>0: print(f' > avg: {avg}')
            sum_feats.append(avg)
        if make_max:
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


def use(name: str=      'use',
        verb=           1,
        # hidden
        make_hidden=    False,
        hid_width=      100,
        # drt
        make_drt=       False,
        drt_shared=     False,
        drt_layers=     6,
        drt_lay_width=  32,
        drt_dns_scale=  6,
        drt_drop=       0.2,
        **kwargs):

    if verb>0: print(f'nn_graph {name} got under kwargs: {kwargs}')

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

        if make_hidden:
            feats = lay_dense(
                input=          embeddings_PH,
                units=          hid_width,
                activation=     tf.nn.relu)
            if verb>0: print(f' > hidden: {feats}')

        if make_drt:
            drt_out = enc_DRT(
                input=          feats,
                shared_lays=    drt_shared,
                n_layers=       drt_layers,
                lay_width=      drt_lay_width,
                dns_scale=      drt_dns_scale,
                dropout=        drt_drop,
                training_flag=  train_flag_PH)
            if verb>0: print(f' > enc_drt_out: {drt_out}')
            feats = drt_out['output']

        logits = lay_dense(input=feats, units=2)
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