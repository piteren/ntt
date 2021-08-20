from defaults import EMB_SHAPE, BPE_STD, MAX_SEQ_LEN

from ptools.neuralmess.get_tf import tf
from ptools.neuralmess.base_elements import my_initializer
from ptools.neuralmess.layers import lay_dense
from ptools.neuralmess.encoders import enc_CNN, enc_TNS


# average of features of all sequence vectors
def avg(name: str=              'avg',
        emb_shape=              EMB_SHAPE,
        trainable_emb: bool=    True,#False,
        verb=                   1,
        **kwargs):

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

        enc_out = enc_CNN(
            input=          feats_lookup,
            n_layers=       6,
            n_filters=      50,
            training_flag=  train_flag_PH)
        feats_enc = enc_out['output']
        if verb>0: print(f' > enc_cnn_out: {enc_out}')

        enc_out = enc_TNS(
            in_seq=         feats_enc,
            seq_out=        False,
            n_blocks=       1,
            n_heads=        1,
            max_seq_len=    MAX_SEQ_LEN)
        feats_enc = enc_out['output']
        if verb>0: print(f' > enc_tns_out: {enc_out}')

        feats_mean = tf.math.reduce_mean(
            input_tensor=   feats_enc,
            axis=           -2)
        feats_max = tf.math.reduce_max(
            input_tensor=   feats_enc,
            axis=           -2)

        feats_mm = feats_enc#tf.concat([feats_mean,feats_max], axis=-1)
        if verb>0: print(f' > feats_mm: {feats_mm}')

        logits = lay_dense(
            input=          feats_mm,
            units=          2,
            activation=     None)
        if verb>0: print(f' > logits: {logits}')

        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = scce(y_true=labels_PH, y_pred=logits)

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