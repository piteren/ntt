from ptools.neuralmess.get_tf import tf
from ptools.neuralmess.layers import lay_dense


# average of features of all sequence vectors
def avg(name: str=              'avg',
        emb_shape=              (1000,100),
        trainable_emb: bool=    False,
        out_activ: bool=        True,
        verb=                   1,
        **kwargs):

    if verb>0: print(f'nn_graph {name} got under kwargs: {kwargs}')

    tokens_PH = tf.compat.v1.placeholder( # tokens placeholder (seq input - IDs)
        name=   'tokens_PH',
        dtype=  tf.int32,
        shape=  (None,None)) # (batch_size,seq_len)
    embeddings_PH = tf.compat.v1.placeholder(
        name=   'embeddings_PH',
        dtype=  tf.float32,
        shape=  emb_shape)
    train_flag_PH = tf.compat.v1.placeholder(
        name=   'train_flag_PH',
        dtype=  tf.bool)
    labels_PH = tf.compat.v1.placeholder(
        name=   'labels_PH',
        dtype=  tf.int32)

    with tf.variable_scope(name):

        emb_var = tf.compat.v1.get_variable(
            name=           'emb_var',
            initializer=    embeddings_PH,
            trainable=      trainable_emb)

        feats = tf.nn.embedding_lookup(
            params=         emb_var,
            ids=            tokens_PH)
        feats = tf.math.reduce_mean(
            input_tensor=   feats,
            axis=           -2)

        logits = lay_dense(
            input=          feats,
            units=          2,
            activation=     tf.nn.relu if out_activ else None)

        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = scce(
            y_true= labels_PH,
            y_pred= logits)

        return {
            'tokens_PH':        tokens_PH,
            'embeddings_PH':    embeddings_PH,
            'train_flag_PH':    train_flag_PH,
            'logits':           logits,
            'loss':             loss}