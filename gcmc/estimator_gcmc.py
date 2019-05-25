from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# TODO: params? what parameters do we need?

#
# # for testing
# params = {'hidden units': [1, 1],
#           'dropout': 0.1,
#           'classes': 2,
#           'learning rate': 0.001}
#
# user_features_all = tf.constant([[1, 0, 0],
#                              [0, 1, 0],
#                              [0, 0, 1]],
#                             dtype=tf.float32)
#
# item_features_all = tf.constant([[1, 0, 0],
#                              [0, 1, 0],
#                              [0, 0, 1]],
#                             dtype=tf.float32)
# item_id = [0, 1]
# user_id = [0, 1]
#
#
# user_conv = [tf.constant([[1, 0, 0],
#                           [0, 1, 0],
#                           [0, 0, 1]],
#                          dtype=tf.float32),
#              tf.constant([[1, 0, 0],
#                           [0, 1, 0],
#                           [0, 0, 1]],
#                          dtype=tf.float32)]
#
# item_conv = [tf.constant([[1, 0, 0],
#                           [0, 1, 0],
#                           [0, 0, 1]],
#                          dtype=tf.float32),
#              tf.constant([[1, 0, 0],
#                           [0, 1, 0],
#                           [0, 0, 1]],
#                          dtype=tf.float32)]
#
# init_op = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init_op)




def gcmc_model_fn(features, labels, mode, params):
    labels -= 1
    
    # user_features_all = features['u_features']
    # item_features_all = features['v_features']
    #
    #
    #
    # user_features_all = tf.feature_column.input_layer(user_features_all,
    #                                                   params.user_features_columns)
    # item_features_all = tf.feature_column.input_layer(item_features_all,
    #                                                   params.item_features_columns)


    user_features_all = tf.constant(1, shape=[9366, 18], dtype=tf.float64)
    item_features_all = tf.constant(1, shape=[4618, 175], dtype=tf.float64)

    user_features_all = tf.cast(user_features_all, tf.float64)
    item_features_all = tf.cast(item_features_all, tf.float64)

    # print(features.keys())
    #
    # for key in features:
    #     print(key, features[key].shape)

    # print(user_features_all.shape)(763, 18)
    # print(item_features_all.shape)(2415, 175)



    """
    batch
    """
    user_features_batch = tf.sparse.matmul(features['user_id'],
                                           user_features_all)
    item_features_batch = tf.sparse.matmul(features['item_id'],
                                           item_features_all)

    """convolution"""
    item_conv = []
    user_conv = []
    for star in range(params.classes):
        # TODO: node dropout
        item_conv.append(tf.sparse.matmul(features['item_neigh_conv{}'.format(star)],
                         user_features_all)
                         )
        user_conv.append(tf.sparse.matmul(features['user_neigh_conv{}'.format(star)],
                         item_features_all)
                         )


    """
    Layers at first level
    """
    # user features
    f_user = tf.layers.dense(user_features_batch,
                             units=params.dim_user_raw,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=True
                             )
    f_user = tf.layers.dropout(f_user, rate=params.dropout)

    # user convolutions
    # TODO: weight sharing
    h_user = {}
    for star, u in enumerate(user_conv):
        h_user[star] = tf.layers.dense(u,
                              units=params.dim_user_conv,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_normal_initializer(),
                              use_bias=False
                              )
        h_user[star] = tf.layers.dropout(h_user[star], rate=params.dropout)
    h_user = tf.concat([h_user[star] for star in h_user.keys()], axis=1)
    h_user = tf.layers.dropout(h_user, rate=params.dropout)

    # item features
    f_item = tf.layers.dense(item_features_batch,
                             units=params.dim_item_raw,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=True
                             )
    f_item = tf.layers.dropout(f_item, rate=params.dropout)

    # item convolution
    h_item = []
    # TODO: weight sharing
    h_item = {}
    for star, v in enumerate(item_conv):
        h_item[star] = tf.layers.dense(v,
                              units=params.dim_item_conv,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_normal_initializer(),
                              use_bias=False
                              )
        h_item[star] = tf.layers.dropout(h_item[star], rate=params.dropout)
    h_item = tf.concat([h_item[star] for star in h_item.keys()], axis=1)
    h_item = tf.layers.dropout(h_item, rate=params.dropout)


    """
    Layers at second level
    """
    f_user = tf.layers.dense(f_user,
                             units=params.dim_user_embedding,
                             activation=None,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=False
                             )
    h_user = tf.layers.dense(h_user,
                             units=params.dim_user_embedding,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=False
                             )
    user_embedding = tf.nn.relu(f_user + h_user)

    f_item = tf.layers.dense(f_item,
                             units=params.dim_item_embedding,
                             activation=None,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=False
                             )
    h_item = tf.layers.dense(h_item,
                             units=params.dim_item_embedding,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=False
                             )
    item_embedding = tf.nn.relu(f_item + h_item)


    """
    decoder
    """
    weights_decoder = {}
    with tf.variable_scope('decoder'):
        for star in range(params.classes):
            weights_decoder[star] = tf.get_variable(name='decoder' + str(star),
                                                    shape=[params.dim_user_embedding,
                                                           params.dim_item_embedding],
                                                    dtype=tf.float64,
                                                    trainable=True,
                                                    initializer=tf.glorot_normal_initializer()
                                                    )

    logits = []
    kernel = 0
    for star in weights_decoder:
        kernel += weights_decoder[star]
        # uQ = tf.matmul(user_embedding, kernel)
        # uQv = tf.reduce_sum(tf.multiply(uQ, item_embedding), axis=1)
        #
        logits.append(tf.reduce_sum(tf.multiply(tf.matmul(user_embedding, kernel),
                                                item_embedding),
                                    axis=1))
    logits = tf.stack(logits, axis=1)


    """
    construct estimator
    """
    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    # use Adam
    # optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

