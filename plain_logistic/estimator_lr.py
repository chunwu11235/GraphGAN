from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# TODO: params? what parameters do we need?





def lr_model_fn(features, labels, mode, params):
    user_features_all = features['u_features']
    item_features_all = features['v_features']
    
    user_features_all = tf.feature_column.input_layer(user_features_all,
                                                      params.user_features_columns)
    item_features_all = tf.feature_column.input_layer(item_features_all,
                                                      params.item_features_columns)



    user_features_all = tf.cast(user_features_all, tf.float64)
    item_features_all = tf.cast(item_features_all, tf.float64)
    """
    batch
    """
    user_features_batch = tf.sparse.matmul(features['user_id'],
                                           user_features_all)
    item_features_batch = tf.sparse.matmul(features['item_id'],
                                           item_features_all)
    
    feat_mat = tf.concat([user_features_batch, item_features_batch], axis =1)

    logits = tf.layers.dense(feat_mat,
                             units=5,
                             activation=None,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=True
                             )
    
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

