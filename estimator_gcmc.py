from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np



# TODO: params? what parameters do we need?


def gcmc_model_fn(features, labels, mode, params):



    # TODO: input and convolution layer
    user_features = features['user_features']
    item_features = features['item_features']


    x_user = tf.feature_column.input_layer(user_features, params['user_features_columns'])
    x_user_conv_1 = tf.feature_column.input_layer(user_features_conv_1, params['user_features_columns'])
    x_user_conv_2 = tf.feature_column.input_layer(user_features_conv_2, params['user_features_columns'])
    x_user_conv_3 = tf.feature_column.input_layer(user_features_conv_3, params['user_features_columns'])
    x_user_conv_4 = tf.feature_column.input_layer(user_features_conv_4, params['user_features_columns'])
    x_user_conv_5 = tf.feature_column.input_layer(user_features_conv_5, params['user_features_columns'])

    x_item = tf.feature_column.input_layer(item_features, params['item_features_columns'])
    x_item_conv_1 = tf.feature_column.input_layer(item_features_conv_1, params['user_features_columns'])
    x_item_conv_2 = tf.feature_column.input_layer(item_features_conv_2, params['user_features_columns'])
    x_item_conv_3 = tf.feature_column.input_layer(item_features_conv_3, params['user_features_columns'])
    x_item_conv_4 = tf.feature_column.input_layer(item_features_conv_4, params['user_features_columns'])
    x_item_conv_5 = tf.feature_column.input_layer(item_features_conv_5, params['user_features_columns'])


    """
    Layers at first level
    """
    # user features
    f_user = tf.layers.dense(user_features,
                             units=params['hidden units'][0],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=True
                             )
    f_user = tf.layers.dropout(f_user, rate=params['dropout'])

    # user convolutions
    h_user = []
    for u in user_conv:
        h_u = tf.layers.dense(u,
                              units=params['hidden units'][0],
                              activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_normal_initializer(),
                              use_bias=False
                              )
        h_u = tf.layers.dropout(h_u, rate=params['dropout'])
        h_user.append(h_u)
    h_user = tf.concat(h_user, axis=1)
    h_user = tf.layers.dropout(h_user, rate=params['dropout'])

    # item features
    f_item = tf.layers.dense(item_features,
                             units=params['hidden units'][0],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=True
                             )
    f_item = tf.layers.dropout(f_item, rate=params['dropout'])

    # item convolution
    h_item = []
    for v in item_conv:
        h_v = tf.layers.dense(v,
                              units=params['hidden units'][0],
                              activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_normal_initializer(),
                              use_bias=False
                              )
        h_v = tf.layers.dropout(h_v, rate=params['dropout'])
        h_item.append(h_v)
    h_item = tf.concat(h_item, axis=1)
    h_item = tf.layers.dropout(h_item, rate=params['dropout'])


    """
    Layers at second level
    """
    f_user = tf.layers.dense(f_user,
                             units=params['hidden units'][1],
                             activation=None,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=False
                             )
    h_user = tf.layers.dense(h_user,
                             units=params['hidden units'][1],
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=False
                             )
    user_embedding = tf.nn.relu(f_user + h_user)

    f_item = tf.layers.dense(f_item,
                             units=params['hidden units'][1],
                             activation=None,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=False
                             )
    h_item = tf.layers.dense(h_item,
                             units=params['hidden units'][1],
                             kernel_initializer=tf.glorot_normal_initializer(),
                             use_bias=False
                             )
    item_embedding = tf.nn.relu(f_item + h_item)


    """
    decoder
    """
    dim = params['hidden units'][1]

    weights_decoder = []
    with tf.variable_scope('decoder'):
        for i in range(params['classes']):
            weights = tf.get_variable(name='decoder' + str(i),
                                      shape=[dim, dim],
                                      trainable=True,
                                      initializer=tf.glorot_normal_initializer()
                                      )
            weights_decoder.append(weights)

    logits = []
    for i, kernel in enumerate(weights_decoder):
        uQ = tf.matmul(user_embedding, kernel)
        uQv = tf.reduce_sum(tf.multiply(uQ, item_embedding), axis=1)
        logits.append(uQv)
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
    optimizer = tf.train.AdamOptimizer(params['learning rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)







#
#
# def main(argv):
#     args = parser.parse_args(argv[1:])
#
#     # Fetch the data
#     (train_x, train_y), (test_x, test_y) = iris_data.load_data()
#
#     # Feature columns describe how to use the input.
#     my_feature_columns = []
#     for key in train_x.keys():
#         my_feature_columns.append(tf.feature_column.numeric_column(key=key))
#
#     # Build 2 hidden layer DNN with 10, 10 units respectively.
#     classifier = tf.estimator.Estimator(
#         model_fn=my_model,
#         params={
#             'feature_columns': my_feature_columns,
#             # Two hidden layers of 10 nodes each.
#             'hidden_units': [10, 10],
#             # The model must choose between 3 classes.
#             'n_classes': 3,
#         })
#
#     # Train the Model.
#     classifier.train(
#         input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
#         steps=args.train_steps)
#
#     # Evaluate the model.
#     eval_result = classifier.evaluate(
#         input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))
#
#     print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
#
#     # Generate predictions from the model
#     expected = ['Setosa', 'Versicolor', 'Virginica']
#     predict_x = {
#         'SepalLength': [5.1, 5.9, 6.9],
#         'SepalWidth': [3.3, 3.0, 3.1],
#         'PetalLength': [1.7, 4.2, 5.4],
#         'PetalWidth': [0.5, 1.5, 2.1],
#     }
#
#     predictions = classifier.predict(
#         input_fn=lambda:iris_data.eval_input_fn(predict_x,
#                                                 labels=None,
#                                                 batch_size=args.batch_size))
#
#     for pred_dict, expec in zip(predictions, expected):
#         template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
#
#         class_id = pred_dict['class_ids'][0]
#         probability = pred_dict['probabilities'][class_id]
#
#         print(template.format(iris_data.SPECIES[class_id],
#                               100 * probability, expec))
#
#

def test_gcmc_model_fn():
    # for testing
    params = { 'hidden units': [1, 1],
               'dropout': 0.1,
               'classes': 2,
               'learning rate': 0.001}

    user_features = tf.constant([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]],
                                dtype=tf.float32)

    item_features = tf.constant([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]],
                                dtype=tf.float32)

    user_conv = [tf.constant([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]],
                             dtype=tf.float32),
                 tf.constant([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]],
                             dtype=tf.float32)]

    item_conv = [tf.constant([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]],
                             dtype=tf.float32),
                 tf.constant([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]],
                             dtype=tf.float32)]


    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)




# if __name__ == '__main__':
#     tf.logging.set_verbosity(tf.logging.INFO)
#     tf.app.run(main)
#
