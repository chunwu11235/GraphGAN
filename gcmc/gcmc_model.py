from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class GCMC:
    def __init__(self, placeholders, params):
        self.loss = 0
        self.accuracy = 0
        self.training_op = None
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.trainable_vars = {}

        self.model_name = 'gcmc'
        self.model_dir = params.model_dir
        self.build(placeholders, params)

    def build(self, placeholders, params):
        # pass parameters
        num_basis = 3
        regularizer = tf.contrib.layers.l2_regularizer
        regularizer_parameter = params.regularizer_parameter


        # input data
        user_features_all = tf.feature_column.input_layer(placeholders['u_features'],
                                                          params.user_features_columns)
        item_features_all = tf.feature_column.input_layer(placeholders['v_features'],
                                                          params.item_features_columns)
        user_features_all = tf.cast(user_features_all, tf.float64)
        item_features_all = tf.cast(item_features_all, tf.float64)

        # get batch
        user_features_batch = tf.sparse.matmul(placeholders['user_id'], user_features_all)
        item_features_batch = tf.sparse.matmul(placeholders['item_id'], item_features_all)

        # get conv
        item_conv, user_conv = [], []
        for star in range(params.classes):
            item_conv.append(tf.sparse.matmul(placeholders['item_neigh_conv{}'.format(star)],
                                              user_features_all))
            user_conv.append(tf.sparse.matmul(placeholders['user_neigh_conv{}'.format(star)],
                                              item_features_all))

        # === dense layers at the first level ===
        f_user = tf.layers.dense(user_features_batch,
                                 units=params.dim_user_raw,
                                 activation=None,
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=regularizer(regularizer_parameter),
                                 use_bias=True,
                                 name='user_features')
        f_item = tf.layers.dense(item_features_batch,
                                 units=params.dim_item_raw,
                                 activation=None,
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=regularizer(regularizer_parameter),
                                 use_bias=True,
                                 name='item_features')
        h_user = []
        for i, u in enumerate(user_conv):
            h_u = tf.layers.dense(u,
                                  units=params.dim_user_conv,
                                  activation=None,
                                  kernel_initializer=tf.glorot_normal_initializer(),
                                  kernel_regularizer=regularizer(regularizer_parameter),
                                  use_bias=False,
                                  name='user_conv_{}'.format(i)
                                  )
            h_user.append(h_u)

        h_item = []
        for i, v in enumerate(item_conv):
            h_v = tf.layers.dense(v,
                                  units=params.dim_item_conv,
                                  activation=None,
                                  kernel_initializer=tf.glorot_normal_initializer(),
                                  kernel_regularizer=regularizer(regularizer_parameter),
                                  use_bias=False,
                                  name='item_conv_{}'.format(i)
                                  )
            h_item.append(h_v)
            
        # is stacked?
        if params.is_stacked:
            h_user = tf.concat(h_user, axis=1)
            h_item = tf.concat(h_item, axis=1)
        else:
            h_user = tf.add_n(h_user)
            h_item = tf.add_n(h_item)

        # batch norm
        f_user = tf.cast(f_user, tf.float32)
        h_user = tf.cast(h_user, tf.float32)
        f_item = tf.cast(f_item, tf.float32)
        h_item = tf.cast(h_item, tf.float32)

        f_user = tf.contrib.layers.batch_norm(f_user,
                                              is_training=placeholders['training'],
                                              trainable=True)
        h_user = tf.contrib.layers.batch_norm(h_user,
                                              is_training=placeholders['training'],
                                              trainable=True)
        f_item = tf.contrib.layers.batch_norm(f_item,
                                              is_training=placeholders['training'],
                                              trainable=True)
        h_item = tf.contrib.layers.batch_norm(h_item,
                                              is_training=placeholders['training'],
                                              trainable=True)

        f_user = tf.cast(f_user, tf.float64)
        h_user = tf.cast(h_user, tf.float64)
        f_item = tf.cast(f_item, tf.float64)
        h_item = tf.cast(h_item, tf.float64)

        # activiation
        f_user = tf.nn.relu(f_user)
        f_item = tf.nn.relu(f_item)
        h_user = tf.nn.relu(h_user)
        h_item = tf.nn.relu(h_item)

        # dropout
        f_user = tf.layers.dropout(f_user, rate=params.dropout, training=placeholders['training'])
        f_item = tf.layers.dropout(f_item, rate=params.dropout, training=placeholders['training'])
        h_user = tf.layers.dropout(h_user, rate=params.dropout, training=placeholders['training'])
        h_item = tf.layers.dropout(h_item, rate=params.dropout, training=placeholders['training'])

        # === dense layers at the 2nd level ===
        f_user = tf.layers.dense(f_user,
                                 units=params.dim_user_embedding,
                                 activation=None,
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=regularizer(regularizer_parameter),
                                 use_bias=False,
                                 name='f_user')
        h_user = tf.layers.dense(h_user,
                                 units=params.dim_user_embedding,
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=regularizer(regularizer_parameter),
                                 use_bias=False,
                                 name='h_user')
        f_item = tf.layers.dense(f_item,
                                 units=params.dim_item_embedding,
                                 activation=None,
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=regularizer(regularizer_parameter),
                                 use_bias=False,
                                 name='f_item')
        h_item = tf.layers.dense(h_item,
                                 units=params.dim_item_embedding,
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=regularizer(regularizer_parameter),
                                 use_bias=False,
                                 name='h_item')

        user_embedding = tf.nn.relu(f_user + h_user)
        item_embedding = tf.nn.relu(f_item + h_item)

        # === decoder ===
        weights_decoder = []
        with tf.variable_scope('decoder'):
            for i in range(num_basis):
                weights = tf.get_variable(name='decoder' + str(i),
                                          shape=[params.dim_user_embedding,
                                                 params.dim_item_embedding],
                                          dtype=tf.float64,
                                          trainable=True,
                                          regularizer=regularizer(regularizer_parameter),
                                          initializer=tf.glorot_normal_initializer()
                                          )
                weights_decoder.append(weights)

            weight_combination = tf.get_variable(name='weight_combination',
                                                 shape=[5, num_basis],
                                                 dtype=tf.float64,
                                                 trainable=True,
                                                 regularizer=None,
                                                 initializer=tf.glorot_normal_initializer()
                                                 )

        logits = []
        for row in range(params.classes):
            kernel = 0
            for k in range(num_basis):
                kernel += weight_combination[row, k] * weights_decoder[k]

            uQ = tf.matmul(user_embedding, kernel)
            uQv = tf.reduce_sum(tf.multiply(uQ, item_embedding), axis=1)
            logits.append(uQv)
        logits = tf.stack(logits, axis=1)

        # predicted labels
        predicted_classes = tf.argmax(logits, 1)

        # === performance measure ===
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=placeholders['labels'], logits=logits)
        self.accuracy = tf.contrib.metrics.accuracy(predictions=predicted_classes, labels=placeholders['labels'])
        # self.mse = tf.metrics.mean_squared_error()

        # summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

        # training
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        self.training_op = optimizer.minimize(self.loss, global_step = self.global_step)

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session is not provided.")
        saver = tf.train.Saver()
        save_path = saver.save(sess, '{}{}_{}.ckpt'.format(self.model_dir,
                                                           self.model_name,
                                                           self.global_step
                                                           )
                               )
        print("Model is saved in file: %s" % save_path)

    def load(self, global_step, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session is not provided.")
        saver = tf.train.Saver()
        save_path = saver.save(sess, '{}{}_{}.ckpt'.format(self.model_dir,
                                                           self.model_name,
                                                           global_step
                                                           )
                               )
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)





