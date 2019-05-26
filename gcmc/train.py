from pipeline import preprocessing, get_input_fn, get_item_feature_columns, get_user_feature_columns, df2tensor, get_type_dict
from estimator_gcmc import gcmc_model_fn


import tensorflow as tf
import functools
import sys
import numpy as np

def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    run_config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_secs=20,
            save_summary_steps=100)

    
    #file_dir = '/Users/Dreamland/Documents/University_of_Washington/STAT548/project/GraphGAN/yelp_dataset/'
    file_dir = '/home/FDSM_lhn/GraphGAN/yelp_dataset/'
    #file_dir = 'yelp_dataset/'
    adj_mat_list, user_norm, item_norm,\
                u_features, v_features, new_reviews, miscellany,\
                N, num_train, num_val, num_test, train_idx, val_idx, test_idx = preprocessing(file_dir, verbose=True, test= False)

    
    # TODO: check
    
    item_type_dict = get_type_dict(v_features)
    user_type_dict = get_type_dict(u_features)
    
    item_feature_columns = get_item_feature_columns(miscellany['business_vocab_list'], item_type_dict)
    user_feature_columns = get_user_feature_columns(user_type_dict)
    
    input_additional_info = {}
    for name in ['adj_mat_list', 'user_norm', 'item_norm', 'new_reviews', 'num_train', 'num_val','num_test', 'train_idx', 'val_idx', 'test_idx']:
        exec("input_additional_info[{0!r}] = {0}".format(name))
    
    input_additional_info['u_features'] = u_features
    input_additional_info['v_features'] = v_features
    input_additional_info['col_mapper'] = miscellany['col_mapper']

    
    model_params = tf.contrib.training.HParams(
    num_users = len(user_norm),
    num_items = len(item_norm),
    batch_size=FLAGS.batch_size,
    learning_rate=FLAGS.batch_size,
    dim_user_raw=FLAGS.dim_user_raw,
    dim_item_raw=FLAGS.dim_item_raw,
    dim_user_conv=FLAGS.dim_user_conv,
    dim_item_conv=FLAGS.dim_item_conv,
    dim_user_embedding=FLAGS.dim_user_embedding,
    dim_item_embedding=FLAGS.dim_item_embedding,
    classes=FLAGS.classes,
    dropout=FLAGS.dropout,
    user_features_columns = user_feature_columns,
    item_features_columns = item_feature_columns)
    
    
    
    input_fn=  get_input_fn(tf.estimator.ModeKeys.TRAIN, model_params, **input_additional_info)
        
    estimator = tf.estimator.Estimator(
            gcmc_model_fn,
            config=run_config,
            params=model_params)

    train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn(
        tf.estimator.ModeKeys.TRAIN, model_params,
        **input_additional_info), max_steps=FLAGS.max_steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn(
        tf.estimator.ModeKeys.EVAL,
        model_params,
        **input_additional_info))

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    
    
if __name__ == "__main__":
    
    
    #TODO: add parameters needed later on.
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_integer('max_steps',10,
    "Number of training steps.")
    flags.DEFINE_integer('batch_size', 1024,
    "Number of observations in a sample")
    flags.DEFINE_float('learning_rate', 0.001,
                         "Number of observations in a sample")
    flags.DEFINE_integer('classes', 5,
                         "Number of observations in a sample")
    flags.DEFINE_integer('dim_user_raw', 20,
                         "Number of observations in a sample")
    flags.DEFINE_integer('dim_item_raw', 20,
                         "Number of observations in a sample")
    flags.DEFINE_integer('dim_user_conv', 30,
                         "Number of observations in a sample")
    flags.DEFINE_integer('dim_item_conv', 30,
                         "Number of observations in a sample")
    flags.DEFINE_integer('dim_user_embedding', 30,
                         "Number of observations in a sample")
    flags.DEFINE_integer('dim_item_embedding', 30,
                         "Number of observations in a sample")
    #directory of various files
    flags.DEFINE_string('model_dir', 'tmp/',
    "Path for storing the model checkpoints.")

    flags.DEFINE_float('dropout',0.2,
    "Dropout used for lstm layers.")

    tf.app.run(main=main)
