import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
#tf.get_logger().setLevel(logging.ERROR)
#logging.getLogger('tensorflow').setLevel(logging.ERROR)

from pipeline import preprocessing, get_input_fn, get_item_feature_columns, get_user_feature_columns, df2tensor, get_type_dict
from estimator_gcmc import gcmc_model_fn


import functools
import sys
import numpy as np


import tensorflow as tf

def main(args):    
    tf.logging.set_verbosity(tf.logging.INFO)    
    #file_dir = '/Users/Dreamland/Documents/University_of_Washington/STAT548/project/GraphGAN/yelp_dataset/'
    file_dir = '/home/FDSM_lhn/GraphGAN/yelp_dataset/'
    #file_dir = 'yelp_dataset/'
    adj_mat_list, user_norm, item_norm,\
                u_features, v_features, new_reviews, miscellany,\
                N, num_train, num_val, num_test, train_idx, val_idx, test_idx = preprocessing(file_dir, verbose=True, test= False)
    placeholders = {
            'user_id': tf.sparse_placeholder(tf.float64),
            'item_id': tf.sparse_placeholder(tf.float64),
            'labels': tf.placeholder(tf.int64, shape = (None,))
            }
    for star in range(5):
        placeholders['item_neigh_conv{}'.format(star)] = tf.sparse_placeholder(tf.float64)
        placeholders['user_neigh_conv{}'.format(star)] = tf.sparse_placeholder(tf.float64)



    item_type_dict = get_type_dict(v_features)
    user_type_dict = get_type_dict(u_features)

    v_feature_placeholder_dict = {}
    for k, v in item_type_dict.items():
        if "categories" != k:
            v_feature_placeholder_dict[k] = tf.placeholder(v, shape = (None,))
    v_feature_placeholder_dict["categories"] = tf.sparse_placeholder(tf.string)
    
    u_feature_placeholder_dict = {}
    for k, v in user_type_dict.items():
        u_feature_placeholder_dict[k] = tf.placeholder(v, shape = (None,))

    item_feature_columns = get_item_feature_columns(miscellany['business_vocab_list'], item_type_dict)
    user_feature_columns = get_user_feature_columns(user_type_dict)
    
    additional_info = {}
    for name in ['adj_mat_list', 'user_norm', 'item_norm', 'v_features', 'u_features']: 
        exec("additional_info[{0!r}] = {0}".format(name))
    
    
#     temp_item_feature_columns = item_feature_columns
#     item_feature_columns =[]
#     for feat_col in temp_item_feature_columns:
#         if 'categories' not in feat_col.name:
#             item_feature_columns.append(feat_col)
    
    
    model_params = tf.contrib.training.HParams(
    num_users = len(user_norm),
    num_items = len(item_norm),
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    dim_user_raw=10,
    dim_item_raw=10,
    dim_user_conv=10,
    dim_item_conv=10,
    dim_user_embedding=10,
    dim_item_embedding=10,
    classes=5,
    dropout=args.dropout,
    user_features_columns = user_feature_columns,
    item_features_columns = item_feature_columns)
    

    
    
    
    
    
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1024, type=int, help= "assign batchsize for training and eval")
    parser.add_argument('--learning_rate', default=0.001,type=float, help= "learning rate for training")
    parser.add_argument('--dropout', default=0.2, type=float, help= "dropout rate")
    parser.add_argument('--max_steps', default = 10, type=int, help="Max steps to train in trainSpec")
    parser.add_argument('--model_dir', default = "tmp/", help="Directory to save model files")
    args = parser.parse_args()
    #args = parser.parse_args(['--max_steps=50'])
    main(args)
