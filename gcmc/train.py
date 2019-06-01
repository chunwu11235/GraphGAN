import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
#tf.get_logger().setLevel(logging.ERROR)
#logging.getLogger('tensorflow').setLevel(logging.ERROR)

from pipeline import preprocessing, get_item_feature_columns, get_user_feature_columns, df2tensor, get_type_dict, construct_feed_dict
from utils import data_iterator, progress_bar
from gcmc_model import GCMC as gcmc_model



import functools
import sys
import numpy as np


import tensorflow as tf


num_epoch = 10



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

    placeholders['u_features'] = u_feature_placeholder_dict
    placeholders['v_features'] = v_feature_placeholder_dict

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
    model_dir = args.model_dir,
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
    
    

    model = gcmc_model(placeholders, model_params)



    with tf.Session() as sess:
        if args.Train:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer()) 
            
            #train_summary_writer = tf.summary.FileWriter(args.model_dir+'', sess.graph) 
            #val_summary_writer = 


            for epoch in range(num_epoch):
                print("This is epoch {}".format(epoch))
                train_data_generator = data_iterator(new_reviews[train_idx], args.batch_size)
                train_total_loss = 0
                train_total_accuracy = 0
                train_total = 0
                             
                try:
                    while True:
                        train_reviews = next(train_data_generator)
                        train_count = len(train_reviews)
                        train_feed_dict = construct_feed_dict(placeholders,train_reviews ,additional_info ,model_params)
                        train_result =sess.run([model.training_op, model.loss, model.accuracy], train_feed_dict)
                        
                        
                        if model.global_step.eval() % args.summary_steps == 0:
                            print("Start Evaluation:") 
                            val_data_generator = data_iterator(new_reviews[val_idx], args.batch_size)
                            
                            val_total_loss = 0
                            val_total_accuracy = 0
                            val_total = 0
                             
                            try:
                                while True:
                                    val_reviews = next(val_data_generator)
                                    val_count = len(val_reviews)
                                    val_feed_dict = construct_feed_dict(placeholders,val_reviews ,additional_info ,model_params)
                                    val_result = sess.run([model.loss, model.accuracy], val_feed_dict)
                                    val_total_loss += val_result[0] * val_count
                                    val_total_accuracy += val_result[1] * val_count
                                    val_total += val_count 
                                    progress_bar(val_total, num_val, 'Loss: %.3f | Acc: %.3f%% (%d/%d)' \
                                            % (val_total_loss/val_total, 100.*val_total_accuracy/val_total, val_total_accuracy, val_total))


                            except StopIteration:
                                pass
                            val_loss = total_loss/num_val
                            val_accuracy = total_accuracy/num_val
                        
                        #if save:

                            '''
                            Save if we evaluate for 3 times or  model performs good 
                            '''
                        
                        

                        train_total_loss += train_result[1] * train_count
                        train_total_accuracy += train_result[2] * train_count
                        train_total += train_count 
                        progress_bar(train_total, num_train, 'Loss: %.3f | Acc: %.3f%% (%d/%d)' \
                                % (train_total_loss/train_total, 100.*train_total_accuracy/train_total, train_total_accuracy, train_total))
                except StopIteration:
                    pass

        else:
            pass
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1024, type=int, help= "assign batchsize for training and eval")
    parser.add_argument('--learning_rate', default=0.001,type=float, help= "learning rate for training")
    parser.add_argument('--dropout', default=0.2, type=float, help= "dropout rate")
    parser.add_argument('--summary_steps', default = 200, type=int, help="number of train steps before evaluation once")
    parser.add_argument('--model_dir', default = "tmp/", help="Directory to save model files")
    parser.add_argument('--Train', default = True, help="training or not")
    #parser.add_argument('--model_dir', default = "tmp/", help="Directory to save model files")
    #parser.add_argument('--model_dir', default = "tmp/", help="Directory to save model files")

    args = parser.parse_args()
    #args = parser.parse_args(['--max_steps=50'])
    main(args)
