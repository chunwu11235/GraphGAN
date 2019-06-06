import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
#tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from pipeline import preprocessing, get_item_feature_columns, get_user_feature_columns, df2tensor, get_type_dict, construct_feed_dict
from utils import data_iterator, progress_bar
from model import GCMC as gcmc_model

import functools
import sys
import numpy as np
import tensorflow as tf



num_epoch = 10

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device

    return _assign


def main(args):    
    tf.logging.set_verbosity(tf.logging.ERROR)    
    #file_dir = '/Users/Dreamland/Documents/University_of_Washington/STAT548/project/GraphGAN/yelp_dataset/'
    file_dir = '/home/FDSM_lhn/GraphGAN/yelp_dataset/'
    #file_dir = 'yelp_dataset/'
    adj_mat_list, user_norm, item_norm,\
                u_features, v_features, new_reviews, miscellany,\
                N, num_train, num_val, num_test, train_idx, val_idx, test_idx = preprocessing(file_dir, seed=129,verbose=True, test= False)
    item_type_dict = get_type_dict(v_features)
    user_type_dict = get_type_dict(u_features)


    item_feature_columns = get_item_feature_columns(miscellany['business_vocab_list'], item_type_dict)
    user_feature_columns = get_user_feature_columns(user_type_dict)
    
    additional_info = {}
    for name in ['adj_mat_list', 'user_norm', 'item_norm', 'v_features', 'u_features']: 
        exec("additional_info[{0!r}] = {0}".format(name))
   
    model_params = tf.contrib.training.HParams(
    num_users=len(user_norm),
    num_items=len(item_norm),
    model_dir=args.model_dir,
    learning_rate=args.learning_rate,
    dim_user_raw=args.dim_user_raw,
    dim_item_raw=args.dim_item_raw,
    dim_user_conv=args.dim_user_conv,
    dim_item_conv=args.dim_item_conv,
    dim_user_embedding=args.dim_user_embedding,
    dim_item_embedding=args.dim_item_embedding,
    regularizer_parameter= args.regularizer_parameter,
    is_Adam= args.is_Adam,
    classes=5,
    num_train = num_train,
    batch_size = args.batch_size,
    num_basis=args.num_basis,
    is_stacked = args.is_stacked,
    dropout=args.dropout,
    user_features_columns = user_feature_columns,
    item_features_columns = item_feature_columns)
    
    placeholders = {
            'user_id': tf.sparse_placeholder(tf.float64),
            'item_id': tf.sparse_placeholder(tf.float64),
            'labels': tf.placeholder(tf.int64, shape = (None,)),
            'training':tf.placeholder(tf.bool) 
        }
    for star in range(5):
        placeholders['item_neigh_conv{}'.format(star)] = tf.sparse_placeholder(tf.float64)
        placeholders['user_neigh_conv{}'.format(star)] = tf.sparse_placeholder(tf.float64)

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
 


    if args.use_gpu:
        with tf.device(assign_to_device('/gpu:{}'.format(0), ps_device='/cpu:0')):
            #initialize placeholder
            #placeholders = {
            #        'user_id': tf.sparse_placeholder(tf.float64),
            #        'item_id': tf.sparse_placeholder(tf.float64),
            #        'labels': tf.placeholder(tf.int64, shape = (None,)),
            #        'training':tf.placeholder(tf.bool) 
            #        }
            #for star in range(5):
            #    placeholders['item_neigh_conv{}'.format(star)] = tf.sparse_placeholder(tf.float64)
            #    placeholders['user_neigh_conv{}'.format(star)] = tf.sparse_placeholder(tf.float64)

            #v_feature_placeholder_dict = {}
            #for k, v in item_type_dict.items():
            #    if "categories" != k:
            #        v_feature_placeholder_dict[k] = tf.placeholder(v, shape = (None,))
            #v_feature_placeholder_dict["categories"] = tf.sparse_placeholder(tf.string)
            #
            #u_feature_placeholder_dict = {}
            #for k, v in user_type_dict.items():
            #    u_feature_placeholder_dict[k] = tf.placeholder(v, shape = (None,))

            #placeholders['u_features'] = u_feature_placeholder_dict
            #placeholders['v_features'] = v_feature_placeholder_dict
     
            #    
            model = gcmc_model(placeholders, model_params)
            merged_summary = tf.summary.merge_all()
    else:
            model = gcmc_model(placeholders, model_params)
            merged_summary = tf.summary.merge_all()
        

    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer()) 
        
        if args.Train:
                        
            if args.continue_training:
                model.load(sess) 
                print('Continue from previous checkout,current step is {}'.format(model.global_step.eval())) 

            train_summary_writer = tf.summary.FileWriter(args.model_dir+'/train') 
            val_summary_writer = tf.summary.FileWriter(args.model_dir+'/val') 



            for epoch in range(num_epoch):
                print("This is epoch {}".format(epoch))
                train_data_generator = data_iterator(new_reviews[train_idx], args.batch_size)
                train_total_loss = 0
                train_total_accuracy = 0
                train_total = 0
                train_total_mse = 0

                try:
                    while True:
                        train_reviews = next(train_data_generator)
                        train_count = len(train_reviews)
                        train_feed_dict = construct_feed_dict(placeholders,train_reviews ,additional_info ,model_params)
                        train_feed_dict[placeholders['training']] = True
                        train_result =sess.run([model.training_op, model.loss, model.accuracy, model.mse], train_feed_dict)
                        
                        
                        if model.global_step.eval() % args.summary_steps == 0:
                            print("\nStart Evaluation:") 
                            val_data_generator = data_iterator(new_reviews[val_idx], args.batch_size)
                            
                            val_total_loss = 0
                            val_total_accuracy = 0
                            val_total = 0
                            val_total_mse = 0 
                            try:
                                while True:
                                    val_reviews = next(val_data_generator)
                                    val_count = len(val_reviews)
                                    val_feed_dict = construct_feed_dict(placeholders,val_reviews ,additional_info ,model_params)
                                    val_feed_dict[placeholders['training']] = False 
                                    val_result = sess.run([model.loss, model.accuracy, model.mse], val_feed_dict)
                                    val_total_loss += val_result[0] * val_count
                                    val_total_accuracy += val_result[1] * val_count
                                    val_total += val_count 
                                    val_total_mse += val_result[2] * val_count
                                    #print("Evaluation loss will be : ", val_result[0])
                                    progress_bar(val_total, num_val, 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Mse: %.3f' \
                                            % (val_total_loss/val_total, 100.*val_total_accuracy/val_total, val_total_accuracy, val_total, val_total_mse/val_total))


                            except StopIteration:
                                pass
                            val_loss = val_total_loss/num_val
                            val_accuracy = val_total_accuracy/num_val
                            
                            summary = sess.run(merged_summary, feed_dict=train_feed_dict)
                            train_summary_writer.add_summary(summary, model.global_step.eval())
                            train_summary_writer.flush()
                            
                            summary = sess.run(merged_summary, feed_dict=val_feed_dict)
                            val_summary_writer.add_summary(summary, model.global_step.eval())
                            val_summary_writer.flush()

                        if model.global_step.eval() % args.save_checkpoint_steps == 0:
                            '''
                            Save if we evaluate for 3 times or  model performs good 
                            '''
                            model.save(sess)

                        train_total_loss += train_result[1] * train_count
                        train_total_accuracy += train_result[2] * train_count
                        train_total_mse += train_result[3] * train_count
                        train_total += train_count 
                        progress_bar(train_total, num_train, 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Mse %.3f' \
                                % (train_total_loss/train_total, 100.*train_total_accuracy/train_total, train_total_accuracy, train_total, train_total_mse/train_total))
                except StopIteration:
                    pass

        else:
            model.load(sess, args.load_version)
            print('Test from {} checkout point'.format(model.global_step.eval())) 
            val_data_generator = data_iterator(new_reviews[test_idx], args.batch_size)
                            
            val_total_loss = 0
            val_total_accuracy = 0
            val_total = 0
            val_total_mse = 0 
            try:
                while True:
                    val_reviews = next(val_data_generator)
                    val_count = len(val_reviews)
                    val_feed_dict = construct_feed_dict(placeholders,val_reviews ,additional_info ,model_params)
                    val_feed_dict[placeholders['training']] = False 
                    val_result = sess.run([model.loss, model.accuracy, model.mse], val_feed_dict)
                    val_total_loss += val_result[0] * val_count
                    val_total_accuracy += val_result[1] * val_count
                    val_total += val_count 
                    val_total_mse += val_result[2] * val_count
<<<<<<< HEAD
                    progress_bar(val_total, num_test, 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Mse: %.3f' \
=======
                    progress_bar(val_total, num_val, 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Mse: %.3f' \
>>>>>>> d4c4a85836ffd852bdf1b800d73e13e6733d7d0b
                            % (val_total_loss/val_total, 100.*val_total_accuracy/val_total, val_total_accuracy, val_total, val_total_mse/val_total))
            except StopIteration:
                pass


    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_steps', default = 200, type=int, help="number of train steps before evaluation once")
    parser.add_argument('--model_dir', default = "tmp/", help="Directory to save model files")
    parser.add_argument('--use_gpu', action='store_true', help="num of hidden units")
    parser.add_argument('--dropout', default=0.7, type=float, help= "dropout rate")
    parser.add_argument('--save_checkpoint_steps', default = 200, type=int, help="number of train steps before evaluation once")
    parser.add_argument('--Train', action='store_false', help="training or not")
    parser.add_argument('--load_version', default = 1234,type=int,  help="Model version for Testing")

    parser.add_argument('--is_stacked', action='store_true', help="using stack or sum for h layer")
    parser.add_argument('--num_basis', default = 3, type=int, help="using stack or sum for h layer")
   
    parser.add_argument('--is_Adam', action='store_false', help='which optimizer')
    parser.add_argument('--regularizer_parameter', default = 0.0001, type=float, help="Directory to save model files")
    parser.add_argument('--batch_size', default=10000, type=int, help= "assign batchsize for training and eval")
    parser.add_argument('--learning_rate', default=0.007,type=float, help= "learning rate for training")
    parser.add_argument('--dim_user_raw', default=64, type=int, help="num of hidden units")
    parser.add_argument('--dim_item_raw', default=128, type=int, help="num of hidden units")
    parser.add_argument('--dim_user_conv', default=64, type=int, help="num of hidden units")
    parser.add_argument('--dim_item_conv', default=128, type=int, help="num of hidden units")
    parser.add_argument('--dim_user_embedding', default=128, type=int, help="num of hidden units")
    parser.add_argument('--dim_item_embedding', default=128, type=int, help="num of hidden units")
    parser.add_argument('--continue_training', default =0, type = int, help='continue from which timestamp training or not')

    args = parser.parse_args()
    #args = parser.parse_args(['--max_steps=50'])
    main(args)
