import numpy as np
from data_utils import data_loading
from scipy.sparse import csr_matrix

from collections import OrderedDict
import tensorflow as tf
from tensorflow.feature_column import indicator_column,numeric_column, embedding_column, bucketized_column,categorical_column_with_vocabulary_list




def get_type_dict(features):
    item_datatypes = features.dtypes.iteritems()
    
    type_dict = {}
    #col_mapper is only useful for item feature
    for k, v in item_datatypes:
        if k in ['categories', 'city'] or  v == np.object:
            type_dict[k] = tf.string
        elif v == np.float and k in ['stars', 'average_stars']:
            type_dict[k] = tf.float64
        else:
            type_dict[k] = tf.int64
    
    return type_dict




def get_item_feature_columns(business_vocab_list, item_type_dict):
    
    items_feature_columns = []
    
    bucketized_boundary = {'stars': [2.5, 4]}
    embedding_size = {"categories": 8,
            "city": 4}
    
    for k, v in business_vocab_list.items():
        
        if k in ['review_count']:
            col = numeric_column(k, default_value = 0, dtype= item_type_dict[k])
        elif k in ['stars']:
            col = bucketized_column(numeric_column(k, default_value = 0, dtype= item_type_dict[k]), bucketized_boundary[k])
        elif k in ['categories', 'city']:
            col = embedding_column(categorical_column_with_vocabulary_list(k, sorted(v), default_value = -1, dtype= item_type_dict[k]), dimension = embedding_size[k])
        else:   
            col = indicator_column(categorical_column_with_vocabulary_list(k, sorted(v), default_value = -1, dtype= item_type_dict[k]))

        items_feature_columns.append(col)

    return items_feature_columns


def get_user_feature_columns(user_type_dict):
    items_feature_columns = []
    for k, v in user_type_dict.items():
        col = numeric_column(k, default_value = 0, dtype= v)
        items_feature_columns.append(col)
    return items_feature_columns

def list2sparsetensor(list_feat):
    '''
    Turn list from pd.DataFrame.to_dict for embedding feature column 
    '''
    
    indices = []
    value = []
    max_count =0
    
    for row, cur_list in enumerate(list_feat):
        count = 0
        if isinstance(cur_list, list):
            for cate in cur_list:
                indices.append([row, count])
                value.append(cate.encode('utf8'))
                count += 1
        else:
            # for case where each element is not list but single one
            indices.append([row, count])
            value.append(cur_list.encode('utf8'))
            count += 1
        max_count = max(max_count, count) 

#     indices = tf.convert_to_tensor(indices, tf.int64)
#     value = tf.convert_to_tensor(value, tf.string) 
    
    return tf.SparseTensorValue(indices, value, dense_shape = [len(list_feat), max_count])
   
def df2tensor(features, slice_list):
    
    item_datatypes = features.dtypes.iteritems()
    
    new_features = features.loc[slice_list]
    result_dict = new_features.to_dict(orient = 'list')
    for k, v in item_datatypes:
        if k in ['categories'] :
            result_dict[k] = list2sparsetensor(result_dict[k])
    
    return result_dict


def create_trainvaltest_split(N, verbose, seed=129):
    """
    we have 6million links total
    
    """
    
    all_idx = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(all_idx)
    test_prop = 0.1
    val_prop = 0.05
    split2 = int(N * (1- test_prop))
    num_val = int(split2 * 0.05)
    
    split1 = split2 - num_val
    
    
    train_idx, val_idx, test_idx = np.split(all_idx, [split1, split2])
    if verbose:
        print("We have {} training rating, {} val rating and {}".format(split1, split2-split1, N- split2))
    
    return split1, split2-split1, N- split2, train_idx, val_idx, test_idx



def preprocessing(file_dir, seed=129,verbose = True ,test= False):

    u_features, v_features, new_reviews, miscellany = data_loading(file_dir, verbose = verbose, test = test)
    N = new_reviews.shape[0]
    num_item = miscellany['num_item']
    num_user = miscellany['num_user']
    
    num_train, num_val, num_test, train_idx, val_idx, test_idx = create_trainvaltest_split(N, seed=seed,verbose= verbose)
    
    train_reviews = new_reviews[train_idx]
    adj_mat = csr_matrix((train_reviews[:,2], (train_reviews[:,0], train_reviews[:,1])), shape = (num_user, num_item))
    adj_mat_list = []
    
    for star in range(1, 6):
        new_sparse = csr_matrix(adj_mat ==star, dtype = np.int)
        adj_mat_list.append(new_sparse)
        
    #normalization 
    tot_adj = np.sum(adj_mat_list)
    
    user_node_degree = np.array(tot_adj.sum(axis= 1)).squeeze().astype(np.float)
    item_node_degree = np.array(tot_adj.sum(axis= 0)).squeeze().astype(np.float)
    
    user_node_degree[user_node_degree==0] = np.inf
    item_node_degree[item_node_degree==0] = np.inf
    
    user_norm = 1/user_node_degree
    item_norm = 1/item_node_degree
    
    
    for colname in v_features.columns:
        if v_features[colname].dtype == np.object or colname in ['categories','']:
            v_features[colname].fillna("", inplace=True)
        else:
            v_features[colname].fillna(-1, inplace=True)
    v_features['categories']= v_features['categories'].str.split(', ')
    
    u_features.fillna(0, inplace=True)

    #Note the categories part for item will still be list type inorder for the furthur slice processing
    #This implementation slice after input  
    
    return adj_mat_list, user_norm, item_norm,\
            u_features, v_features, new_reviews, miscellany,\
            N, num_train, num_val, num_test, train_idx, val_idx, test_idx
    
def new_id_mapper(id_list, mapper, cur_count):
    
    new_id_list = np.zeros(len(id_list))
    for num, i in enumerate(id_list):
        if i not in mapper:
            mapper[i] = cur_count
            cur_count += 1
        new_id_list[num] = mapper[i]
    return new_id_list, cur_count
    


def construct_feed_dict(placeholders, cur_review, additional_info, params):
    user_id = cur_review[:, 0]
    item_id = cur_review[:, 1]
    
    
    num_users = params.num_users
    num_items = params.num_items
    
    item_dict = OrderedDict()
    user_dict = OrderedDict()
    item_id_count = 0
    user_id_count = 0
    
    item_indices_list = []
    item_value_list = []
    
    user_indices_list = []
    user_value_list = []
    
    
    for star, adj_mat in enumerate(additional_info["adj_mat_list"]):
        item_sparse = adj_mat[:, item_id]
        user_sparse = adj_mat[user_id, :]
        
        item_neigh_id = item_sparse.getH().nonzero()[1]
        user_neigh_id = user_sparse.nonzero()[1]
        
        new_item_neigh_id = np.zeros(len(item_neigh_id))
        new_user_neigh_id = np.zeros(len(item_neigh_id))
        
        
        new_item_neigh_id, user_id_count = new_id_mapper(item_neigh_id, user_dict, user_id_count)
        new_user_neigh_id, item_id_count = new_id_mapper(user_neigh_id, item_dict, item_id_count)
                    
        item_neigh_num = item_sparse.sum(axis=0).A1
        user_neigh_num = user_sparse.sum(axis=1).A1
        
        item_row = np.repeat(np.arange(len(item_id)), item_neigh_num)
        
        item_indices = np.stack([item_row, new_item_neigh_id], axis = 1)
        #left normalization
        item_value = np.repeat(additional_info["item_norm"][item_id], item_neigh_num)
        
        
        user_row = np.repeat(np.arange(len(user_id)), user_neigh_num)
        user_indices = np.stack([user_row, new_user_neigh_id], axis = 1)
        #left normalization
        user_value = np.repeat(additional_info["user_norm"][user_id], user_neigh_num)
        
        
        item_indices_list.append(item_indices)
        item_value_list.append(item_value)
        user_indices_list.append(user_indices)
        user_value_list.append(user_value)
        
    new_user_id, user_id_count = new_id_mapper(user_id, user_dict, user_id_count)
    new_item_id, item_id_count = new_id_mapper(item_id, item_dict, item_id_count)
    
    result_dict = {}
    for star in range(5):
        result_dict[placeholders['item_neigh_conv{}'.format(star)]]= \
            tf.SparseTensorValue(item_indices_list[star], item_value_list[star].astype(np.float64), dense_shape = [len(item_id), user_id_count])
        result_dict[placeholders['user_neigh_conv{}'.format(star)]]= \
            tf.SparseTensorValue(user_indices_list[star], user_value_list[star].astype(np.float64), dense_shape = [len(user_id), item_id_count])

    user_indices = np.stack([np.arange(len(user_id)), new_user_id], axis = 1)
    user_value = np.ones(len(user_id))
    item_indices = np.stack([np.arange(len(item_id)), new_item_id], axis = 1)
    item_value = np.ones(len(item_id))
    
    
    result_dict[placeholders['user_id']]= tf.SparseTensorValue(user_indices, user_value.astype(np.float64), dense_shape = [len(user_id), user_id_count])
    result_dict[placeholders['item_id']]= tf.SparseTensorValue(item_indices, item_value.astype(np.float64), dense_shape = [len(item_id), item_id_count])
    
    v_tensor_dict = df2tensor(additional_info["v_features"], list(item_dict.keys())) 
    u_tensor_dict = df2tensor(additional_info["u_features"], list(user_dict.keys()))

    for key in v_tensor_dict:
        result_dict[placeholders['v_features'][key]] = v_tensor_dict[key]
    for key in u_tensor_dict:
        result_dict[placeholders['u_features'][key]] = u_tensor_dict[key]

    result_dict[placeholders['labels']] =   cur_review[:,2]-1
    
    return result_dict


def construct_feed_dict_2(placeholders, cur_review, additional_info, params):
    user_id = cur_review[:, 0]
    item_id = cur_review[:, 1]


    item_dict = OrderedDict()
    user_dict = OrderedDict()
    item_id_count = 0
    user_id_count = 0

    item_indices_list = []
    item_value_list = []

    user_indices_list = []
    user_value_list = []

    # for star, adj_mat in enumerate(additional_info["adj_mat_list"]):
    #     item_sparse = adj_mat[:, item_id]
    #     user_sparse = adj_mat[user_id, :]
    #
    #     # item_neigh_id = item_sparse.getH().nonzero()[1]
    #     # user_neigh_id = user_sparse.nonzero()[1]
    #
    #     # new_item_neigh_id = np.zeros(len(item_neigh_id))
    #     # new_user_neigh_id = np.zeros(len(item_neigh_id))
    #
    #     # new_item_neigh_id, user_id_count = new_id_mapper(item_neigh_id, user_dict, user_id_count)
    #     # new_user_neigh_id, item_id_count = new_id_mapper(user_neigh_id, item_dict, item_id_count)
    #
    #     item_neigh_num = item_sparse.sum(axis=0).A1
    #     user_neigh_num = user_sparse.sum(axis=1).A1
    #
    #     item_row = np.repeat(np.arange(len(item_id)), item_neigh_num)
    #
    #     item_indices = np.stack([item_row, new_item_neigh_id], axis=1)
    #     # left normalization
    #     item_value = np.repeat(additional_info["item_norm"][item_id], item_neigh_num)
    #
    #     user_row = np.repeat(np.arange(len(user_id)), user_neigh_num)
    #     user_indices = np.stack([user_row, new_user_neigh_id], axis=1)
    #     # left normalization
    #     user_value = np.repeat(additional_info["user_norm"][user_id], user_neigh_num)
    #
    #     item_indices_list.append(item_indices)
    #     item_value_list.append(item_value)
    #     user_indices_list.append(user_indices)
    #     user_value_list.append(user_value)

    new_user_id, user_id_count = new_id_mapper(user_id, user_dict, user_id_count)
    new_item_id, item_id_count = new_id_mapper(item_id, item_dict, item_id_count)

    result_dict = {}
    for star in range(5):
        result_dict[placeholders['item_neigh_conv{}'.format(star)]] = \
            tf.SparseTensorValue(item_indices_list[star], item_value_list[star].astype(np.float64),
                                 dense_shape=[len(item_id), user_id_count])
        result_dict[placeholders['user_neigh_conv{}'.format(star)]] = \
            tf.SparseTensorValue(user_indices_list[star], user_value_list[star].astype(np.float64),
                                 dense_shape=[len(user_id), item_id_count])

    user_indices = np.stack([np.arange(len(user_id)), new_user_id], axis=1)
    user_value = np.ones(len(user_id))
    item_indices = np.stack([np.arange(len(item_id)), new_item_id], axis=1)
    item_value = np.ones(len(item_id))

    result_dict[placeholders['user_id']] = tf.SparseTensorValue(user_indices, user_value.astype(np.float64),
                                                                dense_shape=[len(user_id), user_id_count])
    result_dict[placeholders['item_id']] = tf.SparseTensorValue(item_indices, item_value.astype(np.float64),
                                                                dense_shape=[len(item_id), item_id_count])

    v_tensor_dict = df2tensor(additional_info["v_features"], list(item_dict.keys()))
    u_tensor_dict = df2tensor(additional_info["u_features"], list(user_dict.keys()))

    for key in v_tensor_dict:
        result_dict[placeholders['v_features'][key]] = v_tensor_dict[key]
    for key in u_tensor_dict:
        result_dict[placeholders['u_features'][key]] = u_tensor_dict[key]

    result_dict[placeholders['labels']] = cur_review[:, 2] - 1

    return result_dict
