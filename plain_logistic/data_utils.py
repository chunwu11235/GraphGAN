import numpy as np
import pandas as pd
import os
import json
import pickle
import sys
from itertools import islice
from collections import defaultdict


def clear_datafiles(file_dir):
    for file_n in os.listdir(file_dir):
        if '.pkl' in file_n or '.npy' in file_n or '.hdf' in file_n:
            os.remove(os.path.join(file_dir, file_n))


def get_useless_item_feat_list():
    return ['address','business_id','hours','is_open','latitude','longitude','name','postal_code','state']

def get_useless_user_feat_list():
    return  ['name', 'yelping_since', 'user_id']


def get_superset_of_column_names_from_file(json_file_path):
    """Read in the json dataset file and return the superset of column names."""
    column_names = set()
    with open(json_file_path) as fin:
        for line in fin:
            line_contents = json.loads(line)
            column_names.update(
                    set(get_key_value_pair(line_contents).keys())
                    )
    return list(column_names)


def strip(value):

    if type(value) == str:
        if len(value) == 0:
            return None
        if value[-1] == "'":
            if value[0] == 'u':
                value = value[2:-1]
            else:
                value = value[1:-1]
            
    return value 

def get_key_value_pair(line_contents, parent_key='', extra_op = False):
    """
    Return a dict of flatten key value pair.
    """
    result = dict()
    for k, v in line_contents.items():
        column_name = "{0}.{1}".format(parent_key, k) if parent_key else k
        
        if v is None:
                continue
        if k == 'attributes':
            sub_result = get_key_value_pair(v, extra_op = True)
            result.update(sub_result)
        else:
            if extra_op and '{' in v:
                v = v.replace('\'', "\"").replace('True', '1').replace('False', '0')
                v = json.loads(v)
                sub_result = get_key_value_pair(v, parent_key = column_name)
                result.update(sub_result)
            elif v != 'None':
                v =  strip(v)
                result.update({column_name:v})
    

    return result

    
def restuarant_loader(file_name, item_id_dict):
    '''
    load restuarant data
    192609 rows in full data 
    flatten all the data in attributes
    categories: 2468 considering embedding    
    only consider restuarant with reviews in our data set
    return 
        pd.Dataframe for item features, 
        vocab for categories[list format],
        item size

    '''
    column_names = get_superset_of_column_names_from_file(file_name)
    result_df = {}
    categories = set()
    business_vocab_list = defaultdict(set)
    count = 0
    
    useless_feat_list = get_useless_item_feat_list()
    with open(file_name) as f:
        for line in f:
            line_contents = json.loads(line)
            result = get_key_value_pair(line_contents)
            if result['business_id'] in item_id_dict.keys():
                
                for col in result.keys():
                    if col== 'categories':
                        business_vocab_list[col].update(set(result['categories'].split(', ')))
                        continue
                        
                    if col not in useless_feat_list and result[col] is not None:
                        business_vocab_list[col].add(result[col])
                
                new_id = item_id_dict[result['business_id']]
                result['business_id'] = new_id
                result_df[new_id] = result
            count =count+ 1
    
    result_vocab= {}
    for feat in business_vocab_list.keys():
        result_vocab[feat] = list(business_vocab_list[feat])
    

    result_df = pd.DataFrame.from_dict(result_df, orient='index', columns=column_names)    
    for feat in useless_feat_list:
        result_df.drop(feat, axis = 1, inplace= True)
        
    
    return result_df, result_vocab, count

    

def user_loader(file_name, user_id_dict):
    result_df = {}
    count = 0
    
    useless_feat_list = get_useless_user_feat_list()
    with open(file_name) as f:
        for line in f:
            line_contents = json.loads(line)
            result = get_key_value_pair(line_contents)
            
            if result['user_id'] in user_id_dict.keys():
                result.pop("friends", None)
                new_id = user_id_dict[result['user_id']]
                result['elite'] = len(result['elite'].split(',')) if result['elite'] is not None else 0
                result['user_id'] = new_id
                result_df[new_id] = result            
            count = count+ 1

    result_df = pd.DataFrame.from_dict(result_df, orient='index')    
    for feat in useless_feat_list:
        result_df.drop(feat, axis = 1, inplace= True)
        
    return result_df, count 
    

def remapping(ori_ids):
    '''
    Give new indices from 
    '''
    uniq_id = set(ori_ids)

    id_dict=  {old:new for new, old in enumerate(uniq_id)}
    new_ids = np.array([id_dict[id] for id in ori_ids])
    n = len(uniq_id)

    return new_ids, id_dict, n


def create_test_file(file, nline = 10000):
    with open(file + ".json", 'rb') as f:
        data = list(islice(f, nline))
    with open(file + "_test.json", 'wb') as f:
        for line in data:
            f.write(line)

def compute_col_mapper(vocab_list):
    '''
    Compute mapper for sparse tensor for embedding layer
    '''
    col_mapper = {}

    for feat in ['categories', 'city']:
        count = 0
        col_mapper[feat] = {}
        for k in vocab_list[feat]:
            col_mapper[feat][k] = count
            count+=1         
    
    return col_mapper
    

def data_loading(file_dir, verbose = False, test= False):
    '''
    preliminary data parsing, and save to another file
    '''
    
    output_file_names = ['u_features.hdf','v_features.hdf', 'new_reviews.npy', 'miscellany.pkl']
    if test:
        output_file_names = ['test'+i for i in output_file_names]
    
    
    
    if output_file_names[0] in os.listdir(file_dir):
        
        output_file_names = [file_dir+i for i in output_file_names]
        u_features = pd.read_hdf(output_file_names[0],'mydata')
        v_features = pd.read_hdf(output_file_names[1],'mydata')
        new_reviews = np.load(output_file_names[2])
        
        with open(output_file_names[3], 'rb') as handle:
            miscellany =  pickle.load(handle)

        return u_features, v_features, new_reviews, miscellany

    
    output_file_names = [file_dir+i for i in output_file_names]
            
    
    file_list = [] 
    
    print("Re-process the full files")
    if test:
        for file in ['business', 'review', 'user']:
            create_test_file(file_dir+file)
        file_list = [file_dir + i + '.json' for i in ['business_test', 'review_test', 'user_test']]
    else:
        file_list = [file_dir + i + '.json' for i in ['business', 'review', 'user']]
        
       
    data = pd.read_json(file_list[1], lines=True)

    new_item_ids, item_id_dict, num_item = remapping(data['business_id'].values)
    new_user_ids, user_id_dict, num_user = remapping(data['user_id'].values)
    
    
    v_features, business_vocab_list, num_v =  restuarant_loader(file_list[0], item_id_dict)
    u_features, num_u  = user_loader(file_list[2], user_id_dict)
    
    
    u_features.to_hdf(output_file_names[0],'mydata',mode='w')
    v_features.to_hdf(output_file_names[1],'mydata',mode='w')
    
    
    
    new_reviews = np.stack([new_user_ids, new_item_ids, data['stars'].values], axis = 1)
    
    np.save(output_file_names[2], new_reviews)
    
    miscellany = {}
    miscellany['num_item'] = num_item
    miscellany['num_user'] = num_user
    miscellany['item_id_dict'] = item_id_dict
    miscellany['user_id_dict'] = user_id_dict
    miscellany['business_vocab_list'] = business_vocab_list
    miscellany['col_mapper'] =  compute_col_mapper(miscellany['business_vocab_list'])
    
    
    with open(output_file_names[3], 'wb') as handle:
        pickle.dump(miscellany, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    return u_features, v_features, new_reviews, miscellany


if __name__ =='__main__':
    file_dir =  "/Users/Dreamland/Documents/University_of_Washington/STAT548/project/GraphGAN/yelp_dataset/"
    file_dir = 'yelp_dataset/'

    if 'business_test.json' not in os.listdir(file_dir):
        for file in ['business', 'review', 'user']:
            create_test_file(file_dir + file)

    u_features, v_features, new_reviews, miscellany = data_loading(file_dir, verbose = False, test = True)
