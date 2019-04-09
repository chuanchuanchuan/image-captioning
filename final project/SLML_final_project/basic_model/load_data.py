import os
import json
import numpy as np

def load_caption(path_to_file):
    '''
    load cap_flickr30k and dic_flickr30k
    path_to_file: where you place 'cap_flickr30k.json' and 'dic_flickr30k.json'
    '''
    # json file: image caption and dictionary
    with open(os.path.join(path_to_file, 'cap_flickr30k.json')) as f:
        data_cap = json.load(f)

    with open(os.path.join(path_to_file, 'dic_flickr30k.json')) as f:
        data_dic = json.load(f)

    return data_cap, data_dic

def load_feature(path_to_feature, fea_name, type='fc'):
    '''
    load a feature from a certain file \n
    path_to_feature: the forder where you place fea_fc and fea_att
    fea_name: the name of the feature file, str, end with '.npy' for fea_fc and '.npz' for fea_att
    '''
    path_of_fea_file = os.path.join(path_to_feature, 'fea_'+type, fea_name)
    imgfea = np.load(path_of_fea_file)
    return imgfea
