# ================ Download and Prepare the Flickr30k dataset =================== #
import numpy as np
import os
import json
import tarfile


### set your paths ###
path = "data" # path of "DATASET_Flickr30k.tar"
path_tar = os.path.join(path, "DATASET_Flickr30k.tar")
path_dir = os.path.join(path, "DATASET_Flickr30k")

data_tar = tarfile.open(path_tar)



# ---------------- process the features -------------------

### extract feature directory 'fea_att' and 'fea_fc' from "DATASET_Flickr30k.tar" ###
# extract all fea_att
ffinfo = [ # 14 GB warning!!! 
        tarinfo for tarinfo in data_tar.getmembers()
        if tarinfo.name.startswith(os.path.join("DATASET_Flickr30k", "resnet101_fea", "fea_att"))
    ]
ff = data_tar.extractall(members=ffinfo)
# extract all fea_fc
ffinfo = [ # 400 MB
        tarinfo for tarinfo in data_tar.getmembers()
        if tarinfo.name.startswith(os.path.join("DATASET_Flickr30k", "resnet101_fea", "fea_fc"))
    ]
ff = data_tar.extractall(members=ffinfo)


### load features as numpy array ###
path_fea = r"F:\resnet101_fea" # feature dir

# fea_fc : full connected features  (2048,)
# fea_att: convolution features     (256, 2048)
fea1_fc = np.load(os.path.join(path_fea,'fea_fc','36979.npy'))
fea1_att = np.load(os.path.join(path_fea,'fea_att','36979.npz'))

print(fea1_att.files)           # 'feat'
print(type(fea1_att['feat']))   # <class 'numpy.ndarray'>
print(fea1_att['feat'].shape)   # (256, 2048) what the mean of this??



# ---------------------- process the image caption and dictionary---------------------

# json file: image caption and dictionary
with open(os.path.join(path_dir, 'cap_flickr30k.json')) as f:
    data_captions = json.load(f)

with open(os.path.join(path_dir, 'dic_flickr30k.json')) as f:
    data_dic = json.load(f)


### define funtions linking the img file number and the index of captions ###
def idx2img(dic, i): 
    # return integer image id of dic['images'][i]
    # dic = json.load('dic_flickr30k.json')
    return dic['images'][i]['id']

def img2idx(dic, img):
    # img: integer image id, dic: dic_flickr30k
    # return the index of whose 'id' = img
    for d in dic['images']:
        if d['id'] == img:
            return dic['images'].index(d)
    print("img id not found.\n")

### what does captions look like: ###
# data_captions: <list> []
#   data_captions[0]: <list> of dicts [{}.{}....,{}]
#       data_captions[0][0]: <dict> {'caption':['a', 'young', 'man', 'wearing', ... , 'stand', '.'], 
#                                    'clss': ['man', 'cap', 'stand'], 
#                                    'bbox': [[204.0, 58.0, 363.0, 326.0], [255.0, 60.0, 305.0, 94.0], [121.0, 5.0, 427.0, 316.0]], 
#                                    'idx': [2, 6, 14]}

### what does dictionarys look like: ###
# data_dic: <dict>
# data_dic.keys(): dict_keys(['images', 'wtol', 'ix_to_word', 'wtod'])
# data_dic['images'][0]: {'id': 4801369809, 'split': 'train', 'file_path': '4801369809.jpg'}
# data_dic['wtol']['travelers']: 'traveler'  # words normalization
# data_dic['ix_to_word']['2333']: 'comic'    # dictionary with integer index 1-8638, len(data_dic['ix_to_word']) = 8638, data_dic['ix_to_word']['8638'] = 'UNK'
# data_dic['wtod'], i don't know what's the fucking use of this??

# --------------- extracting images -----------------
### tarinfo of all images ###
subdir_img = [
        tarinfo for tarinfo in data_tar.getmembers()
        if tarinfo.name.startswith(os.path.join("DATASET_Flickr30k","images"))
    ]

### extract a certain image ###
for tarinfo in data_tar.getmembers():
    if tarinfo.name.startswith(os.path.join("DATASET_Flickr30k","images","4801369809.jpg")):
        f = data_tar.extract(tarinfo)


# --------------------------
data_tar.close()

