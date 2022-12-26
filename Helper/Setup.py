# -- Import Libraries :
from multiprocessing.spawn import import_main_path
import pandas as pd
import numpy as np
from pprint import pprint
import os
from tqdm import tqdm

# -- Setup all helpers functions :

# -- Finding unique lables :
def unique(list_):
    npArray = np.array(list_)
    uniqueNpArray = np.unique(npArray)
    return uniqueNpArray.tolist()


# -- The json data mapper : 
mapper = pd.read_json('WLASL_v0.3.json')

# -- Get all labels : 
labels = unique(list(mapper["gloss"]))

# -- Create a dict for easy access to videos per label :
list_videos = []
target = mapper["instances"]
for index_1 in range(len(target)) :
    temp = []
    for index_2 in range(len(target[index_1])):
        id = target[index_1][index_2]["video_id"]
        temp.append(str(id)+".mp4")
    list_videos.append(temp)

map_dict = dict(zip(labels, list_videos))

# -- Move files to its own places : 
path = os.getcwd()
src_path = os.path.join(path, "videos")
src_path = src_path.replace("\\", "/")
dist_path = os.path.join(path, "labels")
dist_path = dist_path.replace("\\", "/")

# -- Create a folder per label :
for label in tqdm(map_dict.keys()):
    label_path = os.path.join(path, "labels")
    label_path = label_path.replace("\\", "/")
    label_path = label_path+"/"+label
    if not os.path.isdir(label_path):
        os.makedirs(label_path)

for label in tqdm(map_dict.keys()):
    for file_ in tqdm(map_dict[label]) :
        # -- Verify :
        # if os.path.exists(src_path+"/"+file_) & os.path.exists(src_path+"/"+file_) :
        #     pass
        # else :
        #     pprint(f'Problem in file : {file_}' )
        # -- 
        os.rename(src_path+"/"+file_, dist_path+"/"+label+"/"+file_)







