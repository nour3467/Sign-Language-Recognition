# -- Import Libraries :
import pandas as pd
from pprint import pprint
import os
from tqdm import tqdm

# -- Get data path :
path = os.getcwd().replace("\\", "/")


# -- Reverse :
src_path = os.path.join(path, "Segmented")
src_path = src_path.replace("\\", "/")
dist_path = os.path.join(path, "labels")
dist_path = dist_path.replace("\\", "/")



# -- Get labels : 
labels = []
for file_ in os.listdir(src_path) :
    labels.append(file_.split("-")[0])
labels = list(set(labels))

# labels = os.listdir(src_path)


# pprint(labels)
# pprint(len(labels))



# -- Create a folder per label :
# for label in tqdm(labels):
#     label_path = os.path.join(path, "labels")
#     label_path = label_path.replace("\\", "/")
#     label_path = label_path+"/"+label
#     if not os.path.isdir(label_path):
#         os.makedirs(label_path)


for file_ in tqdm(os.listdir(src_path)) :
    label = file_.split("-")[0]
    # print(file_)
    os.rename(src_path+"/"+file_, dist_path+"/"+label+"/"+file_)

# for label in tqdm(labels) :
#     list_ = os.listdir(src_path+"/"+label)
#     for file_ in list_ :
#         # print(file_)
#         os.rename(src_path+"/"+label+"/"+file_, dist_path+"/"+file_)








