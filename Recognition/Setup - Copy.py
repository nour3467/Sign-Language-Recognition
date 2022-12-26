# -- Import Libraries :
import pandas as pd
from pprint import pprint
import os
from tqdm import tqdm

# -- Get data path :
path = os.getcwd().replace("\\", "/")


# -- Reverse :
src_path = os.path.join(path, "WLASL2000")
src_path = src_path.replace("\\", "/")
dist_path = os.path.join(path, "labels-W")
dist_path = dist_path.replace("\\", "/")



mapper = pd.read_json("WLASL_v0.3.json")

labels_W = list(set(list(mapper["gloss"])))


src_path_2 = os.path.join(path, "labels")
src_path_2 = src_path_2.replace("\\", "/")

# -- Get labels : 
# labels = []
# for file_ in os.listdir(src_path_2) :
#     labels.append(file_.split("-")[0])
# labels = list(set(labels))

labels = os.listdir(src_path_2)

# -- Finding unique lables :
def unique(list_):
    npArray = np.array(list_)
    uniqueNpArray = np.unique(npArray)
    return uniqueNpArray.tolist()

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

# -- Get labels that are in WLASL2000 and not in SEGMENTED :
Global_labels = [label_ for label_ in labels_W if label_ not in labels]

# pprint(Global_labels)
# pprint(len(Global_labels))

# Global_list = []
# for label in Global_labels :
#     list_ = map_dict[label]
#     Global_list.append(list_)

# Global = zip(Global_labels, Global_list)

# pprint(Global)

pprint(map_dict["dawn"])

# pprint(src_path_2)

# pprint(labels_W)
# pprint(len(labels_W))

# -- Create a folder per label :
# for label in tqdm(Global_labels):
#     label_path = os.path.join(path, "Global_labels")
#     label_path = label_path.replace("\\", "/")
#     label_path = label_path+"/"+label
#     if not os.path.isdir(label_path):
#         os.makedirs(label_path)


# for file_ in tqdm(os.listdir(src_path)) :
#     label = file_.split("-")[0]
#     # print(file_)
#     os.rename(src_path+"/"+file_, dist_path+"/"+label+"/"+file_)

# for label in tqdm(labels) :
#     list_ = os.listdir(src_path+"/"+label)
#     for file_ in list_ :
#         # print(file_)
#         os.rename(src_path+"/"+label+"/"+file_, dist_path+"/"+file_)








