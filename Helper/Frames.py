# -- Import Libraries :
from multiprocessing.spawn import import_main_path
import pandas as pd
import numpy as np
from pprint import pprint
import os
from tqdm import tqdm
import subprocess


# -- Setup all helpers functions :

# -- Finding unique lables :
# def unique(list_):
#     npArray = np.array(list_)
#     uniqueNpArray = np.unique(npArray)
#     return uniqueNpArray.tolist()


# -- The json data mapper : 
# mapper = pd.read_json('WLASL_v0.3.json')

# -- Get all labels : 
# labels = unique(list(mapper["gloss"]))

labels = os.listdir("labels")

# -- Get the duration of a video :
def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

# -- Get videos : 
path = os.getcwd()
base_path = os.path.join(path, "labels")
base_path = base_path.replace("\\", "/")

files_list = []
for label in tqdm(labels) :
    files_list.append(os.listdir(base_path+"/"+label))

# -- videos dict : 
map_dict = dict(zip(labels, files_list))


# -- Check on videos list :
#pprint(map_dict)


# -- Create a folder per label :
for label in tqdm(map_dict.keys()):
    label_path = os.path.join(path, "Frames")
    label_path = label_path.replace("\\", "/")
    label_path = label_path+"/"+label
    if not os.path.isdir(label_path):
        os.makedirs(label_path)
        for file_ in map_dict[label]:
            file_ = file_.split(".")[0]
            file_path = label_path+"/"+file_
            if not os.path.isdir(file_path) :
                os.makedirs(file_path)


# pprint(get_length("in.mp4"))


# -- Generate Frames from a video  :  Appoximate 30 frame 
def frameGenerator(input_file, label) :
    path = os.getcwd()
    in_path = os.path.join(path, "labels")
    in_path = in_path.replace("\\", "/")
    in_path = in_path+"/"+label+"/"+input_file
    in_file = input_file.split(".")[0]
    out_path = "Frames"+"/"+label+"/"+in_file
    if len(os.listdir(path+"/"+out_path)) == 0 :
        dur = get_length(in_path)
        if dur == 0 :
            dur = 1
            
        fps = int(30/dur)+1
        cmd = f'ffmpeg -i {in_path} -vf fps={fps} {out_path}/{in_file}_%d.png'
        # subprocess.run(cmd)
        os.system(cmd)


for label in tqdm(labels) :

    # -- Get label files list :
    list_files = map_dict[label]

    for file_ in tqdm(list_files) :
        # -- Generate frames for each file : 
        frameGenerator(file_, label)

