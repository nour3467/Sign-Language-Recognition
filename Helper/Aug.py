# -- Import Augmentation dependencies :
import os
import random
from tqdm import tqdm
import numpy as np

# -- Setup :
labels = dict_1.keys()


# -- Generates augmented videos from a source video :
def videoGenerator(input_file, label,  no_rs, prefix):
    
    # -- Rotation :
    rotate = [iter for iter in [8, 12, 24]]

    if prefix==False :
    # -- Rotate and scale  :
        for index in range(no_rs) :
            rotate_index = random.choice(rotate)
            cmd = f'ffmpeg -i {input_file} -vf "rotate=PI/{rotate_index}, scale=750:750" {label}_rs_{prefix}_{index}.mp4'
            os.system(cmd)
    
    else : 
        for index in range(no_rs) :
            rotate_index = random.choice([10 ,18])
            cmd = f'ffmpeg -i {input_file} -vf "rotate=PI/{rotate_index}, scale=750:750" {label}_rs_{prefix}_{index}.mp4'
            os.system(cmd)


# -- Generates augmented videos from a source video :
def videoGenerator_flip(input_file, label):
    cmd = f'ffmpeg -i {input_file} -vf hflip -c:a copy {label}_rs_{flip}_{index}.mp4'
    os.system(cmd)
    


    

    
    
    
# -- Data Balance :

# -- Create a dict for easy access to videos per label :

# -- Figure out file count : 
count = []
for label in labels:
    count.append(len(dict_1[label]))

balance_map = dict(zip(labels, count))

# print(balance_map)

# -- Augmentation Calculate :
for label in tqdm(labels) :

    # -- Get file count :
    Count_ = balance_map[label]

    #Augmentation logic : videoGenerator(no_s, no_r, no_rs n_io)
    if Count_ >= 20 :
        continue
    else :
        ne = (20 - Count_)
        no_rs = int(ne / Count_) 
        sub_no_rs = 20 - (Count_ + Count_ * no_rs)

    for file_ in dict_1[label] :
        videoGenerator(file_path, label, no_rs, False)

    for file_ in dict_1[label][:sub_no_rs] :
        videoGenerator(file_path, label, 1, True)

    for file_ in #List of all file of a label :
        videoGenerator_flip(file_path, label)
    

 
    
    










# -- Back up :
#for index in range(no_r) :
    #    rotate_index = random.choice(rotate)
    #    cmd = f"ffmpeg -i {input_file} -vf rotate=PI/{rotate_index} {output_file}_rot_{index}.mp4"
    #    os.system(cmd)
    
    # -- Scale :
    # scale = [iter for iter in [-4, -3, -2, 2, 3, 4]]
    # for index in range(no_s) :
    #    scale_index = random.choice(scale)
    #    cmd = f"ffmpeg -i {input_file} -vf scale={scale_index}*iw:-1 {output_file}_s_{index}.mp4"
    #    os.system(cmd)

# -- Setup all helpers functions :

# -- Finding unique lables :
# def unique(list_):
#     npArray = np.array(list_)
#     uniqueNpArray = np.unique(npArray)
#     return uniqueNpArray.tolist()

# -- The json data mapper : 
#mapper = pd.read_json('WLASL_v0.3.json')

# -- Get all labels : 
#labels = unique(list(mapper["gloss"])) 

# -------------------------------
# -- zoom in and out :
    #for index in range(no_io) :
    #    cmd = f"ffmpeg -i {input_file} -filter_complex zoompan=z='if(lte(mod(it*25,42),10),min(max(zoom,pzoom)+0.02,1.5),min(max(zoom,pzoom)-0.0065,1.5))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1 {output_file}_io_{index}.mp4 -y"
    #    os.system(cmd)

# ----------------------------------------------
# path = os.getcwd()
# base_path = os.path.join(path, "labels")
# base_path = base_path.replace("\\", "/")

# --------------------
# -- Get path :
# path = os.getcwd()
# base_path = os.path.join(path, "labels")
# base_path = base_path.replace("\\", "/")