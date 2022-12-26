# -- Import Libraries :
import pandas as pd
import numpy as np
from pprint import pprint
import os
from tqdm import tqdm


labels = os.listdir("labels")


# -- Get videos : 
path = os.getcwd()
base_path = os.path.join(path, "labels")
base_path = base_path.replace("\\", "/")

files_list = []
for label in tqdm(labels) :
    files_list.append(os.listdir(base_path+"/"+label))

# -- videos dict : 
map_dict = dict(zip(labels, files_list))


# -- Extract keypoints from a single pic :

import mediapipe as mp # Import mediapipe
import cv2 # Import opencv

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

 
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

    return landmark_point


# Important : Preprocessing :
 
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    if max_value == 0 :
        max_value = 1

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

import copy
import itertools

def draw_landmarks(image, results):
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
    # Draw left hand connections
    mp_drawing.draw_landmarks(
            image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(232, 254, 255), thickness=1, circle_radius=4
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 249, 161), thickness=2, circle_radius=2
            ),
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(
            image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(232, 254, 255), thickness=1, circle_radius=4
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 249, 161), thickness=2, circle_radius=2
            ),
    )


  
def extract(file_path):
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,static_image_mode=True) as holistic:

        image = cv2.imread(file_path)
        # image_height, image_width, _ = image.shape
        image = cv2.resize(image, (750, 750))
        # Convert the BGR image to RGB before processing.
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # draw_landmarks(image, results)

        # landmark list
        hand_landmarks_right = results.right_hand_landmarks
        hand_landmarks_left = results.left_hand_landmarks 

        if hand_landmarks_right is not None  and hand_landmarks_left is not None :

            # Landmark calculation
            landmark_list_right = calc_landmark_list(image, hand_landmarks_right)
            landmark_list_left = calc_landmark_list(image, hand_landmarks_left)

        
        else :
            if hand_landmarks_right is None and hand_landmarks_left is None:
                # Landmark calculation
                landmark_list_right = [[0.0, 0.0, 0.0] for index in range(21)]
                landmark_list_left = [[0.0, 0.0, 0.0] for index in range(21)]
                # return None

            elif hand_landmarks_left is None :
                # Landmark calculation
                landmark_list_left = [[0.0, 0.0, 0.0] for index in range(21)]
                landmark_list_right = calc_landmark_list(image, hand_landmarks_right)

            else : 
                landmark_list_right = calc_landmark_list(image, hand_landmarks_left)
                landmark_list_left = [[0.0, 0.0, 0.0] for index in range(21)]

        # Conversion to relative coordinates / normalized coordinates
        pre_processed_landmark_list_right = pre_process_landmark(landmark_list_right)
        pre_processed_landmark_list_left = pre_process_landmark(landmark_list_left)

        # Concate rows
        row = pre_processed_landmark_list_right+pre_processed_landmark_list_left


        # print("----------------------------------------------------------")
        # pprint(row)
        # pprint(len(row))
        # print("----------------------------------------------------------")


        # cv2.waitKey(200)    
        # image = cv2.resize(image, (750, 750)) 
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('Extarcting Keypoints', image)
        
        
        return np.array(row)





path = os.getcwd()
# zero_frame = []
# less_frame = []

# -- Get the duration of a video :
def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)
    
# -- Create a folder per label :
for label in tqdm(map_dict.keys()):
    label_path = os.path.join(path, "Frames")
    label_path = label_path.replace("\\", "/")
    label_path = label_path+"/"+label
    if os.path.isdir(label_path):
        for video_ in map_dict[label]:
            video_ = video_.split(".")[0]
            video_path = label_path+"/"+video_

            # Get frames and Extract keypoints
            if os.path.isdir(video_path) :
                frames = os.listdir(video_path)
                # frames = frames[:30]
                for frame_ in frames:
                    frame_ = frame_.split(".")[0]
                    save_dir = path+"/"+"keypoints"+"/"+label+"/"+video_
                    if not os.path.isdir(save_dir) :
                       os.makedirs(save_dir)
                    # if len(os.listdir(save_dir)) >= 30 :
                    #     continue
                    full_dir = video_path+"/"+frame_+".png"
                    # if extract(full_dir) is None :
                    #     continue
                    # else :
                    np.save(save_dir+"/"+frame_+".npy", extract(full_dir))
                    # extract(full_dir)

# full_dir = "E:/WLASL/Frames/solid/04363_rs_0/04363_rs_0_3.png"
# extract(full_dir)
# pprint(zero_frame)
# pprint(less_frame)
                 

                



