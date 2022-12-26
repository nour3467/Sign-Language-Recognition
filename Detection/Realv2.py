import cv2
import time
import numpy as np
from random import randint
import sys, os
from sys import platform
import pandas as pd
import argparse
from math import sqrt, acos, degrees, atan, degrees
import random
import glob
from PIL import Image
import itertools
from Real import FRAMES_FIX
import _pickle as pickle
from datetime import datetime
import mediapipe as mp

actions = np.load("label_unique.npy")
FRAMES_FIX = 20

# -- Mediapipe model configuration :
# Holistic model
mp_holistic = mp.solutions.holistic 
# Drawing utilities
mp_drawing = mp.solutions.drawing_utils 

# -- Keypoints Detection Funtion :
# Image + Model --> Image + Results
def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    # Image is no longer writeable
    image.flags.writeable = False                  
    # Make prediction
    results = model.process(image)                 
    # Image is now writeable
    image.flags.writeable = True                   
    # COLOR COVERSION RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

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

def draw_landmarks(image, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    
    
    
def draw_styled_landmarks(image, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    


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

def extract_keypoints(results):
    

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([rh, lh]) # pose, face, lh, 



def extract(image):
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,static_image_mode=True) as holistic:

        #image = cv2.imread(file_path)
        # image_height, image_width, _ = image.shape
        image = cv2.resize(image, (750, 750))
        # Convert the BGR image to RGB before processing.
        #results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image, results = mediapipe_detection(image, holistic)
        
        
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
                #return None

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
        
        
        return np.array(row)


from tensorflow import keras
model = keras.models.load_model(r'C:\Users\Bouz\Desktop\ASLLVD-Skeleton\normalized\1dcnn_lstm_model.h5')

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.4

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
            try:
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-FRAMES_FIX:]
                print(len(sequence))
                if len(sequence) == FRAMES_FIX:
                    print("Start of Sequence")
                    
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    #print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    y_classes = res.argmax(axis=-1)
                
                    values, counts = np.unique(y_classes, return_counts=True)
                    ind = np.argmax(counts)
                    pred=actions[values[ind]]
                    print(pred)
                    sequence=[]
                    print("End of Sequence")
                    
                if len(sequence) <= 12:
                    pred=""
                

                # Viz probabilities
                #image = prob_viz(res, actions, image, colors)
                    
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, pred, (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
            except Exception as e: # work on python 3.x
                print('The error is: '+ str(e))
                pass
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()