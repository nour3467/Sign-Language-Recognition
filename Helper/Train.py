# -- Import Essential Libraries : 
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from pprint import pprint 
import copy
import itertools
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# -- Setup model extractor :
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# -- Mediapipe extractor :
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

# -- Draw landmarks :
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

# -- Drawing Style :
def draw_styled_landmarks(image, results):
    # # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                             ) 



# -- Preprocessing :
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

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def extract_keypoints(results):
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([rh, lh]) 




# Thirty videos worth of data
# no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 10
# Path for exported data, numpy arrays
path = os.getcwd()
DATA_PATH = os.path.join('keypointsDone') 
DATA_PATH = path+"/"+DATA_PATH
# Actions that we try to detect
labels = os.listdir(DATA_PATH)
labels = np.array(labels)

# pprint(labels)

# print(len(labels))

label_map = {label:num for num, label in enumerate(labels)}    

sequences, annotations = [], []
for label in labels:
    for video_ in os.listdir(DATA_PATH+"/"+label):
        video_path = DATA_PATH+"/"+label+"/"+video_.split(".")[0]
        frames = os.listdir(video_path)
        # -- Renaming :
        # index = 0
        # for frame_ in frames :
        #     if os.path.exists(video_path+"/"+frame_):
        #         # print(video_path+"/"+"frame"+"_"+str(index)+".npy")
        #         os.rename(video_path+"/"+frame_, video_path+"/"+"frame"+"_"+str(index)+".npy")
        #         index += 1
        window = []
        Count_ = len(frames)
        if Count_ < 10 :
            continue
        elif Count_ >= 10 : 
            for frame_num in range(5, 15): # sequence_length
                res = np.load(os.path.join(DATA_PATH, label, video_, "{}.npy".format(video_.split(".")[0]+"_"+str(frame_num))).replace("\\", "/")) # frame_
            
                # print(res)
                # res = np.asarray(res).astype(np.float32)
                window.append(res)
            sequences.append(window)
            annotations.append(label_map[label])

# pprint(len(annotations))

sequences = np.asarray(sequences, dtype=object)

annotations = np.array(annotations, dtype=object)

X = np.array(sequences, dtype=object)
X = np.asarray(X).astype('float32')
pprint("-----------------------------------------------------------------------------------------------------")
pprint("-----------------------------------------------------------------------------------------------------")
pprint(X.shape)
pprint("-----------------------------------------------------------------------------------------------------")
pprint("-----------------------------------------------------------------------------------------------------")


# For ConvoLSTM :
# X = X.reshape(X.shape[0],10,7,6,3)

y = to_categorical(annotations).astype(int)

# pprint(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -- Model definition : 
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,126)))  # (30,126)
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(labels.shape[0], activation='softmax'))

filepath = "models/lstm_model_10_best_new_{epoch}_{categorical_accuracy}.h5"

check = ModelCheckpoint(
    filepath,
    monitor='categorical_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    save_freq='epoch',
    options=None,
    initial_value_threshold=None
)


# # Display the models summary.
# # pprint(model.summary()) 

# # res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2000, callbacks=[tb_callback, check])

res = model.predict(X_test)

pprint(labels[np.argmax(res[0])])

pprint(labels[np.argmax(y_test[0])])








