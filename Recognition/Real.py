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

# For ConvoLSTM : 
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions


def draw_finger_angles(image, results, joint_list):
    
    angles = []
    # Loop through hands
    if results.multi_hand_landmarks :
        for hand in results.multi_hand_landmarks:
            #Loop through joint sets 
            for joint in joint_list:
                # First coord
                a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) 
                # Second coord
                b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) 
                # Third coord
                c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) 
                
                radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                angle = np.abs(radians*180.0/np.pi)
                
                if angle > 180.0:
                    angle = 360-angle

                angles.append(angle)
                # angles.append()
                    
                # cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        if len(angles) == 19 :
            angles = angles + [0.0 for iter in range(19)]
    else :
        angles = [0.0 for iter in range(38)]

    return np.array(angles) # image, 

def angle(image, joint_list):

    use_static_image_mode = True 
    use_brect = True

    # -- Model load : 
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    

    image = cv2.resize(image, (750, 750))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    
    # -- Calculate and print Angles :
    angles = draw_finger_angles(image, results, joint_list) # debug_image, 

    return angles



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


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                             ) 

import copy
import itertools

  
def extract(image):
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,static_image_mode=True) as holistic:

        #image = cv2.imread(file_path)
        # image_height, image_width, _ = image.shape
        image = cv2.resize(image, (750, 750))
        # Convert the BGR image to RGB before processing.
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        
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
        
        
        return np.array(row)


sequence_length = 20
# Path for exported data, numpy arrays
path = os.getcwd()
DATA_PATH = os.path.join('keypoints') 
DATA_PATH = path+"/"+DATA_PATH
# Actions that we try to detect
labels = os.listdir(DATA_PATH)
labels = np.array(labels)

label_map = {label:num for num, label in enumerate(labels)}    

sequences, annotations = [], []
for label in labels:
    for video_ in os.listdir(DATA_PATH+"/"+label):
        video_path = DATA_PATH+"/"+label+"/"+video_.split(".")[0]
        frames = os.listdir(video_path)
        # -- Renaming :
        # index = 0
        #for frame_ in frames :
            # if os.path.exists(video_path+"/"+frame_):
            #     # print(video_path+"/"+"frame"+"_"+str(index)+".npy")
            #     os.rename(video_path+"/"+frame_, video_path+"/"+"frame"+"_"+str(index)+".npy")
            #     index += 1
        window = []
        # Count_ = len(frames)
        # if Count_ < 10 :
        #     continue
        # elif Count_ >= 10 : 
        frames = frames[:20]
        for frame_num in range(1,sequence_length+1):
            res_1 = np.load(os.path.join(DATA_PATH, label, video_, "{}.npy".format(video_+"_"+str(frame_num))).replace("\\", "/"))
            res_2 = np.load(os.path.join(DATA_PATH, label, video_, "{}.npy".format(video_+"_"+str(frame_num)+"_angle")).replace("\\", "/"))
            res = np.concatenate((res_1, res_1), axis=0)
            
                # print(res)
                # res = np.asarray(res).astype(np.float32)
            window.append(res)
        sequences.append(window)
        annotations.append(label_map[label])
sequences = np.asarray(sequences, dtype=object)



annotations = np.array(annotations, dtype=object)

X = np.array(sequences, dtype=object)
X = np.asarray(X).astype('float32')


# For ConvoLSTM :
# X = X.reshape(X.shape[0],30,7,6,3)

y = to_categorical(annotations).astype(int)

# pprint(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


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
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20,252)))  # (30,126)
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(labels.shape[0], activation='softmax'))

filepath = "lstm_model_20_best.h5"

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


# Display the models summary.
# pprint(model.summary()) 

# res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


model.load_weights('lstm_model_20_best.h5')

res = model.predict(X_test)

pprint(labels[np.argmax(res[0])])

pprint(labels[np.argmax(y_test[0])])



# -- Real Time : 
sequence = []
sentence = []
threshold = 0.5
        



colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame



cap = cv2.VideoCapture(0)

joint_list = [[4, 3, 2], [8,7,6], [12,11,10], [16,15,14], [20,19,18], [3, 2, 1], [7, 6, 5], [11, 10, 9], [15, 14, 13], [19, 18, 17], [1, 0, 5], [6, 5, 9], [10, 9, 13], [14, 13, 17], [13, 17, 18], [9, 13, 14], [5, 9, 10], [5, 0, 17], [2, 1, 0]]
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        
        
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        row_1 = extract(image)

        row_2 = angle(frame, joint_list)

        row = np.concatenate((row_1, row_2), axis=0)

        # print("--------------------------------------------------------")
        # print(row_2)
        # print(row_2.shape[0])
        # print("--------------------------------------------------------")

        # print(row)

        # if row is None :
        #     continue

        sequence.append(row) # keypoints
        sequence = sequence[-20:] # 30
        # print(np.array(sequence[-1]).shape)
    
        if len(sequence) == 20: # 30
            res = model.predict(np.expand_dims(np.asarray(sequence).astype('float32'),axis=0))[0]
            print(labels[np.argmax(res)])
            # pass


        # #3. Viz logic
        #     if res[np.argmax(res)] > threshold: 
        #         if len(sentence) > 0: 
        #             if labels[np.argmax(res)] != sentence[-1]:
        #                 sentence.append(labels[np.argmax(res)])
        #         else:
        #             sentence.append(labels[np.argmax(res)])

        #     if len(sentence) > 3: 
        #         sentence = sentence[-3:]

        #     # Viz probabilities
        #     # image = prob_viz(res, labels, image, colors)

        #     cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        #     cv2.putText(image, ' '.join(sentence), (3,30),  # 10
        #                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('ASL SYSTEM', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


