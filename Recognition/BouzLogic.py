import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import pandas as pd
from tensorflow import keras
from T2V import t2v
import os
import copy
import itertools

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


model = keras.models.load_model('models_cnn1d.hdf5')


zero = np.zeros(63)
col = np.load("col(1).npy")
df = pd.DataFrame(columns=col)
label=np.load("labels.npy")


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

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_width, image_height = image.shape[1], image.shape[0]
    
    results = hands.process(image)
    hands_list=[]
    data = []
    temp_data = []
    try:
      for idx, hand_handedness in enumerate(results.multi_handedness):
        hands_list.append(hand_handedness.classification[0].label)
    except:
        pass
    hand_max_count = max(hands_list.count("Right"), hands_list.count("Left"))

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
            # Export coordinates
        try:
            # Extract Pose landmark
            for i in range(21):
                    data.append(min(int(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * image_width), image_width - 1))
                    data.append(min(int(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * image_width), image_width - 1))
                    data.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z)
                    temp_data.append(data[i*3:(i*3)+3])
    
            preproc_data=np.array(pre_process_landmark(temp_data))
                
          
            if "Right" in hands_list and "Left" in hands_list:
                df = df.append(pd.DataFrame(preproc_data.reshape(1,-1), columns=col))
                
                
            elif ("Right" in hands_list and "Left" not in hands_list):
                if hand_max_count==1:
                    preproc_data = np.concatenate((preproc_data, zero), axis=None)
        
                    df =df.append(pd.DataFrame(preproc_data.reshape(1,-1), columns=col))
                
                else:
                    preproc_data = np.concatenate((preproc_data[:63], zero), axis=None)
                    df =df.append(pd.DataFrame(preproc_data.reshape(1,-1), columns=col))       
            
            elif ("Right" not in hands_list and "Left" in hands_list):
                if hand_max_count==1:
            
                    preproc_data = np.concatenate((zero, preproc_data), axis=None)
            
                    df =df.append(pd.DataFrame(preproc_data.reshape(1,-1), columns=col))
            
                else:
               
                    preproc_data = np.concatenate((zero, preproc_data[:63]), axis=None)
           
                    df =df.append(pd.DataFrame(preproc_data.reshape(1,-1), columns=col))
                    
            print("True")
                        
            X = df.to_numpy().reshape(df.shape[0],42,3)
            body_language_class = model.predict(X)[len(X)-5:len(X)-1]
            
            
            y_classes = body_language_class.argmax(axis=-1)
            
            values, counts = np.unique(y_classes, return_counts=True)
            
            ind = np.argmax(counts)
            pred=label[values[ind]] # prints the most frequent element
            #pred=label[y_classes]
            #
            
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, pred
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, pred
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            #text to speech
            #t2v(pred)
                  
        except:
            pass
                   
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        
            break
        
        
        #if cv2.waitKey(35000) :
        #    break
cap.release()
cv2.destroyAllWindows()


my_path = './Voice'
for file_name in os.listdir(my_path):
    if file_name.endswith('.mp3'):
        os.remove(my_path +"/"+ file_name)
        