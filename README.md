# Sign Language Detection :pinching_hand:

## :notebook_with_decorative_cover: Introduction 

Our project consists in building a Real-Time Multilingual Sign Language Recognition system to help deaf and dumb people to easily integrate into society and public spaces by overcoming communication barriers using advanced machine learning models.
	
Indeed, our system offers multilingual sign language recognition in real time, or our system can easily play the role of an intermediary between a deaf or mute person and another who knows nothing about sign language by offering real-time subtitles of the gestures made by the deaf and mute person. Moreover, the user can choose the language of the subtitles. 

In addition, our approach is a general purpose approach, so it can be used in any dataset, even it can be used for classification of human actions in real time. We have exploited this flexibility of our approach, to introduce to the user the chance to add their own sign either by using videos to the demonstration using webcam.



Keywords: deaf, manual gestures, sign language, ASL, Deep learning, Computer Vision, mediapipe, features extraction, human pose, skeleton model




## :interrobang: Problematic

Deaf and dumb people use hand signals and gestures to communicate. Normal people have difficulty understanding their language. So there is a need for a system that recognizes the different signs and gestures and transmits the information to normal people. It bridges the gap between disabled and normal people.

## :books: DATASETS

The datasets for the approach of the video models are in the form of the videos for each word:

| DataSet        | Number of words | Link  |
| ------------- |:-------------:| -----:|
| DAI - ASLLVD (bu.edu)| 2,400 | [Link](http://vlm1.uta.edu/~athitsos/asl_lexicon/) |
| ASL-LEX     | 1000      | [Link](https://asl-lex.org/visualization/)  |
| WASL | 2000      |  [Link](https://github.com/dxli94/WLASL)  |

## :chart_with_upwards_trend: Models result

Based on the approach of the static image. We have developed a technique for capturing a constant FIXED_FRAMES of static images (frames) so that the key points of the hand are extracted for each image and restored in a NumPy array. The choice of the images that are extracted from the video depends on the number of static images (frames) of the video and changes from each video reading.

After extracting the key points of the labels, we check if the array has reached the FIXED_FRAMES, if not we apply the filling of the array with zeros.

Finally we pass the set of data, which we have obtained to the model so that we train it.

For this approach we used different models to choose the best one: 


| Model        | Accuracy           |
| ------------- |:-------------:|
| LSTM      | 80%|
| LSTM avec Attention     | 82%     |
| 1DCNN LSTM | 88%      |


## :black_nib: WORD-BASED SENTENCE GENERATION

Our approach is based on the generation or completion of a set of predicted words from the model, we try all possible cases of sentence generation.

Après, on calcule la perplexité des phrases générées. Pour à la fin choisir la phrase du plus grand score. Afin d’enrichir notre modèle avec des milliers de mots en vocabulaire, l’approche de la génération de la phrase a donné un excellent résultat.

## 	:clipboard: REFERENCES


