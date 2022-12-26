import pickle
import numpy as np
import pandas as pd


data = np.load(r'C:\Users\Bouz\Desktop\ASLLVD-Skeleton\normalized\output_files\output_407.npy',allow_pickle=True)
#data=pd.read_csv(r"C:\Users\Bouz\Desktop\PFA\Sentence Level\how2sign_test.csv")


#with open('train_label.pkl', 'rb') as f:
#    data = pickle.load(f)
    

print(data[1].shape)
