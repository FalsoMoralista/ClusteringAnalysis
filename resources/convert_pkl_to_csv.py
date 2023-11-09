import pickle
import torch
import pandas as pd
import numpy as np

feature_dictionary = []
with open('../features/ImageNET/imagenet_features_dictionary.pt', 'rb') as pkl_fle:
    feature_dictionary = torch.load(pkl_fle)

#print(feature_dictionary['10000918'])
print("##############################################################")
print("This is the first tuple" , feature_dictionary['10000918'][0]) 
print("This is the image name" , feature_dictionary['10000918'][0][0])
print("This are the features extracted from it" , feature_dictionary['10000918'][0][1])
print("This is the first element" , feature_dictionary['10000918'][0][1][0].numpy()[0])

'''
    feature_dictionary['10000918'] -> This is a list 
    feature_dictionary['10000918'][0] -> This is the first tuple
    feature_dictionary['10000918'][0][0] -> This is the image name
    feature_dictionary['10000918'][0][1] -> This are the features extracted from it.

'''
df_header = ['class_id', 'image_name']
feature_no = 'n{}'
for i in range(1024):
    df_header.append(feature_no.format(i))

#print(df_header[2:])
#print("#######################")

class_list = list(feature_dictionary.keys())
keys = []
paths = []
features = []
for key in class_list:
    for img in feature_dictionary[str(key)]:
        keys.append(key)
        paths.append(img[0])
        features.append(img[1].numpy())

data_features = pd.DataFrame(data=np.concatenate(features), columns=df_header[2:])
intro = pd.DataFrame({'class_id':keys,'img_path':paths})
merged_df = pd.concat([intro, data_features], axis=1, join="outer")
print(merged_df)
merged_df.to_csv('../features/ImageNET/inet_features.csv')

#dataframe = pd.DataFrame(entries, columns=df_header)

''' TODO:
1 - Create dataframe header (columns): (id, image_name, n0, n1, n2, ..., n999)
2 - 

'''