import torch
import numpy as np
import io
import json
import pandas as pd
import os 
import shutil

class_list = pd.read_csv('hopkins/inet_clusterable_classes.csv')

with open('ViT/feature_extraction/features/ImageNET/imagenet_features_dictionary.pt', 'rb') as f: # extremely faster
    buffer = io.BytesIO(f.read())
    feature_dictionary = torch.load(buffer) # map_location=torch.device('cuda:0')

with open("inet_clusters.json", "r") as infile: 
    dataset_clusters = json.load(infile)

for c in list(dataset_clusters['class_id'].keys()):
    img_paths =  [e[0] for e in feature_dictionary[c]] # [ [0] : Image Source, [1] : extracted features]
    cluster_ids = dataset_clusters['class_id'][c]['cluster_ids']
    concatenated_path_list = [c + '_' + str(directory) + '/' + src for src, directory in zip(img_paths,cluster_ids)]
    for path in concatenated_path_list:
        path = path.split('/')
        if not os.path.exists('train/' + path[0]):
            os.makedirs('train/' + path[0], exist_ok=True)
        source = 'train/'+ c + '/' + path[1]
        destination = 'train/' + path[0]
        shutil.move(source, destination)
    
