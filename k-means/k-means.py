import torch
import numpy as np
from kmeans_pytorch import kmeans

import pickle
import pandas as pd
import numpy as np
import io
from sklearn.metrics import silhouette_score
import json
import sys
from joblib import Parallel, delayed
from joblib import parallel_backend
from sklearn.cluster import KMeans

class_list = pd.read_csv('hopkins/inet_clusterable_classes.csv')

class_0_test = class_list['ids'][0] 

with open('ViT/feature_extraction/features/ImageNET/imagenet_features_dictionary.pt', 'rb') as f: # extremely faster
    buffer = io.BytesIO(f.read())
    feature_dictionary = torch.load(buffer) # map_location=torch.device('cuda:0')

classes = []
best_k_list = []
clusters_ids_x_list = []
dictionary = {
    'class_id' : {}
}

def cluster(num_clusters, features):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=20).fit(features)
    silhouette = silhouette_score(features, kmeans.labels_, metric='euclidean')
    return silhouette

for i in range(len(class_list['ids'])):
    c = class_list['ids'][i]
    total = float(i) / len(class_list['ids'])
    print("Progress: {0:.1%}".format(total))
    features = [e[1] for e in feature_dictionary[str(c)]]
    features = torch.cat(features, dim=0) # fixed
    print("No. objects:", len(features))
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    if len(features) <= 10:
        range_n_clusters = range(2, len(features)) # If there are less than 10 samples in a class (prevents silhouette from breaking)
    # kmeans:
    silhouettes = []
    #for num_clusters in range_n_clusters:
    parallel = Parallel(n_jobs=-1, return_as="generator")
    resulting_silhouettes = parallel(delayed(cluster)(num_clusters, features) for num_clusters in range_n_clusters) # Run k-means for 10 iterations then get the result, recording cluster ids
    for silhouette in resulting_silhouettes: 
        silhouettes.append(silhouette) # this should work as a lock in order to guarantee that parallel finished
    best_k = np.argmax(silhouettes) + 2
    kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=20).fit(features)        
    cluster_ids_x = kmeans.labels_    
    dictionary['class_id'][str(c)] = {'best_k': int(best_k), 'avg_silhouette': float(silhouettes[np.argmax(silhouettes)]), 'cluster_ids': [int(t) for t in cluster_ids_x]}

with open("inet_clusters.json", "w") as outfile: 
    json.dump(dictionary, outfile)
