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

#class_list = pd.read_csv('hopkins/inet_clusterable_classes.csv')
class_list = pd.read_csv('hopkins/inet_128D_embeddings_clusterable_classes.csv')

feature_csv = pd.read_csv('ViT/feature_extraction/features/ImageNET/INET_compressed_features_dictionary_128.csv')

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

for i in range(len(class_list['ids'])): # TODO: STANDARDIZE TO ONE: 'ids' OR 'class_id'
    c = class_list['ids'][i]
    total = float(i) / len(class_list['ids'])
    if total % 0.05 == 0: # not working, TODO: REVIEW
        print("Progress: {0:.1%}".format(total))
        
    data = feature_csv[feature_csv['class_id'] == int(c)] # Load data from csv of normalized features
    features = data[['n{}'.format(i) for i in range(128)]].to_numpy() # Subset feature columns (0-255) then get numpy array TODO: MODIFY HERE

    range_n_clusters = [2, 3, 4, 5]
    if len(features) <= 5:
        range_n_clusters = range(2, len(features)) # If there are less than 10 samples in a class (prevents silhouette from running with less data points than the number of clusters)
        
    # kmeans:
    silhouettes = []
    parallel = Parallel(n_jobs=-1, return_as="generator")
    resulting_silhouettes = parallel(delayed(cluster)(num_clusters, features) for num_clusters in range_n_clusters) # Run k-means for 10 iterations then get the result, recording cluster ids
    for silhouette in resulting_silhouettes: 
        silhouettes.append(silhouette) # this should work as a lock in order to guarantee that parallel finished
    best_k = np.argmax(silhouettes) + 2
    kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=20).fit(features)        
    cluster_ids_x = kmeans.labels_    
    dictionary['class_id'][str(c)] = {'best_k': int(best_k), 'avg_silhouette': float(silhouettes[np.argmax(silhouettes)]), 'cluster_ids': [int(t) for t in cluster_ids_x]}

with open("reduced_range_128D_embeddings_inet_clusters.json", "w") as outfile: 
    json.dump(dictionary, outfile)

exit(0)

# Not going to be used (features are unormalized)
#with open('ViT/feature_extraction/features/ImageNET/INET_compressed_features_dictionary_256.pt', 'rb') as f: # extremely faster
#    buffer = io.BytesIO(f.read())
#    feature_dictionary = torch.load(buffer) # map_location=torch.device('cuda:0')

# if from original (uncompressed) embeddings:
# (logistics for handling feature data is different)
for i in range(len(class_list['ids'])):
    c = class_list['ids'][i]
    total = float(i) / len(class_list['ids'])
    print("Progress: {0:.1%}".format(total))
    # TODO: LOAD FROM CSV
    features = [e[1].detach() for e in feature_dictionary[str(c)]]
    print('Features', features)
    features = torch.cat(features, dim=0) # fixed/ TODO: check if this really works
    print('Features afterconcat:', features)
    exit(0)
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
