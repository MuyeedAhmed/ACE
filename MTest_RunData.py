import pandas as pd
import numpy as np
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture

import sys
from ACE.ACE_Clustering import ACE_Clustering
from sklearn.metrics.cluster import adjusted_rand_score



        
    
def runAlgo(file, df, algo, gt_available=False):
    r = df.shape[0]
    c = df.shape[1]

    if "class" in df.columns:
        y=df["class"].to_numpy()
        X=df.drop("class", axis=1)
        c=c-1
    else:
        print("Ground truth not available")
        gt_available = False
        y = [0]*r
        X = df
    X.fillna(X.mean(numeric_only=True).round(1), inplace=True)

    
    try:
        clustering = ACE_Clustering(algoName=algo, fileName=file, n_cluster=3)
        clustering.X = X
        clustering.y = y
        ari, time_ = clustering.run()
        clustering.destroy()

        
    except Exception as e:
        try:
            clustering.destroy()
        except:
            pass
        print(file + " killed. Reason: ", e)
        
        

if __name__ == '__main__':
    algo = "HAC"
    filename = "letter"
    folderpath = 'Dataset/'
    
    df = pd.read_csv(folderpath+filename+".csv")
    
    runAlgo(filename, df, algo, True)
    
    