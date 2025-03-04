import os
import shutil
import glob
import pandas as pd
import numpy as np
import random
# from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
import time
from sklearn.utils import shuffle
import threading
# from memory_profiler import profile
import warnings 
warnings.filterwarnings("ignore")
import itertools
from sklearn.metrics import f1_score
import multiprocessing
from scipy.spatial.distance import pdist, squareform

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class ACE_Clustering:
    def __init__(self, algoName, parameters=None, fileName="_", n_cluster=2, batch_count = 0):
        self.algoName = algoName
        if parameters == None:
            self.parameters = self.algo_parameters(algoName)
        else:
            self.parameters = parameters
        
        self.fileName = fileName
        
        self.X = []
        self.y = []
        self.X_batches = []
        self.y_batches = []
        self.bestParams = []
        self.models = []
        
        self.n_cluster = n_cluster
        self.batch_count = batch_count
        self.determine_param_mode = "ARI"
        self.determine_param_clustering_algo = "KM"
        self.rerun_mode = "A"
        self.mergeStyle = "Distance"
        self.AD_algo_merge = "LOF"
        
        if not os.path.isdir("ClusteringOutput/"):
            os.mkdir("ClusteringOutput")
        if not os.path.isdir("Output/"):    
            os.mkdir("Output")
            os.mkdir("Output/Temp")
                    
    def destroy(self):
        if os.path.isdir("Output"):
            shutil.rmtree("Output")
    
    def algo_parameters(self, algo):
        parameters = []
        if algo == "AP":
            damping = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            max_iter = [50, 100, 200, 300, 400, 500, 750, 1000]
            convergence_iter = [5, 10, 15, 20, 25, 30, 40, 50] 
            
            
            parameters.append(["damping", 0.5, damping])
            parameters.append(["max_iter", 200, max_iter])
            parameters.append(["convergence_iter", 15, convergence_iter])
        
        if algo =="SC":
            eigen_solver = ["arpack", "lobpcg"]
            n_components = [None, 2, 3, 4, 5, 6, 10]
            n_init = [1, 5, 10, 15, 20, 30, 50, 100]
            gamma = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
            affinity = ["nearest_neighbors", "rbf"]
            n_neighbors = [5, 10, 20, 30, 50, 75, 100]
            n_neighbors = [2, 3, 5, 10, 20, 30]
            assign_labels = ["kmeans", "discretize", "cluster_qr"]
            assign_labels = ["kmeans", "cluster_qr"]
            degree = [2,3,4,5]
            n_jobs = [None, -1]
            
            
            parameters.append(["eigen_solver", "arpack", eigen_solver])
            parameters.append(["n_components", None, n_components])        
            parameters.append(["n_init", 10, n_init])
            parameters.append(["gamma", 1.0, gamma])   
            parameters.append(["affinity", "rbf", affinity])
            parameters.append(["n_neighbors", 10, n_neighbors])
            parameters.append(["assign_labels", "kmeans", assign_labels])
            parameters.append(["degree", 3, degree])
            parameters.append(["n_jobs", None, n_jobs])  
        if algo == "GMM":
            covariance_type = ['full', 'tied', 'diag', 'spherical']
            tol = [1e-2, 1e-3, 1e-4, 1e-5]
            reg_covar = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
            max_iter = [50, 100, 150, 200]
            n_init = [1,2,3,5]
            init_params = ['kmeans', 'k-means++', 'random']
            warm_start = [False, True]
            
            parameters.append(['covariance_type', 'full', covariance_type])
            parameters.append(['tol', 1e-3, tol])
            parameters.append(['reg_covar', 1e-6, reg_covar])
            parameters.append(['max_iter', 100, max_iter])
            parameters.append(['n_init', 1, n_init])
            parameters.append(['init_params', 'kmeans', init_params])
            parameters.append(['warm_start', False, warm_start])
        
        if algo == "HAC":
            metric = ["euclidean", "l1", "l2", "manhattan", "cosine"]
            linkage = ["ward", "complete", "average", "single"]
            
            parameters.append(["metric", "euclidean", metric])
            parameters.append(["linkage", "ward", linkage])
        
        if algo == "DBSCAN":
            eps_min_samples = []
            dbscan_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
            
            parameters.append(["eps_min_samples", (0.5, 5), eps_min_samples])
            parameters.append(["algorithm", 'auto', dbscan_algorithm])
            
        return parameters
    
    
    def subSample(self):
        batch_size = int(len(self.X)/self.batch_count)
        self.X_batches = [self.X[i:i+batch_size] for i in range(0, len(self.X), batch_size)]
        self.y_batches = [self.y[i:i+batch_size] for i in range(0, len(self.y), batch_size)]
    
    def set_DBSCAN_param(self):
        #min_samples
        batch_size = int(len(self.X)/self.batch_count)
        min_samples = np.linspace(2, int(np.sqrt(batch_size)), 10)
        #eps
        distance_values = []
        for _ in range(3):
            X = self.X_batches[random.randint(0, self.batch_count-1)]
            distance_matrix = pdist(X.values)
            distance_matrix_square = squareform(distance_matrix)
            upper_tri_indices = np.triu_indices(distance_matrix_square.shape[0], k=1)
            upper_tri_values = distance_matrix_square[upper_tri_indices]
            distance_values = np.concatenate((distance_values,upper_tri_values))
        
        p5 = np.percentile(distance_values, 5)
        if p5 <= 0:
            ii = 6
            while p5 <= 0 and ii < 45:
                p5 = np.percentile(distance_values, ii)
                ii+=1
            if ii == 45:
                print("45th Percentile of distance is: ", p5)
                raise Exception('Stopped: Percentile Issue')
        p50 = np.percentile(distance_values, 50)
        eps = np.linspace(p5, p50, 10)
        self.parameters[0][2] = list(itertools.product(eps, min_samples))
    
    def determineParam(self):
        batch_index = 0
        for params in self.parameters:
            threads = []
            f = open("Output/Rank.csv", 'w')
            f.write("Batch,ParameterIndex,Compare,Time\n")
            f.close()
            start_index = batch_index
            for p_v_i in range(len(params[2])):
                params[1] = params[2][p_v_i]
                parameters_to_send = [p[1] for p in self.parameters]
                t = threading.Thread(target=self.worker_determineParam, args=(parameters_to_send,self.X_batches[batch_index], self.y_batches[batch_index], batch_index, p_v_i))
                threads.append(t)
                t.start()
                if batch_index == self.batch_count-1:
                    batch_index = 0
                else:
                    batch_index += 1
            for t in threads:
                t.join()
            df = pd.read_csv("Output/Rank.csv")
            
            if self.determine_param_mode == "ARI_T":
                df["W"] = df.Compare/df.Time
            elif self.determine_param_mode == "ARI":
                df["W"] = df.Compare
            h_r = df["W"].idxmax()
            
            params[1] = params[2][df["ParameterIndex"].iloc[h_r]]
            # if df["Time"].iloc[h_r] > 10:
            #     print("Subsampling timeout")
            #     raise Exception('')
        self.bestParams = [p[1] for p in self.parameters]

    def worker_determineParam(self, parameter, X, y, batch_index, parameter_index):        
        t0 = time.time()
        if self.algoName == "AP":
            c = AffinityPropagation(damping=parameter[0], max_iter=parameter[1], convergence_iter=parameter[2]).fit(X)
            l = c.labels_
        elif self.algoName == "SC":
            c = SpectralClustering(n_clusters=self.n_cluster, eigen_solver=parameter[0], n_components=parameter[1], 
                                   n_init=parameter[2], gamma=parameter[3], affinity=parameter[4], 
                                   n_neighbors=parameter[5], assign_labels=parameter[6], 
                                   degree=parameter[7], n_jobs=parameter[8]).fit(X)
            l = c.labels_
        elif self.algoName == "GMM":
            c = GaussianMixture(n_components=self.n_cluster, covariance_type=parameter[0], tol=parameter[1], 
                                   reg_covar=parameter[2], max_iter=parameter[3], n_init=parameter[4], 
                                   init_params=parameter[5], warm_start=parameter[6]).fit(X)
            l = c.predict(X)
        elif self.algoName == "HAC":
            if parameter[1] == "ward" and  parameter[0] != "euclidean":
               saveStr = str(batch_index)+","+str(parameter_index)+","+str(-1)+","+str(9999999)+"\n"    
               f = open("Output/Rank.csv", 'a')
               f.write(saveStr)
               f.close()
               return
            c = AgglomerativeClustering(n_clusters=self.n_cluster, metric=parameter[0], linkage=parameter[1]).fit(X)
            l = c.labels_
        elif self.algoName == "DBSCAN":
            c = DBSCAN(eps=parameter[0][0], min_samples=int(parameter[0][1]), algorithm=parameter[1]).fit(X)
            l = c.labels_
        t1 = time.time()
        cost = t1-t0
        ari_comp = self.getARI_Comp(X, l)
        saveStr = str(batch_index)+","+str(parameter_index)+","+str(ari_comp)+","+str(cost)+"\n"    
        f = open("Output/Rank.csv", 'a')
        f.write(saveStr)
        f.close()
    
    def getARI_Comp(self, X, l):
        if all(v == -1 for v in l):
            return -1
        n = len(set(l))
        if n == 1:
            n = 2
        if self.determine_param_clustering_algo == "AP":
            c = AffinityPropagation().fit(X)
        elif self.determine_param_clustering_algo == "KM":
            c = KMeans(n_clusters=n, n_init=5).fit(X)
        elif self.determine_param_clustering_algo == "DBS":
            c = DBSCAN().fit(X)
        elif self.determine_param_clustering_algo == "HAC":
            c = AgglomerativeClustering(n_clusters = n).fit(X)
        elif self.determine_param_clustering_algo == "INERTIA":
            centroids = np.array([X[l == i].mean(axis=0) for i in np.unique(l)])
            distances = np.array([np.sum((X - centroids[l[i]])**2) for i in range(len(X))])
            return np.sum(distances) * -1 
        else:
            c = KMeans(n_clusters=n, n_init=5).fit(X)
            a_l = c.labels_
            ari1 = adjusted_rand_score(l, a_l)
            c = DBSCAN().fit(X)
            a_l = c.labels_
            ari2 = adjusted_rand_score(l, a_l)
            c = AgglomerativeClustering(n_clusters = n).fit(X)
            a_l = c.labels_
            ari3 = adjusted_rand_score(l, a_l)
            return (ari1+ari2+ari3)/3

        a_l = c.labels_
        ari = adjusted_rand_score(l, a_l)
        return ari
            
    
    def rerun(self):
        if self.bestParams == []:
            print("Determine best parameters before this step.")
            return
        
        threads = []
        batch_index = 0
        done = False
        while True:
            if done:
                break
            for _ in range(10):
                if batch_index == self.batch_count:
                    done = True
                    break
                t = threading.Thread(target=self.worker_rerun, args=(self.bestParams,self.X_batches[batch_index], self.y_batches[batch_index], batch_index))
                threads.append(t)
                t.start()
                batch_index += 1
            for t in threads:
                t.join()
        if self.mergeStyle == "Distance":
            self.constantKMerge()
        elif self.mergeStyle == "DistanceRatio":
            self.mergeClusteringOutputs_DistRatio()
        elif self.mergeStyle == "AD":
            self.mergeClusteringOutputs_AD()
        
        
    def worker_rerun(self, parameter, X, y, batch_index):
        if self.rerun_mode == "A":
            if self.algoName == "AP":
                c = AffinityPropagation(damping=parameter[0], max_iter=parameter[1], convergence_iter=parameter[2]).fit(X)
                l = c.labels_
            elif self.algoName == "SC":
                c = SpectralClustering(n_clusters=self.n_cluster, eigen_solver=parameter[0], n_components=parameter[1], 
                                       n_init=parameter[2], gamma=parameter[3], affinity=parameter[4], 
                                       n_neighbors=parameter[5], assign_labels=parameter[6], 
                                       degree=parameter[7], n_jobs=parameter[8]).fit(X)
                l = c.labels_
            elif self.algoName == "GMM":
                c = GaussianMixture(n_components=self.n_cluster, covariance_type=parameter[0], tol=parameter[1], 
                                       reg_covar=parameter[2], max_iter=parameter[3], n_init=parameter[4], 
                                       init_params=parameter[5], warm_start=parameter[6]).fit(X)
            
                l = c.predict(X)
            elif self.algoName == "HAC":
                c = AgglomerativeClustering(n_clusters=self.n_cluster, metric=parameter[0], linkage=parameter[1]).fit(X)
                l = c.labels_
            elif self.algoName == "DBSCAN":
                c = DBSCAN(eps=parameter[0][0], min_samples=int(parameter[0][1]), algorithm=parameter[1]).fit(X)
                l = c.labels_    
            X["y"] = y
            X["l"] = l
            X.to_csv("Output/Temp/"+str(batch_index)+".csv", index=False)

        if self.rerun_mode == "B":
            ll = []
            for c in self.models:
                ll.append(c.predict(X))

            if self.algoName == "AP":
                c = AffinityPropagation(damping=parameter[0], max_iter=parameter[1], convergence_iter=parameter[2]).fit(X)
            
            elif self.algoName == "SC":
                c = SpectralClustering(n_clusters=self.n_cluster, eigen_solver=parameter[0], n_components=parameter[1], 
                                       n_init=parameter[2], gamma=parameter[3], affinity=parameter[4], 
                                       n_neighbors=parameter[5], assign_labels=parameter[6], 
                                       degree=parameter[7], n_jobs=parameter[8]).fit(X)
            elif self.algoName == "GMM":
                c = GaussianMixture(n_components=self.n_cluster, covariance_type=parameter[0], tol=parameter[1], 
                                       reg_covar=parameter[2], max_iter=parameter[3], n_init=parameter[4], 
                                       init_params=parameter[5], warm_start=parameter[6]).fit(X)
            
            l = c.predict(X)
            ll.append(l)
            
            if len(ll) == 1:
                X["y"] = y
                X["l"] = l
                X.to_csv("Output/Temp/"+str(batch_index)+".csv", index=False)
                return
            
            f1Scores = []
            for l_i in range(len(ll)):
                unique_values = set(ll[l_i])
                permutations = list(itertools.permutations(unique_values))
                bestPerm = []
                bestF1 = -1
                for perm in permutations:
                    replacements = {}
                    for i in range(len(unique_values)):
                        replacements[i] = perm[i]
                    new_numbers = self.replace_numbers(ll[l_i], replacements)
                    f1_s = f1_score(l, new_numbers, average='weighted')
                    if f1_s > bestF1:
                        bestF1 = f1_s
                        bestPerm = new_numbers
                f1Scores.append(bestF1)
                ll[l_i] = bestPerm
            
            for i in range(len(ll[0])):
                dict={}
                for j in range(len(ll)):
                    if ll[j][i] in dict:
                        dict[ll[j][i]] += f1Scores[j]
                    else:
                        dict[ll[j][i]] = f1Scores[j]
                
                l[i] = max(dict, key=dict.get)
                
            X["y"] = y
            X["l"] = l
            X.to_csv("Output/Temp/"+str(batch_index)+".csv", index=False) 

            self.models.append(c)
    
    def replace_numbers(numbers, replacements):
        new_numbers = []
        for number in numbers:
            if number in replacements:
                new_numbers.append(replacements[number])
            else:
                new_numbers.append(number)
        return new_numbers
    
    
    def constantKMerge(self):
        csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
        csv_files.sort()
        while len(csv_files) > 1:
            for i in range(0, len(csv_files)-1, 2):
                file1 = csv_files[i]
                file2 = csv_files[i+1]
                df1 = pd.read_csv(file1)
                df2 = pd.read_csv(file2)
                
                os.remove(file1)
                os.remove(file2)
                
                X_1 = df1.drop(["y", "l"], axis=1).to_numpy()
                labels1 = df1["l"].to_numpy()
                unique_labels1 = set(df1["l"])
                X_2 = df2.drop(["y", "l"], axis=1).to_numpy()
                labels2 = df2["l"].to_numpy()
                unique_labels2 = set(df2["l"])
                
                
                if len(unique_labels1) == 1 and len(unique_labels2) == 1:
                    df = pd.concat([df1, df2])
                    df.to_csv(file2, index=False)
                    continue
                
                global_centers = []
                global_centers_frequency = []

                global_centers_count = len(unique_labels1)
                for i in unique_labels1:
                    global_centers.append(X_1[labels1 == i].mean(axis=0))
                    global_centers_frequency.append(len(X_1[labels1 == i]))
                
                df2["ll"] = -2
                
                for i in unique_labels2:
                    c = X_2[labels2 == i].mean(axis=0)
                    distances = [np.linalg.norm(np.array(c) - np.array(z)) for z in global_centers]
                    nearest_cluster = distances.index(min(distances))
                    df2.loc[df2['l'] == i, 'll'] = nearest_cluster
                df2 = df2.drop("l", axis=1)
                df2 = df2.rename(columns={'ll': 'l'})
                
                df = pd.concat([df1, df2])
                df.to_csv(file2, index=False)
            
            csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
            csv_files.sort()
            
        csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
        df = pd.read_csv(csv_files[0])
        df = df.rename(columns={'l': 'ACE_Labels'})
        df.to_csv("ClusteringOutput/"+self.fileName+"_"+self.algoName+".csv", index=False)
        
        
    def mergeClusteringOutputs_AD(self):
        
        csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
        csv_files.sort()

        while len(csv_files) > 1:
            for i in range(0, len(csv_files)-1, 2):
                file1 = csv_files[i]
                file2 = csv_files[i+1]
                df1 = pd.read_csv(file1)
                df2 = pd.read_csv(file2)
                
                os.remove(file1)
                os.remove(file2)
                
                X_1 = df1.drop(["y", "l"], axis=1).to_numpy()
                labels1 = df1["l"].to_numpy()
                unique_labels1 = set(df1["l"])
                
                X_2 = df2.drop(["y", "l"], axis=1).to_numpy()
                labels2 = df2["l"].to_numpy()
                unique_labels2 = set(df2["l"])
                
                
                if len(unique_labels1) == 1 and len(unique_labels2) == 1:
                    df = pd.concat([df1, df2])
                    df.to_csv(file2, index=False)
                    continue
                global_centers = []
                global_centers_frequency = []

                global_centers_count = len(unique_labels1)
                for i in unique_labels1:
                    global_centers.append(X_1[labels1 == i].mean(axis=0))
                    global_centers_frequency.append(len(X_1[labels1 == i]))
                
                df2["ll"] = -2
                
                for i in unique_labels2:
                    c = X_2[labels2 == i].mean(axis=0)
                    distances = [np.linalg.norm(np.array(c) - np.array(z)) for z in global_centers]
                    nearest_cluster = distances.index(min(distances))
                    
                    X_train = X_1[labels1 == nearest_cluster]
                    X_test = X_2[labels2 == i]
                    if len(X_train) == 0:
                        continue
                    if len(X_train) == 1:
                        new_i2 = global_centers_count
                        global_centers_count += 1
                        df2.loc[df2['l'] == i, 'll'] = new_i2
                        continue
                    if self.AD_algo_merge == "LOF":
                        ad = LocalOutlierFactor(n_neighbors=2, novelty=True).fit(X_train)
                    elif self.AD_algo_merge == "IF":
                        ad = IsolationForest().fit(X_train)
                    elif self.AD_algo_merge == "EE":
                        ad = EllipticEnvelope().fit(X_train)
                    elif self.AD_algo_merge == "OCSVM":
                        ad = OneClassSVM(nu=0.1).fit(X_train)
                    predict = ad.predict(X_test)
                    if np.mean(predict) > 0:
                        new_i2 = nearest_cluster
                    else:
                        new_i2 = global_centers_count
                        global_centers_count += 1
                    df2.loc[df2['l'] == i, 'll'] = new_i2
                
                df2 = df2.drop("l", axis=1)
                df2 = df2.rename(columns={'ll': 'l'})
                
                df = pd.concat([df1, df2])
                df.to_csv(file2, index=False)
            
            csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
            csv_files.sort()
            
        csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
        df = pd.read_csv(csv_files[0])
        
        df = df.rename(columns={'l': 'ACE_Labels'})
        df.to_csv("ClusteringOutput/"+self.fileName+"_"+self.algoName+".csv", index=False)
        
        
                        
    
    def reassignAnomalies(self, df):
        X = df.drop(["y", "ACE_Labels"], axis=1).to_numpy()
        centers = []
        ll = df["ACE_Labels"].tolist()
        unique_labels = set(ll)
        centers_count = len(unique_labels)
        
        for i in unique_labels:
            if i != -1:
                centers.append(X[ll == i].mean(axis=0))
        indexes_to_reassign = [index for index, element in enumerate(ll) if element == -1]
        
        for i in indexes_to_reassign:
            distances = [np.linalg.norm(np.array(X[i]) - np.array(z)) for z in centers]
            nearest_cluster = distances.index(min(distances))
            ll[i] = nearest_cluster
        return ll
    
    def AUL_ARI(self, deleteAnomalies=False):
        df = pd.read_csv("ClusteringOutput/"+self.fileName+"_"+self.algoName+".csv")
        yy = df["y"].tolist()
        ll = df["ACE_Labels"].tolist()
        old_ari = adjusted_rand_score(yy, ll)
        
        indexes_to_delete = [index for index, element in enumerate(ll) if element == -1]
        if len(indexes_to_delete) != 0:
            ll = self.reassignAnomalies(df)
        else:
            return old_ari
        if deleteAnomalies:
            ll = [value for index, value in enumerate(ll) if index not in indexes_to_delete]
            yy = [value for index, value in enumerate(yy) if index not in indexes_to_delete]
        ari = adjusted_rand_score(yy, ll)
        
        return ari
    
    def run(self):
        t0 = time.time()

            
        if self.batch_count == 0:
            if self.X.shape[0] < 10000:    
                self.batch_count = 20
            elif self.X.shape[0] > 10000 and self.X.shape[0] < 100000:    
                self.batch_count = 100
            else:
                self.batch_count = int(self.X.shape[0]/100000)*100
        
        self.subSample()
        if self.algoName == "DBSCAN":
            self.set_DBSCAN_param()
        self.determineParam()
        self.subSample()
        self.rerun()
        t1 = time.time()
        ari_ss = self.AUL_ARI()
        time_ss = t1-t0
        print("Folder \"ClusteringOutput\" contains the output labels.")
        return ari_ss, time_ss
        
    