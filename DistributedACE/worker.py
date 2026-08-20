import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import boto3
from sklearn.cluster import AffinityPropagation, SpectralClustering, AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed ACE Worker")
    parser.add_argument("--bucket", type=str, default="ma-njit-ace", help="S3 bucket name")
    parser.add_argument("--s3-key", type=str, default="dataset/letter.csv", help="S3 key of the dataset")
    parser.add_argument("--batch-index", type=int, required=True, help="Batch index to process")
    parser.add_argument("--batch-count", type=int, required=True, help="Total number of batches")
    parser.add_argument("--algo", type=str, default="HAC", help="Clustering algorithm name (HAC, DBSCAN, GMM, SC, AP)")
    parser.add_argument("--n-clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--params", type=str, default="{}", help="JSON string of algorithm hyperparameters")
    return parser.parse_args()

def download_from_s3(bucket, key, local_path):
    print(f"Downloading s3://{bucket}/{key} to {local_path}...")
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local_path)
    print("Download completed.")

def upload_to_s3(local_path, bucket, key):
    print(f"Uploading {local_path} to s3://{bucket}/{key}...")
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, key)
    print("Upload completed.")

def run_clustering(X, algo, n_clusters, params_dict):
    print(f"Running clustering algorithm: {algo} with parameters: {params_dict}")
    
    if algo == "AP":
        damping = params_dict.get("damping", 0.5)
        max_iter = params_dict.get("max_iter", 200)
        convergence_iter = params_dict.get("convergence_iter", 15)
        model = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter)
        labels = model.fit_predict(X)
        
    elif algo == "SC":
        eigen_solver = params_dict.get("eigen_solver", "arpack")
        n_components = params_dict.get("n_components", None)
        n_init = params_dict.get("n_init", 10)
        gamma = params_dict.get("gamma", 1.0)
        affinity = params_dict.get("affinity", "rbf")
        n_neighbors = params_dict.get("n_neighbors", 10)
        assign_labels = params_dict.get("assign_labels", "kmeans")
        degree = params_dict.get("degree", 3)
        n_jobs = params_dict.get("n_jobs", None)
        
        model = SpectralClustering(
            n_clusters=n_clusters, eigen_solver=eigen_solver, n_components=n_components,
            n_init=n_init, gamma=gamma, affinity=affinity, n_neighbors=n_neighbors,
            assign_labels=assign_labels, degree=degree, n_jobs=n_jobs
        )
        labels = model.fit_predict(X)
        
    elif algo == "GMM":
        covariance_type = params_dict.get("covariance_type", "full")
        tol = params_dict.get("tol", 1e-3)
        reg_covar = params_dict.get("reg_covar", 1e-6)
        max_iter = params_dict.get("max_iter", 100)
        n_init = params_dict.get("n_init", 1)
        init_params = params_dict.get("init_params", "kmeans")
        warm_start = params_dict.get("warm_start", False)
        
        model = GaussianMixture(
            n_components=n_clusters, covariance_type=covariance_type, tol=tol,
            reg_covar=reg_covar, max_iter=max_iter, n_init=n_init,
            init_params=init_params, warm_start=warm_start
        )
        labels = model.fit_predict(X)
        
    elif algo == "HAC":
        metric = params_dict.get("metric", "euclidean")
        linkage = params_dict.get("linkage", "ward")
        model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
        labels = model.fit_predict(X)
        
    elif algo == "DBSCAN":
        eps = params_dict.get("eps", 0.5)
        min_samples = params_dict.get("min_samples", 5)
        algorithm = params_dict.get("algorithm", "auto")
        model = DBSCAN(eps=eps, min_samples=min_samples, algorithm=algorithm)
        labels = model.fit_predict(X)
        
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
        
    return labels

def main():
    args = parse_args()
    local_data_path = f"/tmp/letter_{args.batch_index}.csv"    
    download_from_s3(args.bucket, args.s3_key, local_data_path)
    
    df = pd.read_csv(local_data_path)    
    batch_size = int(len(df) / args.batch_count)
    start_idx = args.batch_index * batch_size
    if args.batch_index == args.batch_count - 1:
        df_batch = df.iloc[start_idx:].copy()
    else:
        df_batch = df.iloc[start_idx:start_idx + batch_size].copy()
        
    print(f"Batch {args.batch_index}: Processing {len(df_batch)} records (indices {start_idx} to {start_idx + len(df_batch)}).")
    
    if "class" in df_batch.columns:
        y = df_batch["class"].to_numpy()
        X = df_batch.drop("class", axis=1)
    else:
        y = np.zeros(len(df_batch), dtype=int)
        X = df_batch.copy()
        
    X.fillna(X.mean(numeric_only=True).round(1), inplace=True)
    
    params_dict = json.loads(args.params)
    labels = run_clustering(X, args.algo, args.n_clusters, params_dict)
    
    df_batch["y"] = y
    df_batch["l"] = labels
    
    local_output_path = f"/tmp/result_{args.batch_index}.csv"
    df_batch.to_csv(local_output_path, index=False)
    
    s3_output_key = f"output/{args.batch_index}.csv"
    upload_to_s3(local_output_path, args.bucket, s3_output_key)
    
    try:
        os.remove(local_data_path)
        os.remove(local_output_path)
    except Exception:
        pass
        
    print(f"Worker {args.batch_index} run successfully completed.")

if __name__ == "__main__":
    main()
