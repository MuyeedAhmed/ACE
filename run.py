import argparse
import os
import sys
import time
import glob
import json
import boto3
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

# Add current directory to path so we can import ACE
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ACE.ACE_Clustering import ACE_Clustering

# Load configuration from config.json
CONFIG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.json")
if not os.path.exists(CONFIG_PATH):
    print("Error: config.json not found.", file=sys.stderr)
    print("Please copy config.template.json to config.json and fill in your AWS details.", file=sys.stderr)
    sys.exit(1)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

AMI_ID = config.get("AMI_ID")
REGION = config.get("REGION", "us-east-2")
INSTANCE_TYPE = config.get("INSTANCE_TYPE", "t3.micro")
DEFAULT_BUCKET = config.get("BUCKET_NAME")
IAM_INSTANCE_PROFILE = config.get("IAM_INSTANCE_PROFILE")

POLL_INTERVAL = 10  # seconds
TEMP_DIR = "DistributedACE/temp_output"

def parse_arguments():
    parser = argparse.ArgumentParser(description="ACE Clustering Runner (Local & Distributed)")
    parser.add_argument("mode", choices=["local", "aws"], help="Execution mode (local or aws)")
    parser.add_argument("file_path", type=str, help="Path to the local dataset CSV file")
    parser.add_argument("algo", choices=["HAC", "DBSCAN", "GMM", "SC", "AP"], help="Clustering algorithm name")
    parser.add_argument("--n-clusters", type=int, default=3, help="Number of clusters (default: 3)")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET, help="S3 bucket name (for AWS mode)")
    parser.add_argument("--workers", type=int, default=4, help="Number of EC2 workers (for AWS mode)")
    return parser.parse_args()

# ==========================================
# AWS MODE HELPER FUNCTIONS
# ==========================================

def upload_file_to_s3(s3_client, local_path, bucket, s3_key):
    print(f"Uploading {local_path} to s3://{bucket}/{s3_key}...")
    s3_client.upload_file(local_path, bucket, s3_key)

def launch_ec2_workers(bucket, filename, worker_script_key, algo, n_clusters, num_workers, params_json):
    ec2 = boto3.resource('ec2', region_name=REGION)
    instances = []
    
    print(f"Launching {num_workers} EC2 worker instances in parallel...")
    for i in range(num_workers):
        split_s3_key = f"dataset/splits/{filename}_{i}.csv"
        
        user_data = f"""#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "=== Start Worker {i} Setup ==="

# Ensure instance terminates itself on script exit (success or failure)
trap 'echo "=== Initiating self-termination ==="; shutdown -h now' EXIT

# Configure default region to avoid AWS CLI region prompts or failures
mkdir -p /root/.aws
cat <<EOF > /root/.aws/config
[default]
region = {REGION}
EOF

mkdir -p /home/ec2-user/.aws
cp /root/.aws/config /home/ec2-user/.aws/config
chown -R ec2-user:ec2-user /home/ec2-user/.aws

# Wait up to 30 seconds for S3 network connectivity to become active
for attempt in {{1..15}}; do
    if curl -sI https://s3.{REGION}.amazonaws.com >/dev/null; then
        echo "Network connectivity to S3 is active."
        break
    fi
    echo "Waiting for S3 network connectivity... (attempt \\$attempt)"
    sleep 2
done

# 1. Download code from S3 with retries (in case IAM role is still propagating)
echo "=== Downloading worker script from S3 ==="
for attempt in {{1..12}}; do
    if aws s3 cp s3://{bucket}/{worker_script_key} /home/ec2-user/worker.py --region {REGION}; then
        echo "Successfully downloaded worker script."
        break
    fi
    echo "S3 download failed. Retrying in 5 seconds... (attempt \\$attempt)"
    sleep 5
done
chown ec2-user:ec2-user /home/ec2-user/worker.py

# 2. Execute clustering script passing tuned parameters
echo "=== Running worker script for batch {i} ==="
su - ec2-user -c "python3 /home/ec2-user/worker.py --bucket {bucket} --s3-key {split_s3_key} --batch-index {i} --algo {algo} --n-clusters {n_clusters} --params '{params_json}'"
echo "=== Worker {i} Run Finished ==="
"""
        create_params = {
            'ImageId': AMI_ID,
            'MinCount': 1,
            'MaxCount': 1,
            'InstanceType': INSTANCE_TYPE,
            'UserData': user_data,
            'InstanceInitiatedShutdownBehavior': 'terminate',
            'TagSpecifications': [{
                'ResourceType': 'instance',
                'Tags': [{'Key': 'Name', 'Value': f'ACE-Worker-{i}'}]
            }]
        }
        if IAM_INSTANCE_PROFILE:
            create_params['IamInstanceProfile'] = {'Name': IAM_INSTANCE_PROFILE}
            
        instance = ec2.create_instances(**create_params)[0]
        instances.append(instance)
        print(f"Launched worker {i} (Instance ID: {instance.id})")
        
    return instances

def wait_for_s3_results(s3_client, bucket, num_workers, instances=None):
    expected_files = [f"output/{i}.csv" for i in range(num_workers)]
    start_time = time.time()
    
    while True:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix="output/")
        existing_keys = [obj['Key'] for obj in response.get('Contents', [])]
        
        completed = all(key in existing_keys for key in expected_files)
        if completed:
            break
            
        # Check if any EC2 instances terminated or stopped prematurely
        if instances:
            for inst in instances:
                try:
                    inst.reload()
                except Exception:
                    pass  # Ignore transient API reload issues
            
            states = [inst.state['Name'] for inst in instances]
            if any(state in ['terminated', 'stopped', 'shutting-down'] for state in states):
                # Double-check one final time if all files are in S3 (in case they uploaded just before shutdown)
                response = s3_client.list_objects_v2(Bucket=bucket, Prefix="output/")
                existing_keys = [obj['Key'] for obj in response.get('Contents', [])]
                if all(key in existing_keys for key in expected_files):
                    break
                
                raise RuntimeError(f"One or more EC2 worker instances terminated or stopped prematurely. Instance states: {dict(zip([inst.id for inst in instances], states))}")
        
        elapsed = int(time.time() - start_time)
        print(f"Checking S3 output/ folder... ({len(existing_keys)}/{num_workers} files completed, elapsed: {elapsed}s)")
        time.sleep(POLL_INTERVAL)

def download_s3_results(s3_client, bucket, num_workers):
    print("Downloading intermediate outputs...")
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    for i in range(num_workers):
        local_path = os.path.join(TEMP_DIR, f"{i}.csv")
        s3_client.download_file(bucket, f"output/{i}.csv", local_path)

def constant_k_merge_logic(output_path):
    print("Merging batch results...")
    csv_files = glob.glob(os.path.join(TEMP_DIR, "*.csv"))
    csv_files.sort(key=lambda f: int(os.path.basename(f).split('.')[0]))
    
    while len(csv_files) > 1:
        next_round = []
        for i in range(0, len(csv_files) - 1, 2):
            file1, file2 = csv_files[i], csv_files[i+1]
            df1, df2 = pd.read_csv(file1), pd.read_csv(file2)
            
            X_1 = df1.drop(["y", "l"], axis=1).to_numpy()
            labels1 = df1["l"].to_numpy()
            X_2 = df2.drop(["y", "l"], axis=1).to_numpy()
            labels2 = df2["l"].to_numpy()
            
            centers1 = [X_1[labels1 == lbl].mean(axis=0) for lbl in set(labels1)]
            
            df2["ll"] = -2
            for lbl in set(labels2):
                c = X_2[labels2 == lbl].mean(axis=0)
                distances = [np.linalg.norm(c - z) for z in centers1]
                nearest = distances.index(min(distances))
                df2.loc[df2['l'] == lbl, 'll'] = nearest
                
            df2 = df2.drop("l", axis=1).rename(columns={'ll': 'l'})
            df_merged = pd.concat([df1, df2])
            df_merged.to_csv(file2, index=False)
            os.remove(file1)
            next_round.append(file2)
            
        if len(csv_files) % 2 != 0:
            next_round.append(csv_files[-1])
            
        csv_files = next_round
        csv_files.sort(key=lambda f: int(os.path.basename(f).split('.')[0]))
        
    df_final = pd.read_csv(csv_files[0])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.rename(csv_files[0], output_path)
    print(f"Final merged output saved locally to: {output_path}")
    return df_final

def cleanup_aws(s3_client, bucket, instances):
    print("=== Cleaning Up AWS Resources ===")
    if instances:
        ec2 = boto3.client('ec2', region_name=REGION)
        ids = [inst.id for inst in instances]
        print(f"Terminating instances: {ids}...")
        ec2.terminate_instances(InstanceIds=ids)
        
    print(f"Deleting files in S3 bucket: {bucket}...")
    for prefix in ["output/", "dataset/", "code/"]:
        res = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        keys = [{'Key': obj['Key']} for obj in res.get('Contents', [])]
        if keys:
            s3_client.delete_objects(Bucket=bucket, Delete={'Objects': keys})
            
    print("Cleanup completed.")

# ==========================================
# MAIN
# ==========================================

def main():
    args = parse_arguments()
    if not os.path.exists(args.file_path):
        print(f"Error: Local file {args.file_path} not found.", file=sys.stderr)
        sys.exit(1)
        
    filename = os.path.splitext(os.path.basename(args.file_path))[0]
    
    df = pd.read_csv(args.file_path)
    if "class" in df.columns:
        y = df["class"].to_numpy()
        X = df.drop("class", axis=1)
    else:
        y = np.zeros(df.shape[0])
        X = df.copy()
    X.fillna(X.mean(numeric_only=True).round(1), inplace=True)
    
    if args.mode == "local":
        print(f"--- Running ACE locally (Dataset: {args.file_path}, Algorithm: {args.algo}) ---")
        
        clustering = ACE_Clustering(algoName=args.algo, fileName=filename, n_cluster=args.n_clusters)
        clustering.X = X
        clustering.y = y
        ari, time_elapsed = clustering.run()
        clustering.destroy()
        
        print(f"Local ACE Run Complete. Time: {time_elapsed:.2f}s, Adjusted Rand Score (ARI): {ari:.4f}")
        
    elif args.mode == "aws":
        print(f"--- Running Distributed ACE on AWS EC2 (Dataset: {args.file_path}, Algorithm: {args.algo}) ---")
        
        # Check if IAM Instance Profile is set
        if not IAM_INSTANCE_PROFILE:
            print("Error: IAM_INSTANCE_PROFILE is not set in config.json.", file=sys.stderr)
            print("Please create an IAM Role (Instance Profile) with S3 read/write permissions,", file=sys.stderr)
            print("add its name to config.json, and try again.", file=sys.stderr)
            sys.exit(1)
            
        print("Determining best parameters...")
        clustering = ACE_Clustering(algoName=args.algo, fileName=filename, n_cluster=args.n_clusters, batch_count=args.workers)
        clustering.X = X
        clustering.y = y
        clustering.subSample()
        if args.algo == "DBSCAN":
            clustering.set_DBSCAN_param()
        clustering.determineParam()
        best_params = clustering.bestParams
        clustering.destroy()
        
        params_dict = {}
        if args.algo == "HAC":
            params_dict = {"metric": str(best_params[0]), "linkage": str(best_params[1])}
        elif args.algo == "AP":
            params_dict = {"damping": float(best_params[0]), "max_iter": int(best_params[1]), "convergence_iter": int(best_params[2])}
        elif args.algo == "GMM":
            params_dict = {
                "covariance_type": str(best_params[0]), "tol": float(best_params[1]), "reg_covar": float(best_params[2]),
                "max_iter": int(best_params[3]), "n_init": int(best_params[4]), "init_params": str(best_params[5]),
                "warm_start": bool(best_params[6])
            }
        elif args.algo == "SC":
            params_dict = {
                "eigen_solver": str(best_params[0]), "n_components": int(best_params[1]) if best_params[1] is not None else None, "n_init": int(best_params[2]),
                "gamma": float(best_params[3]), "affinity": str(best_params[4]), "n_neighbors": int(best_params[5]),
                "assign_labels": str(best_params[6]), "degree": int(best_params[7]), "n_jobs": int(best_params[8]) if best_params[8] is not None else None
            }
        elif args.algo == "DBSCAN":
            eps, min_samples = best_params[0]
            params_dict = {"eps": float(eps), "min_samples": int(min_samples), "algorithm": str(best_params[1])}
            
        print(f"Best parameters selected by ACE: {params_dict}")
        params_json = json.dumps(params_dict).replace('"', '\\"')
        
        s3_client = boto3.client('s3', region_name=REGION)
        worker_script_s3_key = "code/worker.py"
        output_local_path = f"ClusteringOutput/{filename}_{args.algo}_distributed.csv"
        
        instances = []
        try:
            # 1. Split and upload data chunks
            print(f"Splitting dataset into {args.workers} batches and uploading to S3...")
            os.makedirs(TEMP_DIR, exist_ok=True)
            batch_size = int(len(X) / args.workers)
            
            df_to_split = X.copy()
            if "class" in df.columns:
                df_to_split["class"] = y
            else:
                df_to_split["class"] = np.zeros(len(X), dtype=int)
                
            for i in range(args.workers):
                start_idx = i * batch_size
                if i == args.workers - 1:
                    df_batch = df_to_split.iloc[start_idx:].copy()
                else:
                    df_batch = df_to_split.iloc[start_idx:start_idx + batch_size].copy()
                    
                split_local_path = os.path.join(TEMP_DIR, f"{filename}_split_{i}.csv")
                df_batch.to_csv(split_local_path, index=False)
                
                split_s3_key = f"dataset/splits/{filename}_{i}.csv"
                upload_file_to_s3(s3_client, split_local_path, args.bucket, split_s3_key)
                os.remove(split_local_path)
            
            # 2. Upload worker script to S3
            upload_file_to_s3(s3_client, "DistributedACE/worker.py", args.bucket, worker_script_s3_key)
            
            # 3. Launch workers
            instances = launch_ec2_workers(
                args.bucket, filename, worker_script_s3_key, 
                args.algo, args.n_clusters, args.workers, params_json
            )
            
            wait_for_s3_results(s3_client, args.bucket, args.workers, instances)
            
            download_s3_results(s3_client, args.bucket, args.workers)
            df_final = constant_k_merge_logic(output_local_path)
            
            score = adjusted_rand_score(df_final["y"].tolist(), df_final["l"].tolist())
            print(f"AWS Distributed ACE Run Complete. Adjusted Rand Score (ARI): {score:.4f}")
            
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user. Cleaning up AWS resources...")
        except Exception as e:
            print(f"An error occurred during AWS execution: {e}", file=sys.stderr)

        finally:
            cleanup_aws(s3_client, args.bucket, instances)

if __name__ == "__main__":
    main()
