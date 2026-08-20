# ACE Clustering (Local & Distributed)

This repository contains the implementation of the **ACE (Approximate Consensus Ensemble)** clustering framework. It supports running standard algorithms locally or in a distributed manner across AWS EC2 and S3 instances.

## Environments & Prerequisites
* **Python**: 3.9+
* **Packages**: `pandas`, `numpy`, `scipy`, `scikit-learn`, `boto3`

---

## Configuration & Credentials (For AWS Mode)

To run the pipeline in AWS mode, you must set up your credentials and environment configuration.

1. **Local AWS Profile**:
   Ensure you have configured your AWS CLI credentials on your local machine:
   ```bash
   aws configure
   ```

2. **Project Configuration File**:
   Copy `config.template.json` to `config.json` (this file is ignored by Git to protect your credentials):
   ```bash
   cp config.template.json config.json
   ```
   Edit the newly created `config.json` file and enter your values:
   * `AMI_ID`: Your custom pre-baked EC2 AMI (with python libraries pre-installed).
   * `REGION`: The AWS region of your resources (e.g. `us-east-2`).
   * `BUCKET_NAME`: The S3 bucket name where datasets and results will be transferred.
   * `INSTANCE_TYPE`: The EC2 instance type for workers (default: `t3.micro`).

---

## Usage

Run the unified `run.py` script at the root of the repository with your chosen mode and parameters.

### 1. Local Mode
Runs the clustering algorithm locally on your machine sequentially.
```bash
python run.py local Dataset/letter.csv HAC
```

### 2. AWS Mode (Distributed)
Splits the dataset, launches EC2 worker instances in parallel to perform clustering, pulls results, performs the hierarchical merge locally, and automatically terminates all instances and flushes the S3 bucket.
```bash
python run.py aws Dataset/letter.csv HAC
```

### Command Line Arguments
* `mode`: `local` or `aws`
* `file_path`: Path to your local CSV dataset file (e.g., `Dataset/letter.csv`).
* `algo`: The clustering algorithm name (`HAC`, `DBSCAN`, `GMM`, `SC`, `AP`).
* `--n-clusters`: Number of clusters to find (default: `3`).
* `--bucket`: Override S3 bucket name.
* `--workers`: Number of EC2 instances to launch in parallel (default: `4`).
