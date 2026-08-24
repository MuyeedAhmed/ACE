# ACE Clustering (Local & Distributed)

[![CI Pipeline](https://github.com/MuyeedAhmed/ACE/actions/workflows/ci.yml/badge.svg)](https://github.com/MuyeedAhmed/ACE/actions/workflows/ci.yml)
[![CI Distributed Pipeline](https://github.com/MuyeedAhmed/ACE/actions/workflows/ci-distributed.yml/badge.svg)](https://github.com/MuyeedAhmed/ACE/actions/workflows/ci-distributed.yml)
[![CD Build and Publish Docker](https://github.com/MuyeedAhmed/ACE/actions/workflows/cd-docker.yml/badge.svg)](https://github.com/MuyeedAhmed/ACE/actions/workflows/cd-docker.yml)

This repository contains the implementation of the **ACE** framework. It supports running standard algorithms locally or in a distributed manner across AWS EC2 and S3 instances.:
* **Local Parallel Mode**: Distributes the workload across concurrent threads on a single local machine.
* **AWS Distributed Mode**: Distributes the workload across independent AWS EC2 worker instances (nodes), coordinated via S3.

This repository contains the official implementation of the paper:
[**ACE: Algorithm-Independent Acceleration and Parallelization of Clustering Implementations**](https://link.springer.com/chapter/10.1007/978-3-031-85697-6_11) (PPAM24).

[![Paper DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--031--85697--6__11-blue.svg)](https://doi.org/10.1007/978-3-031-85697-6_11)

## Architecture

![ACE Architecture](Figures/Architecture.png)

ACE operates in a black-box, algorithm-independent manner, requiring only the black-box clustering implementation and its parameter ranges. The framework consists of four main stages:
1. **Data Partitioning**: Shuffles and divides the dataset $D$ into $n$ smaller partitions ($S_1, S_2, \dots, S_n$) to reduce memory usage and enable parallelization.
2. **Parallel Parameter Tuning (HAPV)**: Conducts a parallel search to determine the hyperparameter values with the highest accuracy (HAPV) using a validation clustering algorithm.
3. **Label Generation**: Concurrently runs the clustering algorithm on all partitions using the tuned HAPV parameters.
4. **Label Merging**: Integrates the local partition clustering labels ($l_1, l_2, \dots, l_n$) into a unified global clustering output ($l$).

## Environments & Prerequisites
* **Python**: 3.9+
* **Packages**: `pandas`, `numpy`, `scipy`, `scikit-learn`, `boto3`

---

## Configuration & Credentials (For AWS Mode)

The system uses an **IAM Instance Profile (Role)** attached to the EC2 instances.

1. **Local AWS Profile**:
   Ensure you have configured your AWS CLI credentials on your local machine:
   ```bash
   aws configure
   ```

2. **Create the IAM Role and Instance Profile**:
   You can easily set up the required IAM role, policy, and instance profile restricted strictly to the project's S3 bucket by running the helper script:
   ```bash
   bash Setup/setup_iam.sh
   ```
   This will create a custom policy called `ACE-S3-Bucket-Policy` and an EC2 role/instance profile called `ACE-EC2-S3-Role`.

3. **Project Configuration File**:
   Copy `config.template.json` to `config.json` (this file is ignored by Git to protect your configuration settings):
   ```bash
   cp config.template.json config.json
   ```
   Edit `config.json` and enter your values:
   * `AMI_ID`: Your custom pre-baked EC2 AMI (with python libraries pre-installed).
   * `REGION`: The AWS region of your resources (e.g. `us-east-2`).
   * `BUCKET_NAME`: The S3 bucket name where datasets and results will be transferred.
   * `INSTANCE_TYPE`: The EC2 instance type for workers (default: `t3.micro`).
   * `IAM_INSTANCE_PROFILE`: The name of the IAM Instance Profile you created (e.g. `ACE-EC2-S3-Role`).

---

## Usage

Run the unified `run.py` script at the root of the repository with your chosen mode and parameters.

### 1. Local Parallel Mode
Runs the clustering algorithm locally on your machine, parallelizing the workload across multiple threads.
```bash
python run.py local Dataset/letter.csv HAC
```

### 2. AWS Distributed Mode
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

---

## Testing

To run the automated tests locally, make sure you have the dependencies installed and run `pytest`:
```bash
python -m pytest tests/
```

## REST API Interface

The ACE framework can be run as a RESTful web service (`app.py`) to expose clustering-as-a-service (CaaS) capabilities. 

### Launching the Service
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoints

#### 1. Execute Clustering Job (`POST /cluster`)
Accepts a dataset file and runs clustering in either local or AWS execution modes. 
* **Local Mode (`mode: local`)**: Free and requires no authentication.
* **AWS Mode (`mode: aws`)**: Requires a valid API key passed in the `X-API-Key` header.
* **One-Time Key Policy**: Temporary keys are marked as spent in `keys.json` immediately upon their first AWS execution to prevent billing abuse.

**Example Request (Local Mode):**
```bash
curl -X POST "http://localhost:8000/cluster" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@Dataset/letter.csv" \
  -F "algo=HAC" \
  -F "mode=local" \
  -F "n_clusters=3"
```

#### 2. Temporary Key Generation (`POST /admin/generate-key`)
Used by administrators to generate one-time use API keys (UUIDs) for AWS runs.
* **Authentication**: Requires the master admin key (`aws-secret-admin-key`) in the `X-API-Key` header.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/admin/generate-key" \
  -H "X-API-Key: aws-secret-admin-key"
```

## Docker Usage


Pulling from GitHub Container Registry (GHCR):

```bash
# Pull the latest image
docker pull ghcr.io/muyeedahmed/ace:latest

# Run local clustering (mounting local Dataset and ClusteringOutput directories)
docker run -v "$(pwd)/Dataset:/app/Dataset" -v "$(pwd)/ClusteringOutput:/app/ClusteringOutput" ghcr.io/muyeedahmed/ace:latest local Dataset/letter.csv HAC
```

Building the Image Locally:

```bash
# Build the image
docker build -t ace-clustering .

# Run the local image
docker run -v "$(pwd)/Dataset:/app/Dataset" -v "$(pwd)/ClusteringOutput:/app/ClusteringOutput" ace-clustering local Dataset/letter.csv HAC
```

---

## Citation

```bibtex
@inproceedings{ppam24ahmed,
   author = {Ahmed, Muyeed and Neamtiu, Iulian},
   title = {ACE: Algorithm-Independent Acceleration and&nbsp;Parallelization of&nbsp;Clustering Implementations},
   year = {2024},
   isbn = {978-3-031-85696-9},
   publisher = {Springer-Verlag},
   address = {Berlin, Heidelberg},
   url = {https://doi.org/10.1007/978-3-031-85697-6_11},
   doi = {10.1007/978-3-031-85697-6_11},
   booktitle = {Parallel Processing and Applied Mathematics: 15th International Conference, PPAM 2024},
   pages = {161–176},
   location = {Ostrava, Czech Republic}
}
```
