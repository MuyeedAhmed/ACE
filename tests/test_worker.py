import os
import sys
import shutil
import tempfile
import json
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DistributedACE")))
import worker

def test_run_clustering():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
    
    labels = worker.run_clustering(X, "HAC", n_clusters=2, params_dict={})
    assert len(labels) == 10
    assert len(set(labels)) <= 2

    labels = worker.run_clustering(X, "GMM", n_clusters=2, params_dict={})
    assert len(labels) == 10

def test_worker_main_execution():
    temp_dir = tempfile.mkdtemp()
    dummy_csv = os.path.join(temp_dir, "dummy_batch.csv")
    
    np.random.seed(42)
    data = np.random.rand(15, 2)
    df = pd.DataFrame(data, columns=["x1", "x2"])
    df["class"] = np.random.choice([0, 1], size=15)
    df.to_csv(dummy_csv, index=False)
    
    def mock_download(bucket, key, local_path):
        shutil.copy(dummy_csv, local_path)

    uploaded_files = []
    def mock_upload(local_path, bucket, key):
        uploaded_files.append(local_path)
        out_df = pd.read_csv(local_path)
        assert "y" in out_df.columns
        assert "l" in out_df.columns
        assert len(out_df) == 15

    test_args = [
        "worker.py",
        "--bucket", "dummy-bucket",
        "--s3-key", "dummy-key",
        "--batch-index", "0",
        "--algo", "HAC",
        "--n-clusters", "2",
        "--params", json.dumps({"metric": "euclidean", "linkage": "ward"})
    ]

    try:
        with patch.object(sys, 'argv', test_args), \
             patch('worker.download_from_s3', side_effect=mock_download), \
             patch('worker.upload_to_s3', side_effect=mock_upload):
            worker.main()
            
        assert len(uploaded_files) == 1
        
    finally:
        shutil.rmtree(temp_dir)
