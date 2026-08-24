import os
import sys
import shutil
import subprocess
import tempfile
import pandas as pd
import numpy as np
import pytest
from ACE.ACE_Clustering import ACE_Clustering

def test_algo_parameters():
    ace = ACE_Clustering(algoName="HAC", n_cluster=2)
    params = ace.parameters
    assert len(params) == 2
    assert params[0][0] == "metric"
    assert params[1][0] == "linkage"

    ace_dbscan = ACE_Clustering(algoName="DBSCAN", n_cluster=2)
    params_dbscan = ace_dbscan.parameters
    assert len(params_dbscan) == 2
    assert params_dbscan[0][0] == "eps_min_samples"

def test_cli_local_run():
    config_created = False
    if not os.path.exists("config.json"):
        shutil.copy("config.template.json", "config.json")
        config_created = True
        
    temp_dir = tempfile.mkdtemp()
    dummy_csv = os.path.join(temp_dir, "dummy.csv")
    
    np.random.seed(42)
    data = np.random.rand(400, 2)
    df = pd.DataFrame(data, columns=["x1", "x2"])
    df["class"] = np.random.choice([0, 1], size=400)
    df.to_csv(dummy_csv, index=False)
    
    try:
        cmd = [
            sys.executable, "run.py", "local",
            dummy_csv,
            "HAC",
            "--n-clusters", "3"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Verify output CSV is created under ClusteringOutput/
        expected_output = "ClusteringOutput/dummy_HAC.csv"
        assert os.path.exists(expected_output)
        os.remove(expected_output)
        
    finally:
        shutil.rmtree(temp_dir)
        if config_created and os.path.exists("config.json"):
            os.remove("config.json")
