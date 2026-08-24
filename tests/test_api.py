import os
import io
import shutil
import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app, API_KEY

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_keys_file():
    KEYS_FILE = "keys.json"
    backup_file = "keys.json.bak"
    has_backup = os.path.exists(KEYS_FILE)
    if has_backup:
        shutil.copy(KEYS_FILE, backup_file)
        os.remove(KEYS_FILE)
    yield
    if os.path.exists(KEYS_FILE):
        os.remove(KEYS_FILE)
    if has_backup:
        shutil.move(backup_file, KEYS_FILE)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_get_algorithms():
    response = client.get("/algorithms")
    assert response.status_code == 200
    assert "HAC" in response.json()

def test_local_clustering_success():
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(400, 2), columns=["x1", "x2"])
    df["class"] = np.random.choice([0, 1], size=400)
    
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    
    response = client.post(
        "/cluster",
        files={"file": ("dummy.csv", csv_bytes, "text/csv")},
        data={"algo": "HAC", "mode": "local", "n_clusters": 3}
    )
    assert response.status_code == 200
    out_df = pd.read_csv(io.StringIO(response.text))
    assert "ACE_Labels" in out_df.columns
    assert len(out_df) == 400

def test_aws_clustering_unauthorized():
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    
    response1 = client.post(
        "/cluster",
        files={"file": ("dummy.csv", csv_bytes, "text/csv")},
        data={"algo": "HAC", "mode": "aws"}
    )
    assert response1.status_code == 401

    csv_bytes.seek(0)
    response2 = client.post(
        "/cluster",
        files={"file": ("dummy.csv", csv_bytes, "text/csv")},
        data={"algo": "HAC", "mode": "aws"},
        headers={"X-API-Key": "wrong-key"}
    )
    assert response2.status_code == 401

def test_admin_generate_key_unauthorized():
    assert client.post("/admin/generate-key").status_code == 401
    assert client.post("/admin/generate-key", headers={"X-API-Key": "wrong"}).status_code == 401

def test_admin_generate_key_success():
    response = client.post("/admin/generate-key", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert "api_key" in response.json()

def test_aws_clustering_one_time_key_flow():
    gen_response = client.post("/admin/generate-key", headers={"X-API-Key": API_KEY})
    temp_key = gen_response.json()["api_key"]
    
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    def mock_subprocess_run(cmd, *args, **kwargs):
        file_id = os.path.splitext(os.path.basename(cmd[3]))[0]
        os.makedirs("ClusteringOutput", exist_ok=True)
        df_out = df.copy()
        df_out["ACE_Labels"] = [0] * 10
        df_out.to_csv(f"ClusteringOutput/{file_id}_HAC.csv", index=False)
        mock_res = MagicMock()
        mock_res.returncode = 0
        return mock_res

    with patch("subprocess.run", side_effect=mock_subprocess_run):
        response1 = client.post(
            "/cluster",
            files={"file": ("dummy.csv", csv_bytes, "text/csv")},
            data={"algo": "HAC", "mode": "aws"},
            headers={"X-API-Key": temp_key}
        )
    assert response1.status_code == 200
    
    csv_bytes.seek(0)
    response2 = client.post(
        "/cluster",
        files={"file": ("dummy.csv", csv_bytes, "text/csv")},
        data={"algo": "HAC", "mode": "aws"},
        headers={"X-API-Key": temp_key}
    )
    assert response2.status_code == 401
    assert "already been used" in response2.json()["detail"]
