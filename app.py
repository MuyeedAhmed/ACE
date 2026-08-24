import os
import sys
import shutil
import subprocess
import tempfile
import uuid
import json
import datetime
import threading
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, status, BackgroundTasks
from fastapi.responses import FileResponse

app = FastAPI(
    title="ACE Clustering API",
    description="API for Algorithm-independent Clustering Acceleration (Local & Distributed Mode)",
    version="1.0.0"
)

API_KEY = os.getenv("ACE_API_KEY", "aws-secret-admin-key")
KEYS_FILE = "keys.json"
keys_lock = threading.Lock()

def save_keys(keys_data):
    with keys_lock:
        with open(KEYS_FILE, "w") as f:
            json.dump(keys_data, f, indent=2)

def load_keys():
    master_key = os.getenv("ACE_API_KEY", "aws-secret-admin-key")
    if not os.path.exists(KEYS_FILE):
        data = {
            "keys": {
                master_key: {
                    "one_time": False,
                    "used": False,
                    "created_at": str(datetime.date.today())
                }
            }
        }
        save_keys(data)
        return data
        
    with keys_lock:
        with open(KEYS_FILE, "r") as f:
            try:
                data = json.load(f)
                # Ensure master key is always present in memory/state
                if "keys" not in data:
                    data = {"keys": {}}
                if master_key not in data["keys"]:
                    data["keys"][master_key] = {
                        "one_time": False,
                        "used": False,
                        "created_at": str(datetime.date.today())
                    }
                return data
            except json.JSONDecodeError:
                return {"keys": {}}

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the ACE Clustering API",
        "docs_url": "/docs",
        "note": "AWS distributed mode requires an API key. To get a one-time use API key, please send an email request to ma234@njit.edu.",
        "endpoints": {
            "POST /cluster": "Upload a CSV dataset and run clustering (Local or AWS mode)",
            "GET /algorithms": "List available clustering algorithms",
            "POST /admin/generate-key": "Generate a new temporary one-time use API key (Admin only)"
        }
    }

@app.get("/algorithms")
def get_algorithms():
    return {
        "HAC": "Hierarchical Agglomerative Clustering",
        "DBSCAN": "Density-Based Spatial Clustering of Applications with Noise",
        "GMM": "Gaussian Mixture Model",
        "SC": "Spectral Clustering",
        "AP": "Affinity Propagation"
    }

@app.post("/admin/generate-key")
def generate_key(x_api_key: Optional[str] = Header(None)):
    master_key = os.getenv("ACE_API_KEY", "aws-secret-admin-key")
    if not x_api_key or x_api_key != master_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized. Master API key required in X-API-Key header."
        )
        
    new_key = f"ace-temp-{uuid.uuid4()}"
    keys_data = load_keys()
    keys_data["keys"][new_key] = {
        "one_time": True,
        "used": False,
        "created_at": str(datetime.date.today())
    }
    save_keys(keys_data)
    return {"api_key": new_key, "detail": "One-time use API key generated successfully."}

@app.post("/cluster")
def run_clustering_endpoint(
    file: UploadFile = File(...),
    algo: str = Form(...),
    mode: str = Form("local"),
    n_clusters: int = Form(3),
    workers: int = Form(4),
    background_tasks: BackgroundTasks = None,
    x_api_key: Optional[str] = Header(None)
):
    if algo not in ["HAC", "DBSCAN", "GMM", "SC", "AP"]:
        raise HTTPException(status_code=400, detail=f"Invalid algorithm: {algo}. Must be one of HAC, DBSCAN, GMM, SC, AP.")
    
    if mode not in ["local", "aws"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'local' or 'aws'.")
    
    if mode == "aws":
        if not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header. AWS mode is restricted."
            )
            
        keys_data = load_keys()
        all_keys = keys_data.get("keys", {})
        
        if x_api_key not in all_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key. Access denied."
            )
            
        key_info = all_keys[x_api_key]
        if key_info.get("used", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="This API Key has already been used. Please request a new one."
            )
            
        if key_info.get("one_time", False):
            key_info["used"] = True
            save_keys(keys_data)
            
    temp_dir = tempfile.mkdtemp()
    
    file_id = str(uuid.uuid4())
    temp_input_csv = os.path.join(temp_dir, f"{file_id}.csv")
    
    config_created = False
    if not os.path.exists("config.json"):
        shutil.copy("config.template.json", "config.json")
        config_created = True
    
    try:
        with open(temp_input_csv, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        cmd = [
            sys.executable, "run.py", mode,
            temp_input_csv,
            algo,
            "--n-clusters", str(n_clusters)
        ]
        
        if mode == "aws":
            cmd += ["--workers", str(workers)]
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Clustering run failed:\nStdout: {result.stdout}\nStderr: {result.stderr}"
            )
            
        expected_output = f"ClusteringOutput/{file_id}_{algo}.csv"
        
        if not os.path.exists(expected_output):
            raise HTTPException(
                status_code=500,
                detail=f"Expected output file {expected_output} was not created.\nStdout: {result.stdout}"
            )
            
        local_output_path = os.path.join(temp_dir, f"result_{algo}.csv")
        shutil.move(expected_output, local_output_path)
        
        if config_created and os.path.exists("config.json"):
            os.remove("config.json")
            config_created = False
            
        if background_tasks:
            background_tasks.add_task(shutil.rmtree, temp_dir)
        
        return FileResponse(
            path=local_output_path,
            filename=f"ace_clustered_{file.filename}",
            media_type="text/csv"
        )
        
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if config_created and os.path.exists("config.json"):
            os.remove("config.json")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
