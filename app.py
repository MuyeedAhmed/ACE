import os
import sys
import shutil
import subprocess
import tempfile
import uuid
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, status, BackgroundTasks
from fastapi.responses import FileResponse

app = FastAPI(
    title="ACE Clustering API",
    description="API for Algorithm-independent Clustering Acceleration (Local & Distributed Mode)",
    version="1.0.0"
)

API_KEY = os.getenv("ACE_API_KEY", "aws-secret-admin-key")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the ACE Clustering API",
        "docs_url": "/docs",
        "endpoints": {
            "POST /cluster": "Upload a CSV dataset and run clustering (Local or AWS mode)",
            "GET /algorithms": "List available clustering algorithms"
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
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header. AWS mode is restricted."
            )
            
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
