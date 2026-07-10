import logging
import math
from unittest import result
import uuid
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.automl.pipeline import AutoMLPipeline
from backend.utils.logger import get_logger
from backend.utils.helper import set_global_seed

app = FastAPI()
logger = get_logger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

task_store = {}

@app.post("/start-training")
async def start_training(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    task_id = str(uuid.uuid4())
    temp_path = f"temp_{task_id}.csv"
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
        
    task_store[task_id] = {"status": "processing", "progress": 20, "result": None, "error": None}
    
    background_tasks.add_task(run_automl_async, task_id, temp_path, target_column)
    return {"task_id": task_id}
def clean_data(obj):

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    elif isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data(v) for v in obj]
    return obj
def run_automl_async(task_id: str, data_path: str, target_column: str):
    try:
        def update_task_progress(p):
            task_store[task_id]["progress"] = p

        task_store[task_id]["status"] = "processing"
        df = pd.read_csv(data_path)
        
        pipeline = AutoMLPipeline()
        
        best_params, train_acc, test_acc, std_dev, manifest_path, history = pipeline.run(
            df.drop(columns=[target_column]), 
            df[target_column], 
            data_path,
            progress_callback=update_task_progress
        )
        
        task_store[task_id] = {
            "status": "completed", 
            "progress": 100,
            "result": clean_data({ 
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "std_dev": std_dev,
                "params": best_params,
                "history": history
            })
            
        }
        return {"status": "completed", "result": task_store[task_id]["result"]}
    except Exception as e:
        task_store[task_id] = {"status": "failed", "error": str(e)}
@app.get("/status/{task_id}")
async def get_status(task_id: str):
    return task_store.get(task_id, {"status": "not_found", "progress": 0})
@app.get("/")
async def root():
    return {"message": "AutoML-Nexus Backend is Online"}