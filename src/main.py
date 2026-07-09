import logging
import uuid
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.automl.pipeline import AutoMLPipeline
from src.utils.logger import get_logger
from src.utils.helper import set_global_seed

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

def run_automl_async(task_id: str, data_path: str, target_column: str):
    try:
        task_store[task_id]["status"] = "processing"
        print(f"[Task {task_id}] Loading dataset...")
        df = pd.read_csv(data_path)
        
        pipeline = AutoMLPipeline()
        
        print(f"[Task {task_id}] Starting pipeline execution...")
        best_params, train_acc, test_acc, std_dev, manifest = pipeline.run(
            df.drop(columns=[target_column]), df[target_column], data_path
        )
        
        print(f"[Task {task_id}] Completed. Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Std: {std_dev:.4f}")
        
        task_store[task_id] = {
            "status": "completed", 
            "progress": 100,
            "result": {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "std_dev": std_dev,
                "params": best_params
            }
        }
    except Exception as e:
        print(f"[Task {task_id}] ERROR: {str(e)}")
        task_store[task_id] = {"status": "failed", "error": str(e)}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    return task_store.get(task_id, {"status": "not_found", "progress": 0})
@app.get("/")
async def root():
    return {"message": "AutoML-Nexus Backend is Online"}