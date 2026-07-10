import logging
import os
import joblib
import numpy as np
import random
import hashlib
import json
import pkg_resources
import datetime
from sklearn.metrics import r2_score, accuracy_score
from backend.data.preprocessing import DataPreprocessor
from backend.data.splitter import DataSplitter
from backend.automl.detector import ProblemDetector
from backend.optimization.optuna_engine import OptunaEngine
from backend.models.ml_models import MLModelFactory
from backend.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class AutoMLPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.splitter = DataSplitter()
        self.detector = ProblemDetector(threshold=0.05)
        self.config = ConfigLoader.load("config/config.yaml")
        self._set_global_seed(self.config.get("random_state", 42))
        self.seed = self.config.get("random_state", 42)
        
    def _set_global_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _generate_manifest(self, dataset_path, best_params, test_score, history):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Cannot audit: {dataset_path} not found.")

        with open(dataset_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        relevant_pkgs = ['scikit-learn', 'numpy', 'pandas', 'optuna']
        env = {pkg.key: pkg.version for pkg in pkg_resources.working_set if pkg.key in relevant_pkgs}
        
        manifest = {
            "run_id": file_hash[:8],
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset_hash": file_hash,
            "environment": env,
            "results": {"test_score": float(test_score)},
            "best_params": best_params,
            "optimization_history": history
        }
        
        os.makedirs("logs/experiments", exist_ok=True)
        manifest_path = f"logs/experiments/run_{manifest['run_id']}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
        return manifest_path

    def run(self, x, y, dataset_path, progress_callback=None):
        try:
            task = self.detector.detect(y)
            logger.info(f"Pipeline initialized. Task detected: {task}")
            
            x_processed = self.preprocessor.fit_transform(x)
            x_train, x_val, x_test, y_train, y_val, y_test = self.splitter.split(x_processed, y, task)
            
            engine = OptunaEngine(task)
            best_params, history = engine.run(x_train, y_train, progress_callback=progress_callback)
            
            model_params = {k: v for k, v in best_params.items() if k != "model_type"}
            best_model = MLModelFactory.get_model(best_params["model_type"], task, **model_params)
            best_model.fit(x_train, y_train)
            
            cv_scores = engine.get_cv_scores(x_train, y_train, best_params)
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            
            if task == "classification":
                test_score = accuracy_score(y_test, best_model.predict(x_test))
            else:
                test_score = r2_score(y_test, best_model.predict(x_test))
            
            os.makedirs("models", exist_ok=True)
            joblib.dump(best_model, "models/best_model.pkl")
            
            manifest_path = self._generate_manifest(dataset_path, best_params, test_score, history)
            
            return best_params, mean_cv_score, test_score, std_cv_score, manifest_path, history

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise