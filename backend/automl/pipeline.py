import logging
import os
import joblib
import numpy as np
import random
import hashlib
import json
import pkg_resources
import datetime
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
        self.detector = ProblemDetector()
        self.config = ConfigLoader.load("config/config.yaml")
        self._set_global_seed(self.config.get("random_state", 42))

    def _set_global_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _generate_manifest(self, dataset_path, best_params, test_score):
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
            "best_params": best_params
        }
        
        os.makedirs("logs/experiments", exist_ok=True)
        manifest_path = f"logs/experiments/run_{manifest['run_id']}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
        return manifest_path

    def run(self, x, y, dataset_path):
        best_model = None
        
        try:
            x = x.sort_index()
            y = y.sort_index()
            task = self.detector.detect(y)
            x_processed = self.preprocessor.fit_transform(x)
            x_train, x_val, x_test, y_train, y_val, y_test = self.splitter.split(x_processed, y, task)
            
            engine = OptunaEngine(task)
            best_params = engine.run(x_train, y_train)
            
            cv_scores = engine.get_cv_scores(x_train, y_train, best_params)
            mean_acc = np.mean(cv_scores)
            std_dev = np.std(cv_scores)
            
            model_params = {k: v for k, v in best_params.items() if k != "model_type"}
            best_model = MLModelFactory.get_model(best_params["model_type"], task, **model_params)
            
            if best_model is None:
                raise ValueError("MLModelFactory failed to instantiate the model.")
                
            best_model.fit(x_train, y_train)
            test_score = best_model.score(x_test, y_test)
            
            os.makedirs("models", exist_ok=True)
            joblib.dump(best_model, "models/best_model.pkl")
            
            manifest_path = self._generate_manifest(dataset_path, best_params, test_score)
            
            return best_params, mean_acc, test_score, std_dev, manifest_path

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise