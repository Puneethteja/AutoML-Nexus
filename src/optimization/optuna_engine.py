import optuna
import logging
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from src.models.ml_models import MLModelFactory
from src.optimization.search_space import get_search_space
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class OptunaEngine:
    def __init__(self, task_type: str):
        self.config = ConfigLoader.load("config/config.yaml")
        self.seed = self.config.get("random_state", 42)
        self.sampler = optuna.samplers.TPESampler(seed=self.seed)
        self.task_type = task_type
        self.study = optuna.create_study(direction="maximize", sampler=self.sampler)

    def objective(self, trial, x_train, y_train):
        try:
            model_type = trial.suggest_categorical("model_type", ["rf", "xgb"])
            params = get_search_space(trial, model_type, self.task_type)
            
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_train)

            model = MLModelFactory.get_model(model_type, self.task_type, **params)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            scores = cross_val_score(model, x_scaled, y_train, cv=cv, scoring='accuracy')
            
            return scores.mean()
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return -np.inf 

    def run(self, x_train, y_train, n_trials: int = 20):
        self.study.optimize(lambda trial: self.objective(trial, x_train, y_train), n_trials=n_trials)
        
        if len(self.study.trials) == 0 or self.study.best_trial is None:
            raise RuntimeError("Optimization failed: No trials were completed successfully.")
            
        return self.study.best_params

    def get_cv_scores(self, x_train, y_train, best_params):
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)
        model = MLModelFactory.get_model(
            best_params["model_type"], 
            self.task_type, 
            **{k: v for k, v in best_params.items() if k != "model_type"}
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        return cross_val_score(model, x_scaled, y_train, cv=cv, scoring='accuracy')