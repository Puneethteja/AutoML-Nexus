import optuna
import logging
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from backend.models.ml_models import MLModelFactory
from backend.optimization.search_space import get_search_space
from backend.utils.config_loader import ConfigLoader

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
            
            x_train = np.nan_to_num(x_train, nan=0.0) 
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_train)

            model = MLModelFactory.get_model(model_type, self.task_type, **params)
            
            if self.task_type == "classification":
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
                scoring = 'r2'

            scores = cross_val_score(model, x_scaled, y_train, cv=cv, scoring=scoring)
            mean_score = scores.mean()
            
            return -np.inf if np.isnan(mean_score) else mean_score
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return -np.inf

    def run(self, x_train, y_train, n_trials: int = 20, progress_callback=None):
        def optuna_callback(study, trial):
            if progress_callback:
                progress = int((((trial.number + 1) / n_trials) * 80) + 20)
                progress_callback(progress)

        self.study.optimize(
            lambda trial: self.objective(trial, x_train, y_train), 
            n_trials=n_trials, 
            callbacks=[optuna_callback] if progress_callback else None
        )
        
        history = []
        for trial in self.study.trials:
            if trial.value is not None:
                history.append({
                    "trial": trial.number + 1, 
                    "accuracy": float(trial.value)
                })
        
        if len(self.study.trials) == 0 or self.study.best_trial is None:
            raise RuntimeError("Optimization failed: No trials were completed successfully.")
            
        return self.study.best_params, history

    def get_cv_scores(self, x_train, y_train, best_params):
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)
        model = MLModelFactory.get_model(
            best_params["model_type"], 
            self.task_type, 
            **{k: v for k, v in best_params.items() if k != "model_type"}
        )
        
        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            scoring = 'r2'
            
        return cross_val_score(model, x_scaled, y_train, cv=cv, scoring=scoring)