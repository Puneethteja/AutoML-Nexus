import optuna

def get_search_space(trial: optuna.Trial, model_type: str, task_type: str):
  
    if model_type == "rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 32)
        }
        if task_type == "regression":
            params["criterion"] = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
        else:
            params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
        return params

    elif model_type == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10)
        }
        if task_type == "regression":
            params["objective"] = "reg:squarederror"
        else:
            params["objective"] = "binary:logistic" if task_type == "binary" else "multi:softprob"
        return params

    elif model_type == "ann":
        return {
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "hidden_units": trial.suggest_int("hidden_units", 32, 256),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 10, 100)
        }
        
    return {}