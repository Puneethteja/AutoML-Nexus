from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

class MLModelFactory:
    @staticmethod
    def get_model(model_name: str, task_type: str, **kwargs):
        if task_type == "classification":
            models = {
                "logistic": LogisticRegression,
                "rf": RandomForestClassifier,
                "svm": SVC,
                "xgb": XGBClassifier
            }
        else:
            models = {
                "rf": RandomForestRegressor,
                "svm": SVR,
                "xgb": XGBRegressor
            }
        
        return models[model_name](**kwargs)