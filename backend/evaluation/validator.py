from sklearn.model_selection import cross_val_score

class ModelValidator:
    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds

    def evaluate(self, model, x, y, scoring: str):
        scores = cross_val_score(model, x, y, cv=self.cv_folds, scoring=scoring)
        return scores.mean()