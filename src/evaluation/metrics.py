from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

class MetricsCalculator:
    @staticmethod
    def get_metrics(y_true, y_pred, task_type: str):
        if task_type == "classification":
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average="weighted")
            }
        else:
            return {
                "mse": mean_squared_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred)
            }