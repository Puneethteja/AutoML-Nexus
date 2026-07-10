import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelSelector:
    @staticmethod
    def select_best(results: Dict[str, Any], task: str) -> str:

        if task == "regression":
            best_model = min(results, key=lambda k: results[k]["error"])
        else:
            best_model = max(results, key=lambda k: results[k]["score"])
            
        logger.info(f"Best model selected: {best_model} for task: {task}")
        return best_model