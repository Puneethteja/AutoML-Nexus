import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelSelector:
    @staticmethod
    def select_best(results: Dict[str, Any]) -> str:
        best_model = max(results, key=lambda k: results[k]["score"])
        logger.info(f"Model selection complete. Best model: {best_model}")
        return best_model