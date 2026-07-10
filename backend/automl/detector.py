import logging
import pandas as pd
import numpy as np
from typing import Literal

logger = logging.getLogger(__name__)

class ProblemDetector:
    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def detect(self, y: pd.Series) -> Literal["classification", "regression"]:
        if y.dtype == "object" or y.dtype.name == "category" or y.dtype == "bool":
            return "classification"
            
        unique_ratio = y.nunique() / len(y)
        
        if unique_ratio <= self.threshold or y.nunique() <= 20:
            logger.info(f"Detected classification (unique ratio: {unique_ratio:.4f})")
            return "classification"

        logger.info(f"Detected regression (unique ratio: {unique_ratio:.4f})")
        return "regression"