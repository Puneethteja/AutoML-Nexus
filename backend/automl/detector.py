import logging
import pandas as pd
import numpy as np
from typing import Literal

logger = logging.getLogger(__name__)

class ProblemDetector:

    def __init__(self, threshold: int = 20) -> None:
        self.threshold = threshold

    def detect(self, y: pd.Series) -> Literal["classification", "regression"]:
        
        if y.dtype == "object" or y.dtype == "category" or y.dtype == "bool":
            logger.info("Task detected as classification based on dtype.")
            return "classification"

        unique_count = y.nunique()
        
        if unique_count <= self.threshold:
            logger.info(f"Task detected as classification (unique values: {unique_count}).")
            return "classification"

        logger.info(f"Task detected as regression (unique values: {unique_count}).")
        return "regression"