import logging
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self) -> None:
        self.config = ConfigLoader.load("config\\config.yaml")
        self.test_size = self.config.get("test_size", 0.2)
        self.val_size = self.config.get("validation_size", 0.1)
        self.random_state = self.config.get("random_state", 42)
        self.stratify = True

    def split(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        problem_type: str = "classification"
    ) -> Tuple[pd.DataFrame, ...]:
        
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
            
        stratify_col = y if (self.stratify and problem_type == "classification") else None

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )

        if self.val_size > 0:
            val_ratio = self.val_size / (1 - self.test_size)
            stratify_val = y_train if (self.stratify and problem_type == "classification") else None

            x_train, x_val, y_train, y_val = train_test_split(
                x_train,
                y_train,
                test_size=val_ratio,
                random_state=self.random_state,
                stratify=stratify_val
            )
            
            logger.info(f"Split complete: Train({len(x_train)}), Val({len(x_val)}), Test({len(x_test)})")
            return x_train, x_val, x_test, y_train, y_val, y_test

        logger.info(f"Split complete: Train({len(x_train)}), Test({len(x_test)})")
        return x_train, x_test, y_train, y_test