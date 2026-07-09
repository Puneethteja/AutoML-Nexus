import logging
import joblib
import pandas as pd
from typing import Dict, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self) -> None:
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []

    def _detect_column_types(self, x_df: pd.DataFrame) -> None:
        # Detect types
        self.numeric_features = x_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_features = x_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        logger.info(f"Detected {len(self.numeric_features)} num and {len(self.categorical_features)} cat features.")

    def fit_transform(self, x_df: pd.DataFrame) -> pd.DataFrame:
        self._detect_column_types(x_df)
        
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer([
            ("num", numeric_transformer, self.numeric_features),
            ("cat", categorical_transformer, self.categorical_features)
        ])
        
        return self.preprocessor.fit_transform(x_df)

    def transform(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet!")
        return self.preprocessor.transform(x_df)

    def save(self, path: str) -> None:
        data_to_save = {
            "preprocessor": self.preprocessor,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features
        }
        joblib.dump(data_to_save, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.preprocessor = data["preprocessor"]
        self.numeric_features = data["numeric_features"]
        self.categorical_features = data["categorical_features"]