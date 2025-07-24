import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from typing import Dict, Optional, Any
from pathlib import Path
import os

class ModelUtils:
    @staticmethod
    def train_demand_model(X: pd.DataFrame, y: pd.Series, save_path: Path) -> None:
        """Trains and saves the demand prediction model."""
        if not save_path.parent.exists():
            os.makedirs(save_path.parent, exist_ok=True)
            
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X, y)
        
        print(f"Model trained and saved to {save_path}")
        joblib.dump(model, save_path)

    @staticmethod
    def load_model(model_path: Path) -> Optional[Any]:
        """Loads a saved model from disk."""
        try:
            if model_path.exists():
                return joblib.load(model_path)
        except Exception as e:
            print(f"❗️ Error loading model from {model_path}: {e}")
        return None

    @staticmethod
    def predict_with_fallback(model: Any, features: Dict, fallback_func) -> float:
        """Predicts using the model, but uses a fallback function if it fails."""
        if not model:
            return fallback_func(features)
        try:
            input_df = pd.DataFrame([features])
            # Ensure column order is the same as the one the model was trained on
            input_df = input_df[model.feature_names_in_]
            prediction = model.predict(input_df)[0]
            return float(prediction)
        except Exception as e:
            print(f"❗️ Model prediction failed, using fallback. Error: {e}")
            return fallback_func(features)