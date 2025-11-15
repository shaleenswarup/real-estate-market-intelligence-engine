"""ML-based price prediction engine using ensemble methods."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor
import joblib

class PricePredictorEnsemble:
    """Ensemble price prediction with XGBoost, GradientBoosting, RandomForest."""
    
    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()
        self.ensemble = None
        
    def prepare_features(self, X_train, X_test):
        """Scale and prepare features."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def build_ensemble(self, X_train, y_train):
        """Build ensemble model combining multiple algorithms."""
        xgb = XGBRegressor(max_depth=7, learning_rate=0.1, n_estimators=200)
        gb = GradientBoostingRegressor(max_depth=5, learning_rate=0.1, n_estimators=150)
        rf = RandomForestRegressor(n_estimators=100, max_depth=15)
        
        self.ensemble = VotingRegressor(
            estimators=[('xgb', xgb), ('gb', gb), ('rf', rf)],
            weights=[0.5, 0.3, 0.2]
        )
        self.ensemble.fit(X_train, y_train)
        return self.ensemble
    
    def predict(self, X):
        """Make price predictions."""
        X_scaled = self.scaler.transform(X)
        return self.ensemble.predict(X_scaled)
    
    def cross_validate(self, X, y, cv=5):
        """Evaluate model with cross-validation."""
        scores = cross_val_score(self.ensemble, X, y, cv=cv, scoring='r2')
        return {'mean_r2': scores.mean(), 'std': scores.std(), 'scores': scores}
