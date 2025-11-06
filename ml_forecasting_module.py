"""
Advanced ML Forecasting Engine for Energy Demand Prediction
Goal: Achieve MAPE ≤ 5% for best-in-market performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Optional: Uncomment if you have these libraries installed
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")


class EnergyDemandForecaster:
    """
    Advanced forecasting system for energy demand prediction.
    Implements multiple ML models with sophisticated feature engineering.
    """
    
    def __init__(self, target_mape=5.0):
        self.target_mape = target_mape
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.best_model = None
        self.performance_metrics = {}
        
    def create_advanced_features(self, df, target_col='load'):
        """
        Comprehensive feature engineering for time series forecasting
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # ===== Temporal Features =====
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Binary temporal indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        df['is_peak_hour'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # ===== Cyclical Encoding (Critical for time series) =====
        # Hour of day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of year (seasonality)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # ===== Lag Features (Multiple horizons) =====
        # 5-minute intervals: 12 per hour, 288 per day, 2016 per week
        lag_periods = [
            1, 2, 3,          # Recent (5, 10, 15 min ago)
            12, 24, 36,       # Hourly (1, 2, 3 hours ago)
            288, 576,         # Daily (1, 2 days ago)
            2016,             # Weekly (1 week ago)
        ]
        
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # ===== Rolling Window Features =====
        windows = [12, 24, 288, 2016]  # 1h, 2h, 1d, 1w
        
        for window in windows:
            # Rolling mean
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).mean()
            
            # Rolling std (volatility)
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).std()
            
            # Rolling min/max
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).max()
        
        # ===== Exponential Moving Averages =====
        df[f'{target_col}_ema_12'] = df[target_col].ewm(span=12, adjust=False).mean()
        df[f'{target_col}_ema_288'] = df[target_col].ewm(span=288, adjust=False).mean()
        
        # ===== Difference Features (Trend) =====
        df[f'{target_col}_diff_1'] = df[target_col].diff(1)
        df[f'{target_col}_diff_12'] = df[target_col].diff(12)
        df[f'{target_col}_diff_288'] = df[target_col].diff(288)
        
        # ===== Interaction Features =====
        df['hour_x_dow'] = df['hour'] * df['day_of_week']
        df['hour_x_month'] = df['hour'] * df['month']
        df['is_weekend_x_hour'] = df['is_weekend'] * df['hour']
        
        # ===== Statistical Features =====
        # Same hour last week
        df[f'{target_col}_same_hour_last_week'] = df[target_col].shift(2016)
        
        # Difference from yesterday same time
        df[f'{target_col}_daily_change'] = df[target_col] - df[target_col].shift(288)
        
        # Drop NaN values created by lag features
        df = df.dropna()
        
        return df
    
    def prepare_data(self, df, target_col='load', test_size=0.2):
        """
        Prepare data for training with train/test split
        """
        # Create features
        df_featured = self.create_advanced_features(df, target_col)
        
        # Identify feature columns (exclude timestamp and target)
        exclude_cols = ['timestamp', target_col]
        if 'price' in df_featured.columns:
            exclude_cols.append('price')
        
        self.feature_cols = [col for col in df_featured.columns if col not in exclude_cols]
        
        # Time series split (no shuffling)
        split_idx = int(len(df_featured) * (1 - test_size))
        
        train_data = df_featured.iloc[:split_idx]
        test_data = df_featured.iloc[split_idx:]
        
        X_train = train_data[self.feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[self.feature_cols]
        y_test = test_data[target_col]
        
        return X_train, X_test, y_train, y_test, test_data
    
    def train_models(self, X_train, y_train, optimize_hyperparameters=False):
        """
        Train multiple ML models
        """
        print("=" * 60)
        print("Training Models...")
        print("=" * 60)
        
        # 1. Ridge Regression (Baseline)
        print("\n[1/5] Training Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        self.models['Ridge'] = ridge
        
        # 2. Lasso (Feature selection)
        print("[2/5] Training Lasso Regression...")
        lasso = Lasso(alpha=0.1, max_iter=2000)
        lasso.fit(X_train, y_train)
        self.models['Lasso'] = lasso
        
        # 3. Histogram Gradient Boosting (Fast and accurate)
        print("[3/5] Training Histogram Gradient Boosting...")
        if optimize_hyperparameters:
            hgb_params = {
                'max_iter': [200, 300],
                'learning_rate': [0.05, 0.1],
                'max_depth': [8, 10],
                'min_samples_leaf': [20, 30]
            }
            hgb = GridSearchCV(
                HistGradientBoostingRegressor(random_state=42),
                hgb_params,
                cv=3,
                scoring='neg_mean_absolute_percentage_error',
                n_jobs=-1
            )
            hgb.fit(X_train, y_train)
            self.models['HGB'] = hgb.best_estimator_
            print(f"   Best params: {hgb.best_params_}")
        else:
            hgb = HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_depth=10,
                min_samples_leaf=20,
                random_state=42
            )
            hgb.fit(X_train, y_train)
            self.models['HGB'] = hgb
        
        # 4. Random Forest (Robust ensemble)
        print("[4/5] Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['RandomForest'] = rf
        
        # 5. LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            print("[5/5] Training LightGBM...")
            lgb = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=10,
                num_leaves=31,
                min_child_samples=20,
                random_state=42,
                verbose=-1
            )
            lgb.fit(X_train, y_train)
            self.models['LightGBM'] = lgb
        
        print("\n✓ Training complete!")
        
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models and identify best performer
        """
        print("\n" + "=" * 60)
        print("Model Evaluation")
        print("=" * 60)
        
        results = []
        best_mape = float('inf')
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results.append({
                'Model': name,
                'MAPE (%)': mape,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'predictions': y_pred
            })
            
            # Track best model
            if mape < best_mape:
                best_mape = mape
                self.best_model = name
            
            # Print results
            status = "✓" if mape <= self.target_mape else "✗"
            print(f"\n{status} {name}:")
            print(f"   MAPE: {mape:.3f}% (Target: ≤{self.target_mape}%)")
            print(f"   MAE:  {mae:.2f} MW")
            print(f"   RMSE: {rmse:.2f} MW")
            print(f"   R²:   {r2:.4f}")
        
        self.performance_metrics = pd.DataFrame(results)
        
        print("\n" + "=" * 60)
        print(f"🏆 Best Model: {self.best_model} (MAPE: {best_mape:.3f}%)")
        print("=" * 60)
        
        return self.performance_metrics
    
    def create_ensemble(self, X_test, method='weighted_average'):
        """
        Create ensemble predictions from all models
        """
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X_test)
        
        if method == 'simple_average':
            # Simple average
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            
        elif method == 'weighted_average':
            # Weight by inverse MAPE (better models get higher weight)
            weights = {}
            total_weight = 0
            
            for _, row in self.performance_metrics.iterrows():
                weight = 1 / (row['MAPE (%)'] + 0.001)  # Add small value to avoid division by zero
                weights[row['Model']] = weight
                total_weight += weight
            
            # Normalize weights
            weights = {k: v/total_weight for k, v in weights.items()}
            
            # Weighted sum
            ensemble_pred = np.zeros(len(X_test))
            for name, pred in predictions.items():
                ensemble_pred += weights[name] * pred
        
        return ensemble_pred
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from tree-based models
        """
        if self.best_model in ['HGB', 'RandomForest', 'LightGBM']:
            model = self.models[self.best_model]
            
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': self.feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(top_n)
                
                return importance_df
        
        return None
    
    def forecast_future(self, recent_data, horizon_hours=24):
        """
        Generate forecasts for future time periods
        """
        # Use best model for forecasting
        model = self.models[self.best_model]
        
        # Prepare features for forecasting
        # This is simplified - in production, you'd need to handle rolling predictions
        forecast_data = self.create_advanced_features(recent_data)
        X_forecast = forecast_data[self.feature_cols].tail(horizon_hours * 12)
        
        predictions = model.predict(X_forecast)
        
        return predictions


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """
    Example usage of the Energy Demand Forecaster
    """
    print("\n🚀 Energy Demand Forecasting System")
    print("Goal: Achieve MAPE ≤ 5% for best-in-market performance\n")
    
    # Load your data here
    # df = pd.read_csv('your_energy_data.csv')
    # For demo purposes, generate sample data
    
    print("📊 Loading data...")
    # Replace this with your actual data loading
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30*24*12, freq='5min')
    
    # Generate realistic load pattern
    hour_of_day = dates.hour + dates.minute/60
    base_load = 25000 + 8000 * np.sin((hour_of_day - 6) * np.pi / 12)
    solar_impact = -5000 * np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12)) * (hour_of_day > 8) * (hour_of_day < 18)
    noise = np.random.normal(0, 500, len(dates))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'load': base_load + solar_impact + noise
    })
    
    print(f"✓ Loaded {len(df)} records ({len(df)/(12*24):.1f} days)")
    
    # Initialize forecaster
    forecaster = EnergyDemandForecaster(target_mape=5.0)
    
    # Prepare data
    print("\n🔧 Engineering features...")
    X_train, X_test, y_train, y_test, test_data = forecaster.prepare_data(df)
    print(f"✓ Created {len(forecaster.feature_cols)} features")
    print(f"✓ Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train models
    forecaster.train_models(X_train, y_train, optimize_hyperparameters=False)
    
    # Evaluate
    results = forecaster.evaluate_models(X_test, y_test)
    
    # Create ensemble
    print("\n🎯 Creating ensemble prediction...")
    ensemble_pred = forecaster.create_ensemble(X_test, method='weighted_average')
    ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
    print(f"✓ Ensemble MAPE: {ensemble_mape:.3f}%")
    
    # Feature importance
    importance = forecaster.get_feature_importance(top_n=15)
    if importance is not None:
        print("\n📊 Top 15 Most Important Features:")
        print(importance.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("🎉 Forecasting system ready for deployment!")
    print("=" * 60)
    
    return forecaster, results


if __name__ == "__main__":
    forecaster, results = main()
