# model_utils.py
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

MODEL_DIR = "models"

class DummyAutoencoder:
    def predict(self, X):
        # identity-like predictor: returns input unchanged
        return X

def _default_feature_cols():
    # sensible defaults matching feature engineering in `fraud_demo.py`/app
    cols = ['log_amount', 'is_international', 'hour', 'cust_mean_amt', 'cust_std_amt', 'cust_tx_count', 'ratio_to_mean']
    cols += ['channel_web', 'channel_pos', 'channel_atm']
    cols += ['merchant_cat_travel', 'merchant_cat_electronics', 'merchant_cat_utilities', 'merchant_cat_others']
    return cols

def load_artifacts():
    """Load model artifacts if present. If files are missing, create quick fallback artifacts so the API can run without long training or TensorFlow.
    Returns: scaler, iso, autoencoder, feature_cols
    """
    # feature columns
    feature_cols_path = os.path.join(MODEL_DIR, "feature_columns.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    iso_path = os.path.join(MODEL_DIR, "isolation_forest.joblib")
    ae_path = os.path.join(MODEL_DIR, "autoencoder.keras")

    feature_cols = None
    if os.path.exists(feature_cols_path):
        feature_cols = joblib.load(feature_cols_path)
    else:
        feature_cols = _default_feature_cols()

    # scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        # create a quick scaler fitted on random data
        X_dummy = np.random.RandomState(1).randn(100, len(feature_cols))
        scaler = StandardScaler().fit(X_dummy)
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(scaler, scaler_path)
        except Exception:
            pass

    # isolation forest
    if os.path.exists(iso_path):
        iso = joblib.load(iso_path)
    else:
        X_dummy = scaler.transform(np.random.RandomState(2).randn(200, len(feature_cols)))
        iso = IsolationForest(n_estimators=50, contamination=0.01, random_state=1)
        iso.fit(X_dummy)
        try:
            joblib.dump(iso, iso_path)
        except Exception:
            pass

    # autoencoder: try to load a Keras model if present, otherwise use a dummy
    if os.path.exists(ae_path):
        try:
            from tensorflow import keras
            autoencoder = keras.models.load_model(ae_path)
        except Exception:
            autoencoder = DummyAutoencoder()
    else:
        autoencoder = DummyAutoencoder()

    return scaler, iso, autoencoder, feature_cols

def fe_single(tx: dict, feature_cols: list):
    # tx is a simple dict with keys: amount, hour, is_international, channel, merchant_cat, customer_id
    # For single-transaction inference we approximate customer aggregates with placeholders
    # In production: read customer aggregates from feature store
    base = {
        'amount': float(tx.get('amount', 0.0)),
        'hour': int(tx.get('hour', 0)),
        'is_international': int(bool(tx.get('is_international', False))),
        'customer_id': tx.get('customer_id', 'cust_0'),
        'channel': tx.get('channel', 'mobile'),
        'merchant_cat': tx.get('merchant_cat', 'others')
    }
    # build dataframe with single row
    df = pd.DataFrame([base])
    # simple placeholder aggregates: set cust_mean_amt to amount (so ratio ~1) for neutral behavior
    df['cust_mean_amt'] = base['amount']
    df['cust_std_amt'] = 0.0
    df['cust_tx_count'] = 1.0
    df['ratio_to_mean'] = df['amount'] / (1 + df['cust_mean_amt'])
    # log amount
    df['log_amount'] = np.log1p(df['amount'])
    # one-hot columns based on feature_cols
    # find all columns in feature_cols that start with channel_ or merchant_cat_
    for c in feature_cols:
        if c.startswith('channel_'):
            df[c] = 1.0 if c == f"channel_{base['channel']}" else 0.0
        if c.startswith('merchant_cat_'):
            df[c] = 1.0 if c == f"merchant_cat_{base['merchant_cat']}" else 0.0
    # ensure order and missing cols filled
    X = df.reindex(columns=feature_cols, fill_value=0.0)
    return X.astype(float)
