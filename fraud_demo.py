# fraud_demo.py
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

def gen_synthetic_transactions(n_customers=1000, n_tx=50000, fraud_rate=0.002):
    customers = [f'cust_{i}' for i in range(n_customers)]
    rows = []
    for i in range(n_tx):
        cust = random.choice(customers)
        base_amount = max(1, np.random.exponential(scale=50))
        cust_mult = 1 + (int(cust.split("_")[1]) % 5) * 0.2
        amount = base_amount * cust_mult
        hour = np.random.randint(0,24)
        is_international = np.random.rand() < 0.02
        channel = np.random.choice(['mobile','web','pos','atm'], p=[0.4,0.25,0.25,0.1])
        merchant_cat = np.random.choice(['grocery','travel','electronics','utilities','others'],
                                        p=[0.4,0.15,0.15,0.15,0.15])
        label = 0
        if np.random.rand() < fraud_rate:
            label = 1
            amount *= np.random.uniform(5, 50)
            is_international = True if np.random.rand() < 0.5 else is_international
            channel = 'web' if np.random.rand() < 0.7 else channel
        rows.append({
            'transaction_id': f"tx_{i}",
            'customer_id': cust,
            'amount': float(amount),
            'hour': hour,
            'is_international': int(is_international),
            'channel': channel,
            'merchant_cat': merchant_cat,
            'label': label
        })
    return pd.DataFrame(rows)

def feature_engineer(df):
    # log amount
    df['log_amount'] = np.log1p(df['amount'])
    # one-hot small categories
    df = pd.get_dummies(df, columns=['channel','merchant_cat'], drop_first=True)
    # customer-level simple aggregates
    cust_stats = df.groupby('customer_id')['amount'].agg(['mean','std','count']).reset_index()
    cust_stats.columns = ['customer_id','cust_mean_amt','cust_std_amt','cust_tx_count']
    df = df.merge(cust_stats, on='customer_id', how='left')
    df['ratio_to_mean'] = df['amount'] / (1 + df['cust_mean_amt'])
    # pick features used by inference
    features = ['log_amount', 'is_international', 'hour', 'cust_mean_amt','cust_std_amt','cust_tx_count','ratio_to_mean']
    features += [c for c in df.columns if c.startswith('channel_') or c.startswith('merchant_cat_')]
    X = df[features].fillna(0).astype(float)
    return X, df

def train_save_models():
    print("Generating synthetic data...")
    df = gen_synthetic_transactions()
    X, df_fe = feature_engineer(df)
    y = df_fe['label'].values
    print("Data shapes:", X.shape)

    # scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    print("Training IsolationForest...")
    iso = IsolationForest(n_estimators=200, contamination=0.002, random_state=1)
    iso.fit(X_scaled)
    joblib.dump(iso, f"{MODEL_DIR}/isolation_forest.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    # Save column order for inference
    joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_columns.joblib")
    print("Saved IsolationForest + scaler + feature list.")

    # Autoencoder
    print("Training Autoencoder...")
    input_dim = X_scaled.shape[1]
    encoding_dim = max(4, input_dim // 3)
    encoder_input = keras.Input(shape=(input_dim,))
    x = layers.Dense(encoding_dim*2, activation='relu')(encoder_input)
    x = layers.Dense(encoding_dim, activation='relu')(x)
    x = layers.Dense(encoding_dim*2, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=15, batch_size=512, verbose=1)
    autoencoder.save(f"{MODEL_DIR}/autoencoder.keras")
    print("Saved Autoencoder.")

if __name__ == "__main__":
    train_save_models()
    print("Training complete. Models are in the 'models/' directory.")
