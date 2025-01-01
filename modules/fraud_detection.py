# order_processing_system/fraud_detection.py

import logging
import joblib
import pandas as pd
from config import FRAUD_MODEL_PATH

def load_fraud_detection_model():
    """
    Loads the pre-trained fraud detection model.
    """
    try:
        model = joblib.load(FRAUD_MODEL_PATH)
        logging.info("Fraud detection model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading fraud detection model: {e}")
        return None

def feature_engineering(df):
    """
    Performs feature engineering on the order data.
    """
    try:
        df['total_customer_spent'] = df.groupby('customer_id')['transaction_amount'].transform('sum')
        df['total_customer_orders'] = df.groupby('customer_id')['transaction_id'].transform('count')
        df['avg_transaction_amount'] = df['total_customer_spent'] / df['total_customer_orders']
        df['order_frequency'] = df.groupby('customer_id')['transaction_date'].transform(
            lambda x: (pd.to_datetime(x).max() - pd.to_datetime(x).min()).days if pd.to_datetime(x).max() != pd.to_datetime(x).min() else 0
        )
        logging.info("Feature engineering completed.")
        return df
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        return df

def detect_fraud(order_details, model):
    """
    Detects if an order is potentially fraudulent using the trained model.
    """
    try:
        # Convert order details into a DataFrame for prediction
        order_df = pd.DataFrame([order_details])

        # Perform feature engineering
        order_df = feature_engineering(order_df)

        # Select features used in training the model
        features = ['total_customer_spent', 'total_customer_orders', 'avg_transaction_amount', 'order_frequency', 'transaction_amount']

        # Check if all required features are present
        if not all(feature in order_df.columns for feature in features):
            logging.error("Missing features for fraud detection.")
            return False, 0.0

        # Predict fraud probability
        fraud_prob = model.predict_proba(order_df[features])[:, 1]  # Probability of fraud
        is_fraud = fraud_prob >= 0.9  # Threshold for flagging fraud

        logging.info(f"Fraud detection: is_fraud={is_fraud[0]}, fraud_prob={fraud_prob[0]:.2f}")
        return is_fraud[0], fraud_prob[0]
    except Exception as e:
        logging.error(f"Error detecting fraud: {e}")
        return False, 0.0
