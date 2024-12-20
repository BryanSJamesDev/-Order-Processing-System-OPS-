# order_processing_system/config.py

import os

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'bryansamjames@gmail.com'
EMAIL_PASSWORD = 'abcd efgh ijkl mnop'  # Replace this with your actual app password

# OpenAI API Key setup for Chatbot
OPENAI_API_KEY = 'API Key'  

# Real-time Dashboard Update Interval (in seconds)
DASHBOARD_UPDATE_INTERVAL = 10

# Fraud Detection Model Path
FRAUD_MODEL_PATH = 'fraud_detection_model.pkl'
