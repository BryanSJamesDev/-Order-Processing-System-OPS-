# order_processing_system/utils.py

from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

def hash_password(password):
    """
    Generates a hashed password using Werkzeug's generate_password_hash.
    """
    return generate_password_hash(password)

def verify_password(hashed_password, input_password):
    """
    Verifies an input password against the hashed password.
    """
    return check_password_hash(hashed_password, input_password)

def get_current_timestamp():
    """
    Returns the current timestamp in a standardized format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
