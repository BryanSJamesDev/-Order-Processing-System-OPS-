import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
import pandas as pd
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import threading
import time
import hashlib
import re
import imaplib
import email
import random  
import os      
import datetime

# Machine learning and forecasting imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib  
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from prophet import Prophet
import openai

# ------------------------------
# CONFIGURATION & LOGGING SETUP
# ------------------------------
logging.basicConfig(
    filename='order_system.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

openai.api_key = "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"  

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'bryansamjames@gmail.com'
EMAIL_PASSWORD = 'lctn vhdt fppo awig'

IMAP_SERVER = 'imap.gmail.com'
IMAP_PORT = 993
IMAP_USER = 'bryansamjames@gmail.com'
IMAP_PASS = 'YOUR_IMAP_PASSWORD'

DASHBOARD_UPDATE_INTERVAL = 10  # seconds
FRAUD_MODEL_PATH = 'fraud_detection_model.pkl'

# ------------------------------
# BLOCKCHAIN CLASSES
# ------------------------------
class Block:
    def __init__(self, index, timestamp, data, previous_hash=''):
        self.index = index
        self.timestamp = timestamp
        self.data = data  
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}".encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, time.time(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.hash != current.calculate_hash() or current.previous_hash != previous.hash:
                return False
        return True

# ------------------------------
# EMAIL FUNCTIONS
# ------------------------------
def send_email(subject, body, to_address):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_address
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        logging.info(f"Email sent to {to_address} with subject: {subject}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def fetch_incoming_orders_from_email():
    orders_list = []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(IMAP_USER, IMAP_PASS)
        mail.select('inbox')
        status, data = mail.search(None, '(UNSEEN SUBJECT "New Order")')
        mail_ids = data[0]
        id_list = mail_ids.split()
        for num in id_list:
            status, msg_data = mail.fetch(num, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    raw_email = response_part[1]
                    msg = email.message_from_bytes(raw_email)
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == 'text/plain':
                                body = part.get_payload(decode=True).decode()
                                break
                    else:
                        body = msg.get_payload(decode=True).decode()
                    customer_match = re.search(r'Customer:\s*(.*)', body)
                    product_match = re.search(r'Product:\s*(.*)', body)
                    quantity_match = re.search(r'Quantity:\s*(\d+)', body)
                    if customer_match and product_match and quantity_match:
                        order_data = {
                            'customer_name': customer_match.group(1).strip(),
                            'product_id': product_match.group(1).strip(),
                            'quantity': quantity_match.group(1).strip()
                        }
                        orders_list.append(order_data)
                        logging.info(f"Parsed order from email: {order_data}")
        mail.close()
        mail.logout()
    except Exception as e:
        logging.error(f"Error fetching email orders: {e}")
    return orders_list

# ------------------------------
# DATABASE INITIALIZATION
# ------------------------------
def init_db():
    with sqlite3.connect('order_system.db') as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT,
                role TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_registrations (
                registration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT,
                email TEXT,
                role TEXT,
                status TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                name TEXT,
                price REAL,
                stock INTEGER
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_name TEXT,
                date TEXT,
                status TEXT,
                order_status TEXT DEFAULT 'Processing',
                fraud_flag INTEGER DEFAULT 0,
                fraud_prob REAL DEFAULT 0.0
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS order_details (
                order_id TEXT,
                product_id TEXT,
                quantity INTEGER,
                FOREIGN KEY(order_id) REFERENCES orders(order_id),
                FOREIGN KEY(product_id) REFERENCES products(product_id)
            )
        ''')
        conn.commit()
    logging.info("Database initialized.")

def alter_orders_table():
    try:
        with sqlite3.connect('order_system.db') as conn:
            c = conn.cursor()
            c.execute("ALTER TABLE orders ADD COLUMN fraud_flag INTEGER DEFAULT 0")
            c.execute("ALTER TABLE orders ADD COLUMN fraud_prob REAL DEFAULT 0.0")
            conn.commit()
        logging.info("Orders table altered for fraud detection.")
    except sqlite3.OperationalError as e:
        logging.warning(f"Alter table skipped: {e}")

# ------------------------------
# ADD INITIAL PRODUCTS (HVAC/AC EQUIPMENT)
# ------------------------------
def add_initial_products():
    # Updated HVAC/AC product list with lower prices
    products = [
        ('HV001', 'Central Air Conditioning System', 1500.00, 10),
        ('HV002', 'Ductless Mini Split AC', 700.00, 20),
        ('HV003', 'Industrial HVAC Fan', 300.00, 15),
        ('HV004', 'HVAC Control Panel', 500.00, 8),
        ('HV005', 'Water Cooled AC Unit', 2000.00, 4),
        ('HV006', 'Air Handling Unit (AHU)', 1200.00, 7),
        ('HV007', 'Heat Exchanger Unit', 600.00, 12),
        ('HV008', 'Ventilation Fan', 200.00, 25),
        ('HV009', 'Chiller System', 2500.00, 5),
        ('HV010', 'Energy Recovery Ventilator', 1500.00, 6)
    ]
    try:
        with sqlite3.connect('order_system.db') as conn:
            c = conn.cursor()
            c.executemany("""
                INSERT OR IGNORE INTO products (product_id, name, price, stock)
                VALUES (?, ?, ?, ?)
            """, products)
            conn.commit()
        logging.info("Initial HVAC products added with updated lower prices.")
    except Exception as e:
        logging.error(f"Error adding products: {e}")

def add_initial_users():
    users = [
        ('admin', generate_password_hash('password'), 'admin'),
        ('manager', generate_password_hash('password'), 'manager'),
        ('employee', generate_password_hash('password'), 'employee')
    ]
    try:
        with sqlite3.connect('order_system.db') as conn:
            c = conn.cursor()
            c.executemany("""
                INSERT OR IGNORE INTO users (username, password, role)
                VALUES (?, ?, ?)
            """, users)
            conn.commit()
        logging.info("Initial users added.")
    except Exception as e:
        logging.error(f"Error adding users: {e}")

# ------------------------------
# BATCH ORDER INSERTION: CLEAR AND REPOLLUATE ORDERS
# ------------------------------
# A larger pool of first and last names for realistic customer names.
FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Donald", "Sandra", "Mark", "Ashley",
    "Paul", "Kimberly", "Steven", "Emily", "Andrew", "Donna", "Kenneth", "Michelle",
    "George", "Dorothy", "Joshua", "Carol", "Kevin", "Amanda", "Brian", "Melissa",
    "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Laura"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson",
    "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
    "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee",
    "Walker", "Hall", "Allen", "Young", "King", "Wright", "Scott", "Torres",
    "Nguyen", "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Rivera", "Campbell",
    "Mitchell", "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz"
]


def clear_and_populate_orders(num_orders=5000):
    """
    Clears all existing orders (and their order_details) and repopulates
    the orders table with 'num_orders' new orders. Customer names are generated
    from a large pool of first and last names, orders use only the HVAC products,
    and order datetimes are randomized (date and time) between January 1, 2025
    and April 14, 2025.
    """
    db_path = 'order_system.db'
    
    # Helper function to generate a random datetime between two datetime objects
    def random_datetime(start_dt, end_dt):
        delta = end_dt - start_dt
        int_delta = int(delta.total_seconds())
        random_second_offset = random.randrange(int_delta + 1)
        return start_dt + datetime.timedelta(seconds=random_second_offset)
    
    # Define the datetime range (inclusive)
    start_dt = datetime.datetime(2025, 1, 1, 0, 0, 0)
    end_dt = datetime.datetime(2025, 4, 14, 23, 59, 59)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Clear existing orders
            cursor.execute("DELETE FROM order_details")
            cursor.execute("DELETE FROM orders")
            conn.commit()
            print("Cleared existing orders and order_details.")

            # Get current product IDs from products table that are HVAC products (start with 'HV')
            cursor.execute("SELECT product_id FROM products WHERE product_id LIKE 'HV%'")
            product_ids = [row[0] for row in cursor.fetchall()]

            if not product_ids:
                print("No HVAC products found. Please run add_initial_products() first.")
                return

            # Start order IDs at 1 (since tables are now empty)
            current_id = 0
            for _ in range(num_orders):
                current_id += 1
                order_id_str = str(current_id)
                first_name = random.choice(FIRST_NAMES)
                last_name = random.choice(LAST_NAMES)
                customer_name = f"{first_name} {last_name}"
                product_id = random.choice(product_ids)
                quantity = random.randint(1, 10)

                # Generate a random datetime within the defined range
                order_dt = random_datetime(start_dt, end_dt)
                # Format it to include both date and time
                order_date_str = order_dt.strftime('%Y-%m-%d %H:%M:%S')

                cursor.execute(
                    "INSERT INTO orders (order_id, customer_name, date, status) VALUES (?, ?, ?, 'Processing')",
                    (order_id_str, customer_name, order_date_str)
                )
                cursor.execute(
                    "INSERT INTO order_details (order_id, product_id, quantity) VALUES (?, ?, ?)",
                    (order_id_str, product_id, quantity)
                )
            conn.commit()
            print(f"Inserted {num_orders} new random orders successfully.")
    except Exception as e:
        logging.error(f"Error in clear_and_populate_orders: {e}")
        print(f"Error clearing and populating orders: {e}")



# ------------------------------
# CHATBOT FUNCTIONALITY
# ------------------------------
def chatbot_response(prompt):
    """
    Uses openai.ChatCompletion for chatbot responses.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return "Error connecting to chatbot service."

class ChatbotTab:
    def __init__(self, master):
        self.master = master
        self.setup_ui()

    def setup_ui(self):
        self.frame = ttk.Frame(self.master)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(self.frame, text="Ask a question:", font=("Helvetica", 14)).grid(row=0, column=0, sticky="w")
        self.user_input = ttk.Entry(self.frame, width=50)
        self.user_input.grid(row=1, column=0, pady=5)
        self.submit_button = ttk.Button(self.frame, text="Ask", command=self.ask_chatbot)
        self.submit_button.grid(row=1, column=1, padx=5)
        self.response_area = tk.Text(self.frame, wrap="word", height=10, width=60)
        self.response_area.grid(row=2, column=0, columnspan=2, pady=10)

    def ask_chatbot(self):
        prompt = self.user_input.get()
        if prompt:
            response = chatbot_response(prompt)
            self.response_area.delete("1.0", tk.END)
            self.response_area.insert(tk.END, f"Chatbot: {response}\n")
        else:
            self.response_area.delete("1.0", tk.END)
            self.response_area.insert(tk.END, "Please enter a question.\n")

# ------------------------------
# FRAUD DETECTION & FEATURE ENGINEERING
# ------------------------------

def get_product_price(product_id):
    """
    Given a product ID, retrieves the actual product price from the products table.
    Returns the price as a float or 0.0 if not found.
    """
    try:
        with sqlite3.connect('order_system.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
            result = cursor.fetchone()
        if result:
            return float(result[0])
        else:
            logging.warning(f"Price not found for product_id: {product_id}")
            return 0.0
    except Exception as e:
        logging.error(f"Error fetching price for product_id {product_id}: {e}")
        return 0.0

def feature_engineering(df):
    """
    Computes aggregated features required for fraud detection.
    
    Assumes the following columns exist in the DataFrame:
      - transaction_amount
      - customer_id
      - transaction_id
      - transaction_date
     
    Creates new columns:
      - total_customer_spent: Total money spent by the customer.
      - total_customer_orders: Count of orders by the customer.
      - avg_transaction_amount: Average amount per order.
      - order_frequency: Difference in days between the customer's latest and earliest order.
    """
    df['total_customer_spent'] = df.groupby('customer_id')['transaction_amount'].transform('sum')
    df['total_customer_orders'] = df.groupby('customer_id')['transaction_id'].transform('count')
    df['avg_transaction_amount'] = df['total_customer_spent'] / df['total_customer_orders']
    df['order_frequency'] = df.groupby('customer_id')['transaction_date'].transform(lambda x: (x.max() - x.min()).days)
    return df

def load_fraud_detection_model():
    """
    Loads and returns the pre-trained fraud detection model from FRAUD_MODEL_PATH.
    """
    try:
        model = joblib.load(FRAUD_MODEL_PATH)
        logging.info("Fraud detection model loaded.")
        return model
    except Exception as e:
        logging.error(f"Error loading fraud model: {e}")
        return None

def detect_fraud(order_details, model):
    """
    Receives order details, computes features using the actual product price, and uses the
    pre-trained model to predict fraud probability.
    
    Steps:
      - Create a DataFrame from order_details.
      - Retrieve the real product price via get_product_price() and compute:
            transaction_amount = (actual product price) * (quantity)
      - Set customer_id (from order_details['customer_name']), transaction_id (from order_details['order_id']),
        and transaction_date (current timestamp).
      - Call feature_engineering() to compute aggregated features.
      - Predict fraud probability using features:
            ['total_customer_spent', 'total_customer_orders', 'avg_transaction_amount', 'order_frequency', 'transaction_amount']
      - Flag the order as fraudulent if the predicted probability is >= 0.9.
    
    Returns:
      (is_fraud, fraud_prob): Tuple where is_fraud is True if fraud_prob >= 0.9, along with the fraud probability.
    """
    if not model:
        return False, 0.0
    try:
        order_df = pd.DataFrame([order_details])
        # Retrieve the real product price and compute transaction_amount
        price = get_product_price(order_details.get('product_id'))
        quantity = float(order_details.get('quantity', 1))
        order_df['transaction_amount'] = price * quantity
        
        # Set additional required fields
        order_df['customer_id'] = order_details['customer_name']
        order_df['transaction_id'] = order_details['order_id']
        order_df['transaction_date'] = pd.to_datetime('now')
        
        # Compute aggregated features
        order_df = feature_engineering(order_df)
        
        features = ['total_customer_spent', 'total_customer_orders', 'avg_transaction_amount', 'order_frequency', 'transaction_amount']
        fraud_prob = model.predict_proba(order_df[features])[:, 1]
        is_fraud = fraud_prob >= 0.9
        
        return bool(is_fraud[0]), fraud_prob[0]
    except Exception as e:
        logging.error(f"Fraud detection error: {e}")
        return False, 0.0


# ------------------------------
# FORECASTING UTILITIES (LSTM, ARIMA, Prophet)
# ------------------------------
def train_lstm_model(data, feature='quantity'):
    if data.empty:
        return None, None
    data = data.reset_index(drop=True)
    data[feature].fillna(0, inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature].values.reshape(-1, 1))
    x_train, y_train = [], []
    sequence_length = 30
    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)
    return model, scaler

def lstm_predict_next_30_days(data, model, scaler):
    if model is None or scaler is None or data.empty:
        return []
    last_30 = data.iloc[-30:].values.reshape(-1, 1)
    if len(last_30) < 30:
        return []
    scaled_last_30 = scaler.transform(last_30)
    x_input = np.reshape(scaled_last_30, (1, scaled_last_30.shape[0], 1))
    predictions = []
    for i in range(30):
        pred = model.predict(x_input, verbose=0)
        predictions.append(pred[0, 0])
        x_input = np.append(x_input[:,1:,:], [[pred]], axis=1)
    predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    return predicted_values.ravel().tolist()

def train_arima_model(data, feature='quantity'):
    if data.empty:
        return None
    data = data.set_index('date').asfreq('D').fillna(0)
    model = sm.tsa.ARIMA(data[feature], order=(5,1,0))
    model_fit = model.fit(disp=0)
    return model_fit

def arima_predict_next_30_days(model_fit):
    if not model_fit:
        return []
    forecast = model_fit.forecast(steps=30)[0]
    return forecast.tolist()

def train_prophet_model(data):
    if data.empty:
        return None
    prophet_data = data.rename(columns={'date': 'ds', 'quantity': 'y'})[['ds','y']]
    model = Prophet()
    model.fit(prophet_data)
    return model

def prophet_predict_next_30_days(model):
    if not model:
        return []
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    result = forecast[['ds','yhat']].tail(30)
    return result.values.tolist()

# ------------------------------
# ORDER PROCESSOR CLASS
# ------------------------------
class OrderProcessor:
    def __init__(self, ui):
        self.ui = ui
        self.db_path = 'order_system.db'
        self.fraud_model = load_fraud_detection_model()
        self.blockchain = Blockchain()

    def connect_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA foreign_keys = ON')
        return conn

    def generate_order_id(self):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(CAST(order_id AS INTEGER)) FROM orders")
            max_id = cursor.fetchone()[0]
        return int(max_id) + 1 if max_id else 1

    def reduce_stock(self, cursor, product_id, quantity):
        cursor.execute("SELECT stock FROM products WHERE product_id = ?", (product_id,))
        result = cursor.fetchone()
        if result:
            current_stock = result[0]
            if current_stock >= int(quantity):
                new_stock = current_stock - int(quantity)
                cursor.execute("UPDATE products SET stock = ? WHERE product_id = ?", (new_stock, product_id))
                if new_stock < 20:
                    send_email("Low Stock Alert", f"Product {product_id} is low on stock: {new_stock}", EMAIL_ADDRESS)
            else:
                logging.warning(f"Insufficient stock for {product_id}: Requested {quantity}, Available {current_stock}")
        else:
            logging.warning(f"Product {product_id} not found.")

    def add_order(self, order):
        if not self.validate_order(order):
            self.ui.message_label.config(text="Invalid order details")
            return
        try:
            with self.connect_db() as conn:
                cursor = conn.cursor()
                new_order_id = self.generate_order_id()
                order_id_str = str(new_order_id)
                cursor.execute(
                    "INSERT INTO orders (order_id, customer_name, date, status) VALUES (?, ?, datetime('now'), 'Processing')",
                    (order_id_str, order['customer_name'])
                )
                cursor.execute(
                    "INSERT INTO order_details (order_id, product_id, quantity) VALUES (?, ?, ?)",
                    (order_id_str, order['product_id'], order['quantity'])
                )
                self.reduce_stock(cursor, order['product_id'], int(order['quantity']))
                conn.commit()

                # Fraud detection
                order['order_id'] = order_id_str
                is_fraud, fraud_prob = detect_fraud(order, self.fraud_model)
                flag = 1 if is_fraud else 0
                cursor.execute(
                    "UPDATE orders SET fraud_flag = ?, fraud_prob = ? WHERE order_id = ?",
                    (flag, fraud_prob, order_id_str)
                )
                conn.commit()
                if is_fraud:
                    self.ui.message_label.config(text=f"Order flagged as potentially fraudulent (Prob: {fraud_prob:.2f})")
                else:
                    self.ui.message_label.config(text=f"Order {order_id_str} added successfully.")
                send_email(
                    "Order Confirmation",
                    f"Your order with Order ID {order_id_str} has been placed successfully.",
                    EMAIL_ADDRESS
                )
                # Add to blockchain if total > 1000
                total_price = self.calculate_order_total(order_id_str, cursor)
                if total_price > 1000:
                    block_data = {
                        'order_id': order_id_str,
                        'customer_name': order['customer_name'],
                        'total_price': total_price,
                        'product_id': order['product_id']
                    }
                    new_block = Block(len(self.blockchain.chain), time.time(), str(block_data))
                    self.blockchain.add_block(new_block)
                    logging.info(f"Order {order_id_str} added to blockchain.")
                    self.ui.update_blockchain_ledger()
        except sqlite3.IntegrityError as e:
            logging.error(f"Integrity error: {e}")
            self.ui.message_label.config(text=f"Error: {e}")
        except Exception as e:
            logging.error(f"Order addition error: {e}")
            self.ui.message_label.config(text=f"An error occurred: {str(e)}")

    def calculate_order_total(self, order_id, cursor):
        cursor.execute('''
            SELECT SUM(od.quantity * p.price)
            FROM order_details od
            JOIN products p ON od.product_id = p.product_id
            WHERE od.order_id = ?
        ''', (order_id,))
        total = cursor.fetchone()[0]
        return total if total else 0

    def fetch_flagged_orders(self):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT order_id, customer_name, date, fraud_prob FROM orders WHERE fraud_flag = 1")
            orders = cursor.fetchall()
        return orders

    def validate_order(self, order):
        if not order.get('customer_name') or not order.get('product_id') or not order.get('quantity'):
            logging.warning("Order missing required fields.")
            return False
        if not order['quantity'].isdigit() or int(order['quantity']) <= 0:
            logging.warning("Invalid quantity specified.")
            return False
        return True

    def update_order_status(self, order_id, new_status):
        try:
            with self.connect_db() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE orders SET order_status = ? WHERE order_id = ?", (new_status, order_id))
                conn.commit()
            logging.info(f"Order {order_id} updated to {new_status}")
        except Exception as e:
            logging.error(f"Status update error: {e}")

    def fetch_orders_with_status(self):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT order_id, customer_name, date, order_status FROM orders")
            orders = cursor.fetchall()
        return orders

    def fetch_inventory(self):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, stock FROM products")
            inventory = cursor.fetchall()
        return inventory

    def plot_inventory_levels(self):
        try:
            with self.connect_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name, stock FROM products")
                data = cursor.fetchall()
            if data:
                products, stocks = zip(*data)
                # Increase figure size for better spacing
                plt.figure(figsize=(12, 6))
                plt.bar(products, stocks)
                plt.xlabel('Product Name')
                plt.ylabel('Stock Level')
                plt.title('Inventory Levels')
                # Rotate x-axis labels to prevent overlapping
                plt.xticks(rotation=45, ha='right')
                # Adjust layout automatically to fit labels
                plt.tight_layout()
                plt.show()
            else:
                logging.info("No inventory data available.")
        except Exception as e:
            logging.error(f"Inventory plot error: {e}")


    # NEW: fetch_all_sales_df for Analytics – only include HVAC product orders
    def fetch_all_sales_df(self):
        try:
            with self.connect_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT o.order_id AS "Order ID",
                           o.customer_name AS "Customer Name",
                           o.date AS "Date",
                           p.name AS "Product Name",
                           od.quantity AS "Quantity",
                           p.price AS "Price",
                           (od.quantity * p.price) AS "Total Price"
                    FROM orders o
                    JOIN order_details od ON o.order_id = od.order_id
                    JOIN products p ON od.product_id = p.product_id
                    WHERE p.product_id LIKE 'HV%'
                ''')
                data = cursor.fetchall()
            df = pd.DataFrame(data, columns=[
                "Order ID", "Customer Name", "Date",
                "Product Name", "Quantity", "Price", "Total Price"
            ])
            return df
        except Exception as e:
            logging.error(f"fetch_all_sales_df error: {e}")
            return pd.DataFrame()

    # For the Inventory Tab: open a window with a scrollable report table.
    def generate_sales_report_table(self, parent_window):
        df = self.fetch_all_sales_df()
        if df.empty:
            messagebox.showerror("Error", "No sales data available.")
            return

        report_window = tk.Toplevel(parent_window)
        report_window.title("Sales Report")

        container = ttk.Frame(report_window)
        container.pack(fill=tk.BOTH, expand=True)

        columns = list(df.columns)
        tree = ttk.Treeview(container, columns=columns, show='headings', height=20)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=tk.CENTER)
        for _, row in df.iterrows():
            tree.insert("", tk.END, values=list(row))
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def search_order(self, order_id):
        try:
            with self.connect_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT o.order_id, o.customer_name, o.date, o.order_status,
                           p.name, od.quantity, (od.quantity * p.price) AS total_price
                    FROM orders o
                    JOIN order_details od ON o.order_id = od.order_id
                    JOIN products p ON od.product_id = p.product_id
                    WHERE o.order_id = ?
                ''', (order_id,))
                result = cursor.fetchone()
            return result
        except Exception as e:
            logging.error(f"Search order error: {e}")
            return None

# ------------------------------
# REAL-TIME DASHBOARD
# ------------------------------
class RealTimeDashboard:
    def __init__(self, master):
        self.master = master
        self.setup_ui()
        self.update_dashboard()

    def setup_ui(self):
        self.frame = ttk.Frame(self.master)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.label_sales = ttk.Label(self.frame, text="Total Sales: $0.00", font=("Helvetica", 16))
        self.label_sales.grid(row=0, column=0, sticky="w", pady=5)
        self.label_orders = ttk.Label(self.frame, text="Total Orders: 0", font=("Helvetica", 16))
        self.label_orders.grid(row=1, column=0, sticky="w", pady=5)
        self.label_low_stock = ttk.Label(self.frame, text="Low Stock Alerts: None", font=("Helvetica", 16))
        self.label_low_stock.grid(row=2, column=0, sticky="w", pady=5)

    def update_dashboard(self):
        try:
            with sqlite3.connect('order_system.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT SUM(od.quantity * p.price)
                    FROM order_details od
                    JOIN products p ON od.product_id = p.product_id
                ''')
                total_sales = cursor.fetchone()[0] or 0
                self.label_sales.config(text=f"Total Sales: ${total_sales:.2f}")
                cursor.execute("SELECT COUNT(*) FROM orders")
                total_orders = cursor.fetchone()[0]
                self.label_orders.config(text=f"Total Orders: {total_orders}")
                cursor.execute("SELECT name FROM products WHERE stock < 20")
                low_stock = cursor.fetchall()
                if low_stock:
                    self.label_low_stock.config(text="Low Stock Alerts: " + ", ".join([item[0] for item in low_stock]))
                else:
                    self.label_low_stock.config(text="Low Stock Alerts: None")
        except Exception as e:
            logging.error(f"Dashboard update error: {e}")
        self.master.after(DASHBOARD_UPDATE_INTERVAL * 1000, self.update_dashboard)

# ------------------------------
# LOGIN UI
# ------------------------------
class LoginUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Login")
        self.master.geometry("300x200")
        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.master, padding="10")
        frame.pack(expand=True, fill=tk.BOTH)
        ttk.Label(frame, text="Username:").grid(row=0, column=0, sticky="w", pady=5)
        self.username_entry = ttk.Entry(frame)
        self.username_entry.grid(row=0, column=1, pady=5)
        ttk.Label(frame, text="Password:").grid(row=1, column=0, sticky="w", pady=5)
        self.password_entry = ttk.Entry(frame, show="*")
        self.password_entry.grid(row=1, column=1, pady=5)
        self.message_label = ttk.Label(frame, text="", foreground="red")
        self.message_label.grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(frame, text="Login", command=self.login).grid(row=3, column=0, columnspan=2, pady=10)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if self.authenticate(username, password):
            self.master.destroy()
            root = tk.Tk()
            OrderProcessingUI(root, username, self.user_role)
            root.mainloop()
        else:
            self.message_label.config(text="Invalid credentials")

    def authenticate(self, username, password):
        try:
            with sqlite3.connect('order_system.db') as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT password, role FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()
            if result and check_password_hash(result[0], password):
                self.user_role = result[1]
                return True
            return False
        except Exception as e:
            logging.error(f"Authentication error: {e}")
            return False

# ------------------------------
# MAIN ORDER PROCESSING UI
# ------------------------------
class OrderProcessingUI:
    def __init__(self, master, username, user_role):
        self.master = master
        self.username = username
        self.user_role = user_role
        self.master.title(f"Order Processing System - Logged in as {username}")
        self.master.geometry("1000x700")
        self.order_processor = OrderProcessor(self)
        self.create_menu()
        self.setup_notebook()

    def create_menu(self):
        menu_bar = tk.Menu(self.master)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.master.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Order Processing System v2.0"))
        menu_bar.add_cascade(label="Help", menu=help_menu)
        self.master.config(menu=menu_bar)

    def setup_notebook(self):
        notebook = ttk.Notebook(self.master)
        self.analytics_tab = ttk.Frame(notebook)
        self.dashboard_tab = ttk.Frame(notebook)
        self.blockchain_tab = ttk.Frame(notebook)
        self.fraud_tab = ttk.Frame(notebook)
        self.chatbot_tab = ttk.Frame(notebook)
        self.order_tab = ttk.Frame(notebook)
        self.inventory_tab = ttk.Frame(notebook)
        self.email_tab = ttk.Frame(notebook)
        self.reports_tab = ttk.Frame(notebook)
        self.admin_tab = None

        if self.user_role == 'admin':
            notebook.add(self.analytics_tab, text='Sales Analytics')
            notebook.add(self.dashboard_tab, text='Dashboard')
            notebook.add(self.blockchain_tab, text='Blockchain Ledger')
            notebook.add(self.fraud_tab, text='Fraud Detection')
            notebook.add(self.chatbot_tab, text='Chatbot')
            notebook.add(self.order_tab, text='Order Entry')
            notebook.add(self.inventory_tab, text='Inventory')
            notebook.add(self.email_tab, text='Email Orders')
            notebook.add(self.reports_tab, text='Reports')
            self.admin_tab = ttk.Frame(notebook)
            notebook.add(self.admin_tab, text='Admin')
        elif self.user_role == 'manager':
            notebook.add(self.analytics_tab, text='Sales Analytics')
            notebook.add(self.dashboard_tab, text='Dashboard')
            notebook.add(self.fraud_tab, text='Fraud Detection')
            notebook.add(self.order_tab, text='Order Entry')
            notebook.add(self.inventory_tab, text='Inventory')
            notebook.add(self.email_tab, text='Email Orders')
            notebook.add(self.reports_tab, text='Reports')
        elif self.user_role == 'employee':
            notebook.add(self.order_tab, text='Order Entry')
            notebook.add(self.inventory_tab, text='Inventory')
            notebook.add(self.email_tab, text='Email Orders')

        notebook.pack(expand=1, fill="both")
        self.setup_tabs()

    def setup_tabs(self):
        if self.user_role in ['admin', 'manager']:
            self.setup_analytics_tab()
            self.setup_dashboard_tab()
            self.setup_fraud_tab()
            self.setup_reports_tab()
        if self.user_role == 'admin':
            self.setup_blockchain_tab()
            ChatbotTab(self.chatbot_tab)
        self.setup_order_tab()
        self.setup_inventory_tab()
        self.setup_email_tab()
        if self.user_role == 'admin':
            self.setup_admin_tab()

    # ----------------------
    # Analytics Tab (Plots)
    # ----------------------
    def setup_analytics_tab(self):
        frame = ttk.Frame(self.analytics_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Button(frame, text="View Sales Summary", command=self.view_sales_summary).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Sales by Product", command=self.plot_sales_by_product).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Monthly Sales", command=self.plot_monthly_sales).grid(row=0, column=2, padx=5, pady=5)

        # Order search
        search_frame = ttk.LabelFrame(frame, text="Search Order", padding="10")
        search_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky="ew")
        ttk.Label(search_frame, text="Order ID:").grid(row=0, column=0, sticky="w")
        self.search_entry = ttk.Entry(search_frame, width=20)
        self.search_entry.grid(row=0, column=1, padx=5)
        ttk.Button(search_frame, text="Search", command=self.search_order).grid(row=0, column=2, padx=5)
        self.search_result = ttk.Label(search_frame, text="")
        self.search_result.grid(row=1, column=0, columnspan=3, pady=5)

    def search_order(self):
        order_id = self.search_entry.get().strip()
        if order_id:
            result = self.order_processor.search_order(order_id)
            if result:
                res_text = (f"OrderID: {result[0]}, Customer: {result[1]}, Date: {result[2]}, "
                            f"Status: {result[3]}, Product: {result[4]}, Qty: {result[5]}, "
                            f"Total: ${result[6]:.2f}")
                self.search_result.config(text=res_text)
            else:
                self.search_result.config(text="Order not found.")
        else:
            self.search_result.config(text="Please enter an Order ID.")

    def view_sales_summary(self):
        df = self.order_processor.fetch_all_sales_df()
        if not df.empty:
            summary = df.groupby('Product Name')['Total Price'].sum()
            plt.figure()
            summary.plot(kind='bar', title='Sales Summary by Product')
            plt.xlabel('Product Name')
            plt.ylabel('Total Sales')
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showinfo("No Data", "No sales data available for summary.")

    def plot_sales_by_product(self):
        df = self.order_processor.fetch_all_sales_df()
        if not df.empty:
            plt.figure()
            df.groupby('Product Name')['Quantity'].sum().plot(kind='bar', title='Total Sales by Product')
            plt.xlabel('Product Name')
            plt.ylabel('Quantity Sold')
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showinfo("No Data", "No sales data available for plotting.")

    def plot_monthly_sales(self):
        df = self.order_processor.fetch_all_sales_df()
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            monthly = df.resample('M', on='Date').sum()
            plt.figure()
            monthly['Total Price'].plot(kind='line', marker='o', title='Monthly Sales')
            plt.xlabel('Month')
            plt.ylabel('Total Sales')
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showinfo("No Data", "No monthly sales data available.")

    # ----------------------
    # Dashboard Tab
    # ----------------------
    def setup_dashboard_tab(self):
        RealTimeDashboard(self.dashboard_tab)

    # ----------------------
    # Blockchain Tab
    # ----------------------
    def setup_blockchain_tab(self):
        frame = ttk.Frame(self.blockchain_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Blockchain Ledger", font=("Helvetica", 14)).pack(pady=10)
        columns = ("Block Index", "Timestamp", "Order Data", "Previous Hash", "Hash")
        self.blockchain_tree = ttk.Treeview(frame, columns=columns, show='headings')
        for col in columns:
            self.blockchain_tree.heading(col, text=col)
            self.blockchain_tree.column(col, width=150, anchor=tk.CENTER)
        self.blockchain_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(frame, text="Refresh Ledger", command=self.update_blockchain_ledger).pack(pady=5)
        self.update_blockchain_ledger()

    def update_blockchain_ledger(self):
        for row in self.blockchain_tree.get_children():
            self.blockchain_tree.delete(row)
        for block in self.order_processor.blockchain.chain:
            self.blockchain_tree.insert("", tk.END, values=(
                block.index, block.timestamp, block.data, block.previous_hash, block.hash
            ))

    # ----------------------
    # Fraud Detection Tab
    # ----------------------
    def setup_fraud_tab(self):
        frame = ttk.Frame(self.fraud_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Flagged Fraudulent Orders", font=("Helvetica", 14)).pack(pady=10)
        columns = ("Order ID", "Customer Name", "Date", "Fraud Probability")
        self.fraud_tree = ttk.Treeview(frame, columns=columns, show='headings')
        for col in columns:
            self.fraud_tree.heading(col, text=col)
            self.fraud_tree.column(col, width=150, anchor=tk.CENTER)
        self.fraud_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(frame, text="Refresh Fraud Data", command=self.load_fraud_data).pack(pady=5)
        self.load_fraud_data()

    def load_fraud_data(self):
        for row in self.fraud_tree.get_children():
            self.fraud_tree.delete(row)
        orders = self.order_processor.fetch_flagged_orders()
        for order in orders:
            self.fraud_tree.insert("", tk.END, values=order)

    # ----------------------
    # Reports Tab (Export CSV)
    # ----------------------
    def setup_reports_tab(self):
        frame = ttk.Frame(self.reports_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Button(frame, text="Export Sales Report to CSV", command=self.export_sales_report).pack(pady=10)

    def export_sales_report(self):
        df = self.order_processor.fetch_all_sales_df()
        if not df.empty:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if file_path:
                df.to_csv(file_path, index=False)
                logging.info(f"Sales report exported to {file_path}")
                messagebox.showinfo("Export Successful", f"Report saved to {file_path}")
        else:
            messagebox.showerror("Error", "No sales data available.")

    # ----------------------
    # Order Entry Tab
    # ----------------------
    def setup_order_tab(self):
        frame = ttk.Frame(self.order_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Customer Name:").grid(row=0, column=0, sticky="w", pady=5)
        self.customer_name_entry = ttk.Entry(frame)
        self.customer_name_entry.grid(row=0, column=1, pady=5)
        ttk.Label(frame, text="Product ID:").grid(row=1, column=0, sticky="w", pady=5)
        self.product_id_entry = ttk.Entry(frame)
        self.product_id_entry.grid(row=1, column=1, pady=5)
        ttk.Label(frame, text="Quantity:").grid(row=2, column=0, sticky="w", pady=5)
        self.quantity_entry = ttk.Entry(frame)
        self.quantity_entry.grid(row=2, column=1, pady=5)
        ttk.Button(frame, text="Submit Order", command=self.submit_order).grid(row=3, column=0, pady=10)
        ttk.Button(frame, text="Cancel Order", command=self.cancel_order).grid(row=3, column=1, pady=10)
        # New widgets for updating order status
        ttk.Label(frame, text="Order ID:").grid(row=4, column=0, sticky="w", pady=5)
        self.order_id_entry = ttk.Entry(frame)
        self.order_id_entry.grid(row=4, column=1, pady=5)
        ttk.Label(frame, text="New Status:").grid(row=5, column=0, sticky="w", pady=5)
        self.order_status_entry = ttk.Entry(frame)
        self.order_status_entry.grid(row=5, column=1, pady=5)
        ttk.Button(frame, text="Update Status", command=self.update_status).grid(row=6, column=0, columnspan=2, pady=5)
        self.message_label = ttk.Label(frame, text="", foreground="blue")
        self.message_label.grid(row=7, column=0, columnspan=2, pady=5)

    def submit_order(self):
        order_details = {
            "customer_name": self.customer_name_entry.get().strip(),
            "product_id": self.product_id_entry.get().strip(),
            "quantity": self.quantity_entry.get().strip()
        }
        self.order_processor.add_order(order_details)

    def cancel_order(self):
        self.customer_name_entry.delete(0, tk.END)
        self.product_id_entry.delete(0, tk.END)
        self.quantity_entry.delete(0, tk.END)
        self.order_status_entry.delete(0, tk.END)
        self.order_id_entry.delete(0, tk.END)
        self.message_label.config(text="Order cancelled and fields cleared.")

    def update_status(self):
        order_id = self.order_id_entry.get().strip()  # Now uses the new widget
        new_status = self.order_status_entry.get().strip()
        if order_id and new_status:
            self.order_processor.update_order_status(order_id, new_status)
            self.message_label.config(text=f"Order {order_id} updated to {new_status}")
        else:
            self.message_label.config(text="Provide Order ID and new status.")

    # ----------------------
    # Inventory Tab
    # ----------------------
    def setup_inventory_tab(self):
        frame = ttk.Frame(self.inventory_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Button(frame, text="Plot Inventory Levels", command=self.order_processor.plot_inventory_levels).pack(pady=10)
        ttk.Button(frame, text="Generate Sales Report", command=self.open_sales_report_window).pack(pady=10)

    def open_sales_report_window(self):
        self.order_processor.generate_sales_report_table(self.master)

    # ----------------------
    # Email Orders Tab
    # ----------------------
    def setup_email_tab(self):
        frame = ttk.Frame(self.email_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Fetch Incoming Orders from Email", font=("Helvetica", 14)).pack(pady=10)
        ttk.Button(frame, text="Check Email", command=self.fetch_orders_from_inbox).pack(pady=10)
        self.email_tab_message = ttk.Label(frame, text="")
        self.email_tab_message.pack(pady=10)

    def fetch_orders_from_inbox(self):
        new_orders = fetch_incoming_orders_from_email()
        count = 0
        for order in new_orders:
            self.order_processor.add_order(order)
            count += 1
        msg = f"{count} new order(s) processed from inbox." if count else "No new orders found."
        self.email_tab_message.config(text=msg)

    # ----------------------
    # Admin Tab
    # ----------------------
    def setup_admin_tab(self):
        frame = ttk.Frame(self.admin_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Manage Products Section
        prod_frame = ttk.LabelFrame(frame, text="Manage Products", padding="10")
        prod_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ttk.Label(prod_frame, text="Product ID:").grid(row=0, column=0, sticky="w", pady=2)
        self.new_product_id_entry = ttk.Entry(prod_frame)
        self.new_product_id_entry.grid(row=0, column=1, pady=2)

        ttk.Label(prod_frame, text="Product Name:").grid(row=1, column=0, sticky="w", pady=2)
        self.new_product_name_entry = ttk.Entry(prod_frame)
        self.new_product_name_entry.grid(row=1, column=1, pady=2)

        ttk.Label(prod_frame, text="Price:").grid(row=2, column=0, sticky="w", pady=2)
        self.new_product_price_entry = ttk.Entry(prod_frame)
        self.new_product_price_entry.grid(row=2, column=1, pady=2)

        ttk.Label(prod_frame, text="Stock:").grid(row=3, column=0, sticky="w", pady=2)
        self.new_product_stock_entry = ttk.Entry(prod_frame)
        self.new_product_stock_entry.grid(row=3, column=1, pady=2)

        btn_frame = ttk.Frame(prod_frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(btn_frame, text="Add Product", command=self.add_product).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Update Product", command=self.update_product).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Product", command=self.delete_product).pack(side=tk.LEFT, padx=5)

        self.admin_message_label = ttk.Label(frame, text="", foreground="green")
        self.admin_message_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        # Manage Registrations Section
        reg_frame = ttk.LabelFrame(frame, text="Manage Registrations", padding="10")
        reg_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.registration_listbox = tk.Listbox(reg_frame, height=6)
        self.registration_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        reg_btn_frame = ttk.Frame(reg_frame)
        reg_btn_frame.pack(pady=5)
        ttk.Button(reg_btn_frame, text="Approve", command=self.approve_registration).pack(side=tk.LEFT, padx=5)
        ttk.Button(reg_btn_frame, text="Reject", command=self.reject_registration).pack(side=tk.LEFT, padx=5)

        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)
        self.load_registrations()

    def add_product(self):
        product_id = self.new_product_id_entry.get().strip()
        product_name = self.new_product_name_entry.get().strip()
        price = self.new_product_price_entry.get().strip()
        stock = self.new_product_stock_entry.get().strip()
        if not (product_id and product_name and price and stock):
            self.admin_message_label.config(text="All product fields are required.")
            return
        try:
            price = float(price)
            stock = int(stock)
            with sqlite3.connect('order_system.db') as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO products (product_id, name, price, stock) VALUES (?, ?, ?, ?)",
                    (product_id, product_name, price, stock)
                )
                conn.commit()
            self.admin_message_label.config(text="Product added successfully.")
        except sqlite3.IntegrityError:
            self.admin_message_label.config(text="Product ID already exists.")
        except Exception as e:
            self.admin_message_label.config(text=f"Error: {str(e)}")

    def update_product(self):
        product_id = self.new_product_id_entry.get().strip()
        product_name = self.new_product_name_entry.get().strip()
        price = self.new_product_price_entry.get().strip()
        stock = self.new_product_stock_entry.get().strip()
        if not (product_id and product_name and price and stock):
            self.admin_message_label.config(text="All product fields are required.")
            return
        try:
            price = float(price)
            stock = int(stock)
            with sqlite3.connect('order_system.db') as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE products SET name = ?, price = ?, stock = ? WHERE product_id = ?",
                    (product_name, price, stock, product_id)
                )
                conn.commit()
            self.admin_message_label.config(text="Product updated successfully.")
        except Exception as e:
            self.admin_message_label.config(text=f"Error: {str(e)}")

    def delete_product(self):
        product_id = self.new_product_id_entry.get().strip()
        if not product_id:
            self.admin_message_label.config(text="Product ID is required to delete.")
            return
        try:
            with sqlite3.connect('order_system.db') as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
                conn.commit()
            self.admin_message_label.config(text="Product deleted successfully.")
        except Exception as e:
            self.admin_message_label.config(text=f"Error: {str(e)}")

    def load_registrations(self):
        try:
            with sqlite3.connect('order_system.db') as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT registration_id, username FROM user_registrations WHERE status = 'Pending'")
                regs = cursor.fetchall()
            self.registration_listbox.delete(0, tk.END)
            for reg in regs:
                self.registration_listbox.insert(tk.END, f"{reg[0]}: {reg[1]}")
        except Exception as e:
            logging.error(f"Registration load error: {e}")

    def approve_registration(self):
        selection = self.registration_listbox.curselection()
        if selection:
            reg_id = self.registration_listbox.get(selection[0]).split(":")[0]
            try:
                with sqlite3.connect('order_system.db') as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT username, password, email, role FROM user_registrations WHERE registration_id = ?", (reg_id,))
                    user_data = cursor.fetchone()
                    if user_data:
                        cursor.execute(
                            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                            (user_data[0], user_data[1], user_data[3])
                        )
                        cursor.execute("UPDATE user_registrations SET status = 'Approved' WHERE registration_id = ?", (reg_id,))
                        conn.commit()
                        self.admin_message_label.config(text="User approved and added.")
                self.load_registrations()
            except Exception as e:
                self.admin_message_label.config(text=f"Error: {str(e)}")

    def reject_registration(self):
        selection = self.registration_listbox.curselection()
        if selection:
            reg_id = self.registration_listbox.get(selection[0]).split(":")[0]
            try:
                with sqlite3.connect('order_system.db') as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE user_registrations SET status = 'Rejected' WHERE registration_id = ?", (reg_id,))
                    conn.commit()
                self.admin_message_label.config(text="User registration rejected.")
                self.load_registrations()
            except Exception as e:
                self.admin_message_label.config(text=f"Error: {str(e)}")

# ------------------------------
# MAIN BLOCK
# ------------------------------
if __name__ == "__main__":
    print("Database absolute path:", os.path.abspath('order_system.db'))
    init_db()
    alter_orders_table()
    add_initial_products()
    add_initial_users()
    # Clear all orders and repopulate with 5000 new orders (which will use HVAC products only)
    # clear_and_populate_orders(num_orders=5000)
    with sqlite3.connect('order_system.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM orders")
        count_orders = cursor.fetchone()[0]
        print("Total orders in the database:", count_orders)
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("clam")
    login_app = LoginUI(root)
    root.mainloop()
