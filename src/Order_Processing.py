import sqlite3
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import pandas as pd
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.linear_model import LinearRegression
import numpy as np
import threading
import seaborn as sns  
import time
import openai  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib  # Ensure joblib is imported for model saving/loading
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm  
from prophet import Prophet  
import hashlib
import time


# Configure logging
logging.basicConfig(filename='order_system.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# OpenAI API Key setup for Chatbot
openai.api_key = 'API Key'  # Replace with your OpenAI API key

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'bryansamjames@gmail.com'
EMAIL_PASSWORD = 'abcd efgh ijkl mnop'  # Replace this with your actual app password

# Real-time Dashboard Update Interval (in seconds)
DASHBOARD_UPDATE_INTERVAL = 10

# Fraud Detection Model Path
FRAUD_MODEL_PATH = 'fraud_detection_model.pkl'  # Add this for fraud detection


class Block:
    def __init__(self, index, timestamp, data, previous_hash=''):
        self.index = index
        self.timestamp = timestamp
        self.data = data  # Order details
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
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True
    
    
# Function to send emails
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

# Initialize Database
def init_db():
    conn = sqlite3.connect('order_system.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    ''')
    
    # Create user_registrations table
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
    
    # Create products table
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            name TEXT,
            price REAL,
            stock INTEGER
        )
    ''')
    
    # Create orders table with fraud_flag and fraud_prob columns
    c.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            customer_name TEXT,
            date TEXT,
            status TEXT,
            order_status TEXT DEFAULT 'Processing',
            fraud_flag INTEGER DEFAULT 0,  -- Add fraud_flag (0 for non-fraud, 1 for fraud)
            fraud_prob REAL DEFAULT 0.0    -- Add fraud_prob (Probability of fraud)
        )
    ''')

    # Create order_details table
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
    conn.close()
    logging.info("Database initialized with fraud detection columns.")


def alter_orders_table():
    conn = sqlite3.connect('order_system.db')
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE orders ADD COLUMN fraud_flag INTEGER DEFAULT 0")
        c.execute("ALTER TABLE orders ADD COLUMN fraud_prob REAL DEFAULT 0.0")
        logging.info("Added fraud_flag and fraud_prob columns to the orders table.")
    except sqlite3.OperationalError as e:
        logging.warning(f"Columns already exist or failed to add: {e}")
    conn.commit()
    conn.close()


# Populate initial product data
def add_initial_products():
    products = [
        ('P001', 'Widget A', 19.99, 100),
        ('P002', 'Widget B', 25.99, 200),
        ('P003', 'Widget C', 9.99, 150)
    ]
    try:
        with sqlite3.connect('order_system.db') as conn:
            c = conn.cursor()
            c.executemany('INSERT OR IGNORE INTO products (product_id, name, price, stock) VALUES (?, ?, ?, ?)', products)
            conn.commit()
        logging.info("Initial products added")
    except sqlite3.OperationalError as e:
        logging.error(f"Error adding initial products: {e}")

# Populate initial user data
def add_initial_users():
    users = [
        ('admin', generate_password_hash('password'), 'admin'),
        ('manager', generate_password_hash('password'), 'manager'),
        ('employee', generate_password_hash('password'), 'employee')
    ]
    try:
        with sqlite3.connect('order_system.db') as conn:
            c = conn.cursor()
            c.executemany('INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)', users)
            conn.commit()
        logging.info("Initial users added")
    except sqlite3.OperationalError as e:
        logging.error(f"Error adding initial users: {e}")

# Chatbot integration with OpenAI
def chatbot_response(prompt):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",  # Use the ChatGPT model
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error with OpenAI API: {e}")
        return "There was an issue connecting to the chatbot service. Please try again."

# Chatbot Tab UI
class ChatbotTab:
    def __init__(self, master):
        self.master = master
        self.setup_ui()

    def setup_ui(self):
        # UI elements for chatbot interaction
        self.label = ttk.Label(self.master, text="Ask a question:", font=("Helvetica", 14))
        self.label.pack(pady=10)

        self.user_input = tk.Entry(self.master, width=50)
        self.user_input.pack(pady=10)

        self.submit_button = ttk.Button(self.master, text="Ask", command=self.ask_chatbot)
        self.submit_button.pack(pady=10)

        self.response_area = tk.Text(self.master, wrap="word", height=10, width=60)
        self.response_area.pack(pady=10)

    def ask_chatbot(self):
        user_query = self.user_input.get()
        if user_query:
            response = chatbot_response(user_query)
            self.response_area.delete("1.0", tk.END)
            self.response_area.insert(tk.END, f"Chatbot: {response}\n")
        else:
            self.response_area.delete("1.0", tk.END)
            self.response_area.insert(tk.END, "Please enter a question.\n")

# Fraud Detection Functions
def load_fraud_detection_model():
    """
    Load the pre-trained fraud detection model.
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
    Perform feature engineering on the transaction data.
    """
    df['total_customer_spent'] = df.groupby('customer_id')['transaction_amount'].transform('sum')
    df['total_customer_orders'] = df.groupby('customer_id')['transaction_id'].transform('count')
    df['avg_transaction_amount'] = df['total_customer_spent'] / df['total_customer_orders']
    df['order_frequency'] = df.groupby('customer_id')['transaction_date'].transform(lambda x: (x.max() - x.min()).days)
    return df

def detect_fraud(order_details, model):
    """
    Detect if an order is potentially fraudulent using the trained model.
    """
    try:
        # Convert order details into a DataFrame for prediction
        order_df = pd.DataFrame([order_details])

        # Perform feature engineering
        order_df = feature_engineering(order_df)

        # Select features used in training the model
        features = ['total_customer_spent', 'total_customer_orders', 'avg_transaction_amount', 'order_frequency', 'transaction_amount']

        # Predict fraud
        fraud_prob = model.predict_proba(order_df[features])[:, 1]  # Probability of fraud
        is_fraud = fraud_prob >= 0.9  # Use 0.9 threshold for flagging fraud
        return is_fraud[0], fraud_prob[0]
    except Exception as e:
        logging.error(f"Error detecting fraud: {e}")
        return False, 0.0

# Login UI Class
class LoginUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Login")
        self.master.geometry("300x200")

        ttk.Label(master, text="Username:").pack()
        self.username_entry = ttk.Entry(master)
        self.username_entry.pack()

        ttk.Label(master, text="Password:").pack()
        self.password_entry = ttk.Entry(master, show="*")
        self.password_entry.pack()

        ttk.Button(master, text="Login", command=self.login).pack()
        self.message_label = ttk.Label(master, text="")
        self.message_label.pack()

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if self.authenticate(username, password):
            self.master.destroy()
            root = tk.Tk()
            app = OrderProcessingUI(root, username, self.user_role)
            root.mainloop()
        else:
            self.message_label.config(text="Invalid credentials")

    def authenticate(self, username, password):
        conn = sqlite3.connect('order_system.db')
        cursor = conn.cursor()
        cursor.execute("SELECT password, role FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        if result and check_password_hash(result[0], password):
            self.user_role = result[1]
            return True
        else:
            return False

# Order Processor Class
class OrderProcessor:
    def __init__(self, ui):
        self.ui = ui
        self.db_path = 'order_system.db'
        self.fraud_model = load_fraud_detection_model()  # Load fraud detection model
        self.blockchain = Blockchain()

    def connect_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA foreign_keys = ON')
        return conn
    
    def generate_order_id(self):
        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(CAST(order_id AS INTEGER)) FROM orders")
        max_id = cursor.fetchone()[0]
        conn.close()
        return int(max_id) + 1 if max_id else 1
    
    def reduce_stock(self, cursor, product_id, quantity):
        cursor.execute("SELECT stock FROM products WHERE product_id = ?", (product_id,))
        result = cursor.fetchone()
        if result:
            current_stock = result[0]
            logging.info(f"Current stock for product {product_id}: {current_stock}")
            if current_stock >= int(quantity):
                new_stock = current_stock - int(quantity)
                cursor.execute("UPDATE products SET stock = ? WHERE product_id = ?", (new_stock, product_id))
                cursor.execute("SELECT stock FROM products WHERE product_id = ?", (product_id,))
                updated_result = cursor.fetchone()
                logging.info(f"Updated stock for product {product_id}: {updated_result[0]}")
                if updated_result[0] < 20:  # Low stock threshold
                    send_email("Low Stock Alert", f"The stock for product {product_id} is low: {updated_result[0]}", EMAIL_ADDRESS)
            else:
                logging.warning(f"Not enough stock available for product {product_id}. Requested: {quantity}, Available: {current_stock}")
        else:
            logging.warning(f"No stock information found for product {product_id}")

    def add_order(self, order):
        if not self.validate_order(order):
            self.ui.message_label.config(text="Invalid order details")
            return

        conn = self.connect_db()
        try:
            cursor = conn.cursor()
            logging.info(f"Inserting order: {order}")
            cursor.execute("INSERT INTO orders (order_id, customer_name, date, status) VALUES (?, ?, datetime('now'), 'Processing')",
                           (order['order_id'], order['customer_name']))
            cursor.execute("INSERT INTO order_details (order_id, product_id, quantity) VALUES (?, ?, ?)",
                           (order['order_id'], order['product_id'], order['quantity']))
            
            cursor.execute("SELECT stock FROM products WHERE product_id = ?", (order['product_id'],))
            before_stock = cursor.fetchone()
            logging.info(f"Stock for product {order['product_id']} before reduction: {before_stock[0]}")
            
            self.reduce_stock(cursor, order['product_id'], int(order['quantity']))
            
            cursor.execute("SELECT stock FROM products WHERE product_id = ?", (order['product_id'],))
            after_stock = cursor.fetchone()
            logging.info(f"Stock for product {order['product_id']} after reduction: {after_stock[0]}")
            
            conn.commit()
            logging.info(f"Order {order['order_id']} added to orders and order_details tables and stock reduced.")
            
            # Detect fraud
            is_fraud, fraud_prob = detect_fraud(order, self.fraud_model)
            if is_fraud:
                self.ui.message_label.config(text=f"Order flagged as potentially fraudulent! Fraud Probability: {fraud_prob:.2f}")
                logging.warning(f"Order {order['order_id']} flagged as fraud. Fraud Probability: {fraud_prob:.2f}")
                cursor.execute("UPDATE orders SET fraud_flag = 1, fraud_prob = ? WHERE order_id = ?", (fraud_prob, order['order_id']))
            else:
                self.ui.message_label.config(text="Order added to the database.")
                cursor.execute("UPDATE orders SET fraud_flag = 0, fraud_prob = ? WHERE order_id = ?", (fraud_prob, order['order_id']))

            # Send order confirmation email
            send_email("Order Confirmation", f"Your order with Order ID {order['order_id']} has been placed successfully.", EMAIL_ADDRESS)
            
            # Add high-value order to the blockchain
            total_price = self.calculate_order_total(order['order_id'], cursor)
            if total_price > 1000:  # Define high-value threshold (e.g., $1000)
                block_data = {
                    'order_id': order['order_id'],
                    'customer_name': order['customer_name'],
                    'total_price': total_price,
                    'products': order['product_id']
                }
                new_block = Block(len(self.blockchain.chain), time.time(), str(block_data))
                self.blockchain.add_block(new_block)
                logging.info(f"Order {order['order_id']} added to blockchain.")
                
                self.ui.update_blockchain_ledger()
            
            conn.commit()

        except sqlite3.IntegrityError as e:
            logging.error(f"Integrity error: {e}")
            self.ui.message_label.config(text=f"Error: {e}")
        except Exception as e:
            logging.error(f"Error adding order: {e}")
            self.ui.message_label.config(text=f"An error occurred: {str(e)}")
        finally:
            conn.close()
            
    def calculate_order_total(self, order_id, cursor):
        # Calculate the total price of the order
        cursor.execute('''
            SELECT SUM(od.quantity * p.price) 
            FROM order_details od
            JOIN products p ON od.product_id = p.product_id
            WHERE od.order_id = ?
        ''', (order_id,))
        total_price = cursor.fetchone()[0]
        return total_price if total_price else 0

    def fetch_flagged_orders(self):
        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT order_id, customer_name, date, fraud_prob FROM orders WHERE fraud_flag = 1")
        flagged_orders = cursor.fetchall()
        conn.close()
        return flagged_orders

    def validate_order(self, order):
        # Validate order details
        if not order['customer_name'] or not order['product_id'] or not order['quantity'].isdigit():
            logging.warning("Validation failed for order: missing or invalid fields")
            return False
        if int(order['quantity']) <= 0:
            logging.warning("Validation failed for order: quantity must be positive")
            return False
        return True

    def update_order_status(self, order_id, new_status):
        conn = self.connect_db()
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE orders SET order_status = ? WHERE order_id = ?", (new_status, order_id))
            conn.commit()
            logging.info(f"Order ID {order_id} status updated to {new_status}")
        except Exception as e:
            logging.error(f"Failed to update order status: {e}")
        finally:
            conn.close()


    def fetch_orders_with_status(self):
        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT order_id, customer_name, date, order_status FROM orders")
        orders = cursor.fetchall()
        conn.close()
        return orders

    def fetch_inventory(self):
        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT name, stock FROM products")
        inventory = cursor.fetchall()
        conn.close()
        return inventory

    def plot_inventory_levels(self):
        logging.info("Fetching latest inventory data...")
        conn = None
        try:
            conn = self.connect_db()
            cursor = conn.cursor()
            cursor.execute("SELECT name, stock FROM products")
            data = cursor.fetchall()
            if data:
                products, stock_levels = zip(*data)
                logging.info(f"Latest stock levels: {list(stock_levels)}")
                plt.figure()
                plt.bar(products, stock_levels)
                plt.xlabel('Product Name')
                plt.ylabel('Stock Level')
                plt.title('Inventory Levels')
                plt.show()
            else:
                logging.info("No inventory data available.")
        finally:
            if conn:
                conn.close()


    def generate_sales_report(self):
        conn = self.connect_db()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT o.order_id, o.customer_name, o.date, p.name, od.quantity, p.price, (od.quantity * p.price) AS total_price
                FROM orders o
                JOIN order_details od ON o.order_id = od.order_id
                JOIN products p ON od.product_id = p.product_id
            ''')
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=['Order ID', 'Customer Name', 'Date', 'Product Name', 'Quantity', 'Price', 'Total Price'])
            logging.info(f"Sales report generated with {len(df)} entries.")
            return df
        finally:
            if conn:
                conn.close()
            
        # LSTM Model
    def predict_lstm(self, product_id):
        sales_data = self.fetch_sales_data(product_id)
        model, scaler = train_lstm_model(sales_data)
        predictions = lstm_predict_next_30_days(sales_data, model, scaler)
        return predictions

    # ARIMA Model
    def predict_arima(self, product_id):
        sales_data = self.fetch_sales_data(product_id)
        model_fit = train_arima_model(sales_data)
        predictions = arima_predict_next_30_days(model_fit)
        return predictions

    # Prophet Model
    def predict_prophet(self, product_id):
        sales_data = self.fetch_sales_data(product_id)
        model = train_prophet_model(sales_data)
        predictions = prophet_predict_next_30_days(model)
        return predictions

    def fetch_sales_data(self, product_id):
        conn = self.connect_db()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT o.date, od.quantity
            FROM order_details od
            JOIN orders o ON od.order_id = o.order_id
            WHERE od.product_id = ?
            ORDER BY o.date
        ''', (product_id,))

        data = cursor.fetchall()
        conn.close()

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['date', 'quantity'])
        df['date'] = pd.to_datetime(df['date'])  # Ensure dates are in correct format
        return df


    # Predictive Analytics for Stock Management
    def predict_stock_requirements(product_id):
        conn = sqlite3.connect('order_system.db')
        cursor = conn.cursor()
        
        # Fetch historical sales data for the product
        cursor.execute('''
            SELECT od.quantity, o.date
            FROM order_details od
            JOIN orders o ON od.order_id = o.order_id
            WHERE od.product_id = ?
            ORDER BY o.date
        ''', (product_id,))
        
        data = cursor.fetchall()
        conn.close()
        
        if not data:
            logging.warning(f"No historical data available for product {product_id}")
            return None
        
        # Prepare the data for linear regression
        quantities = [row[0] for row in data]
        dates = pd.to_datetime([row[1] for row in data])
        days = (dates - dates.min()).days.values.reshape(-1, 1)
        
        # Train the linear regression model
        model = LinearRegression()
        model.fit(days, quantities)
        
        # Predict future stock requirements (e.g., for the next 30 days)
        future_days = np.arange(days.max() + 1, days.max() + 31).reshape(-1, 1)
        future_predictions = model.predict(future_days)
        
        return int(np.ceil(future_predictions.sum()))
    
    # LSTM Model Functions
def train_lstm_model(data, feature='quantity'):
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature].values.reshape(-1, 1))

    # Prepare input/output sequences for LSTM
    x_train, y_train = [], []
    sequence_length = 30  # last 30 days data

    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # [samples, time steps, features]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=20)

    return model, scaler

def lstm_predict_next_30_days(data, model, scaler):
    last_30_days = data[-30:].values.reshape(-1, 1)
    scaled_last_30_days = scaler.transform(last_30_days)

    # Predict the next 30 days
    x_input = np.reshape(scaled_last_30_days, (1, scaled_last_30_days.shape[0], 1))
    predictions = []

    for i in range(30):
        pred = model.predict(x_input)
        predictions.append(pred[0, 0])
        x_input = np.append(x_input[:, 1:, :], [[pred]], axis=1)

    predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted_values

# ARIMA Model Functions
def train_arima_model(data, feature='quantity'):
    # Fit ARIMA model
    model = sm.tsa.ARIMA(data[feature].values, order=(5, 1, 0))  # Adjust order for ARIMA
    model_fit = model.fit(disp=0)
    return model_fit

def arima_predict_next_30_days(model_fit):
    # Predict next 30 days
    forecast = model_fit.forecast(steps=30)[0]
    return forecast

# Prophet Model Functions
def train_prophet_model(data):
    # Prepare data for Prophet
    prophet_data = data.rename(columns={'date': 'ds', 'quantity': 'y'})

    # Initialize Prophet model
    model = Prophet()
    model.fit(prophet_data)

    return model

def prophet_predict_next_30_days(model):
    # Predict future values for the next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']][-30:]  # Return the last 30 days of prediction


# Real-time Dashboard Class
# Real-time Dashboard Class
class RealTimeDashboard:
    def __init__(self, master):
        self.master = master
        
        self.label_sales = ttk.Label(master, text="Total Sales: $0", font=("Helvetica", 16))
        self.label_sales.pack(pady=20)
        
        self.label_orders = ttk.Label(master, text="Total Orders: 0", font=("Helvetica", 16))
        self.label_orders.pack(pady=20)
        
        self.label_low_stock = ttk.Label(master, text="Low Stock Alerts: None", font=("Helvetica", 16))
        self.label_low_stock.pack(pady=20)
        
        self.update_dashboard()

    def update_dashboard(self):
        logging.info("Updating dashboard...")
        conn = sqlite3.connect('order_system.db')
        cursor = conn.cursor()
        
        # Update total sales
        cursor.execute('''
            SELECT SUM(od.quantity * p.price)
            FROM order_details od
            JOIN products p ON od.product_id = p.product_id
        ''')
        total_sales = cursor.fetchone()[0] or 0
        self.label_sales.config(text=f"Total Sales: ${total_sales:.2f}")
        
        # Update total orders
        cursor.execute("SELECT COUNT(*) FROM orders")
        total_orders = cursor.fetchone()[0]
        self.label_orders.config(text=f"Total Orders: {total_orders}")
        
        # Update low stock alerts
        cursor.execute("SELECT name FROM products WHERE stock < 20")
        low_stock_items = cursor.fetchall()
        if low_stock_items:
            low_stock_text = "Low Stock Alerts: " + ", ".join([item[0] for item in low_stock_items])
        else:
            low_stock_text = "Low Stock Alerts: None"
        self.label_low_stock.config(text=low_stock_text)
        
        conn.close()

        # Schedule the next update in the main thread
        self.master.after(DASHBOARD_UPDATE_INTERVAL * 1000, self.update_dashboard)


    def start_dashboard_updates(self):
        self.update_dashboard()
        threading.Timer(DASHBOARD_UPDATE_INTERVAL, self.start_dashboard_updates).start()

# Main Application Class (Updated)
# Main Application Class (Updated)
class OrderProcessingUI:
    def __init__(self, master, username, user_role):
        self.master = master
        self.username = username
        self.user_role = user_role
        self.master.title(f"Order Processing System - Logged in as {username}")
        self.master.geometry("800x600")

        self.order_processor = OrderProcessor(self)
        notebook = ttk.Notebook(master)
        self.order_tab = ttk.Frame(notebook)
        self.inventory_tab = ttk.Frame(notebook)
        self.analytics_tab = None
        


        if self.user_role == 'admin':
            self.admin_tab = ttk.Frame(notebook)
            notebook.add(self.admin_tab, text='Admin')

        self.analytics_tab = ttk.Frame(notebook)
        notebook.add(self.analytics_tab, text='Sales Analytics')
        
        self.dashboard_tab = ttk.Frame(notebook)
        notebook.add(self.dashboard_tab, text='Dashboard')
        
        # Adding the Blockchain Ledger Tab
        self.blockchain_tab = ttk.Frame(notebook)
        notebook.add(self.blockchain_tab, text='Blockchain Ledger')
        self.setup_blockchain_tab()

        # Adding the Fraud Detection Tab
        self.fraud_tab = ttk.Frame(notebook)
        notebook.add(self.fraud_tab, text='Fraud Detection')
        self.setup_fraud_dashboard_tab()

        # Adding the Chatbot Tab
        self.chatbot_tab = ttk.Frame(notebook)
        notebook.add(self.chatbot_tab, text='Chatbot')
        ChatbotTab(self.chatbot_tab)  # Initialize the Chatbot tab UI

        notebook.add(self.order_tab, text='Order Entry')
        notebook.add(self.inventory_tab, text='Inventory Management')
        notebook.pack(expand=1, fill="both")

        self.setup_order_tab()
        self.setup_inventory_tab()
        if self.admin_tab:
            self.setup_admin_tab()
        if self.analytics_tab:
            self.setup_analytics_tab()
        if self.dashboard_tab:
            self.setup_dashboard_tab()
            
    def setup_blockchain_tab(self):
        ttk.Label(self.blockchain_tab, text="Blockchain Ledger").pack(pady=10)
        columns = ("Block Index", "Timestamp", "Order Data", "Previous Hash", "Hash")
        self.blockchain_tree = ttk.Treeview(self.blockchain_tab, columns=columns, show='headings')
        for col in columns:
            self.blockchain_tree.heading(col, text=col)
            self.blockchain_tree.column(col, width=150, anchor=tk.CENTER)
        self.blockchain_tree.pack(fill=tk.BOTH, expand=True)

        self.load_blockchain_data()
        
        self.update_blockchain_ledger()

    def load_blockchain_data(self):
        for block in self.order_processor.blockchain.chain:
            self.blockchain_tree.insert("", tk.END, values=(block.index, block.timestamp, block.data, block.previous_hash, block.hash))
            
    def update_blockchain_ledger(self):
    # Clear existing blockchain ledger treeview
        for row in self.blockchain_tree.get_children():
            self.blockchain_tree.delete(row)
    
        # Insert all blocks into the treeview
        for block in self.order_processor.blockchain.chain:
            self.blockchain_tree.insert("", tk.END, values=(
                block.index,
                block.timestamp,
                block.data,
                block.previous_hash,
                block.hash
            ))


    def setup_fraud_dashboard_tab(self):
        ttk.Label(self.fraud_tab, text="Flagged Fraudulent Orders").pack(pady=10)
        columns = ("Order ID", "Customer Name", "Date", "Fraud Probability")
        self.fraud_tree = ttk.Treeview(self.fraud_tab, columns=columns, show='headings')
        for col in columns:
            self.fraud_tree.heading(col, text=col)
            self.fraud_tree.column(col, width=150, anchor=tk.CENTER)
        self.fraud_tree.pack(fill=tk.BOTH, expand=True)

        self.load_fraud_data()

    def load_fraud_data(self):
        orders = self.order_processor.fetch_flagged_orders()
        for order in orders:
            self.fraud_tree.insert("", tk.END, values=order)

    # Other setup functions...

    def setup_order_tab(self):
        ttk.Label(self.order_tab, text="Order ID:").pack()
        self.order_id_label = ttk.Label(self.order_tab, text="")
        self.order_id_label.pack()

        ttk.Label(self.order_tab, text="Customer Name:").pack()
        self.customer_name_entry = ttk.Entry(self.order_tab)
        self.customer_name_entry.pack()

        ttk.Label(self.order_tab, text="Product ID:").pack()
        self.product_id_entry = ttk.Entry(self.order_tab)
        self.product_id_entry.pack()

        ttk.Label(self.order_tab, text="Quantity:").pack()
        self.quantity_entry = ttk.Entry(self.order_tab)
        self.quantity_entry.pack()

        ttk.Button(self.order_tab, text="Submit Order", command=self.submit_order).pack()
        ttk.Button(self.order_tab, text="Cancel Order", command=self.cancel_order).pack()

        self.message_label = ttk.Label(self.order_tab, text="")
        self.message_label.pack()

        ttk.Label(self.order_tab, text="Update Order Status").pack(pady=10)
        self.order_status_entry = ttk.Entry(self.order_tab)
        self.order_status_entry.pack()

        ttk.Button(self.order_tab, text="Update Status", command=self.update_status).pack(pady=10)
    
    def update_status(self):
        order_id = self.order_id_label.cget("text").split(": ")[1]
        new_status = self.order_status_entry.get()
        self.order_processor.update_order_status(order_id, new_status)
        self.message_label.config(text=f"Order ID {order_id} status updated to {new_status}")

    def setup_inventory_tab(self):
        ttk.Button(self.inventory_tab, text="Plot Inventory Levels", command=self.order_processor.plot_inventory_levels).pack(pady=20)
        ttk.Button(self.inventory_tab, text="Generate Sales Report", command=self.generate_sales_report).pack(pady=20)

    def setup_admin_tab(self):
        ttk.Label(self.admin_tab, text="Add New Product").pack()
        ttk.Label(self.admin_tab, text="Product ID:").pack()
        self.new_product_id_entry = ttk.Entry(self.admin_tab)
        self.new_product_id_entry.pack()

        ttk.Label(self.admin_tab, text="Product Name:").pack()
        self.new_product_name_entry = ttk.Entry(self.admin_tab)
        self.new_product_name_entry.pack()

        ttk.Label(self.admin_tab, text="Price:").pack()
        self.new_product_price_entry = ttk.Entry(self.admin_tab)
        self.new_product_price_entry.pack()

        ttk.Label(self.admin_tab, text="Stock:").pack()
        self.new_product_stock_entry = ttk.Entry(self.admin_tab)
        self.new_product_stock_entry.pack()

        ttk.Button(self.admin_tab, text="Add Product", command=self.add_product).pack()
        ttk.Button(self.admin_tab, text="Update Product", command=self.update_product).pack()
        ttk.Button(self.admin_tab, text="Delete Product", command=self.delete_product).pack()

        ttk.Label(self.admin_tab, text="Approve/Reject Registrations").pack(pady=10)
        self.registration_listbox = tk.Listbox(self.admin_tab)
        self.registration_listbox.pack()
        self.load_registrations()

        ttk.Button(self.admin_tab, text="Approve", command=self.approve_registration).pack(side=tk.LEFT, padx=10)
        ttk.Button(self.admin_tab, text="Reject", command=self.reject_registration).pack(side=tk.LEFT)

        self.admin_message_label = ttk.Label(self.admin_tab, text="")
        self.admin_message_label.pack()

    def setup_analytics_tab(self):
        ttk.Button(self.analytics_tab, text="View Sales Summary", command=self.view_sales_summary).pack(pady=10)
        ttk.Button(self.analytics_tab, text="Sales by Product", command=self.plot_sales_by_product).pack(pady=10)
        ttk.Button(self.analytics_tab, text="Monthly Sales", command=self.plot_monthly_sales).pack(pady=10)

    def setup_dashboard_tab(self):
        dashboard = RealTimeDashboard(self.dashboard_tab)

    def load_registrations(self):
        conn = sqlite3.connect('order_system.db')
        cursor = conn.cursor()
        cursor.execute("SELECT registration_id, username FROM user_registrations WHERE status = 'Pending'")
        registrations = cursor.fetchall()
        self.registration_listbox.delete(0, tk.END)
        for reg in registrations:
            self.registration_listbox.insert(tk.END, f"{reg[0]}: {reg[1]}")
        conn.close()

    def approve_registration(self):
        selection = self.registration_listbox.curselection()
        if selection:
            reg_id = self.registration_listbox.get(selection[0]).split(":")[0]
            conn = sqlite3.connect('order_system.db')
            cursor = conn.cursor()
            cursor.execute("SELECT username, password, email, role FROM user_registrations WHERE registration_id = ?", (reg_id,))
            user_data = cursor.fetchone()
            if user_data:
                cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (user_data[0], user_data[1], user_data[3]))
                cursor.execute("UPDATE user_registrations SET status = 'Approved' WHERE registration_id = ?", (reg_id,))
                conn.commit()
                self.admin_message_label.config(text="User approved and added to the system.")
            conn.close()
            self.load_registrations()

    def reject_registration(self):
        selection = self.registration_listbox.curselection()
        if selection:
            reg_id = self.registration_listbox.get(selection[0]).split(":")[0]
            conn = sqlite3.connect('order_system.db')
            cursor = conn.cursor()
            cursor.execute("UPDATE user_registrations SET status = 'Rejected' WHERE registration_id = ?", (reg_id,))
            conn.commit()
            conn.close()
            self.load_registrations()
            self.admin_message_label.config(text="User registration rejected.")

    def add_product(self):
        product_id = self.new_product_id_entry.get()
        product_name = self.new_product_name_entry.get()
        price = self.new_product_price_entry.get()
        stock = self.new_product_stock_entry.get()

        if not product_id or not product_name or not price or not stock:
            self.admin_message_label.config(text="All fields are required")
            return

        try:
            price = float(price)
            stock = int(stock)
        except ValueError:
            self.admin_message_label.config(text="Invalid price or stock value")
            return

        try:
            with sqlite3.connect('order_system.db') as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO products (product_id, name, price, stock) VALUES (?, ?, ?, ?)",
                               (product_id, product_name, price, stock))
                conn.commit()
            self.admin_message_label.config(text="Product added successfully")
        except sqlite3.IntegrityError:
            self.admin_message_label.config(text="Product ID already exists")
        except Exception as e:
            self.admin_message_label.config(text=f"An error occurred: {str(e)}")

    def update_product(self):
        product_id = self.new_product_id_entry.get()
        product_name = self.new_product_name_entry.get()
        price = self.new_product_price_entry.get()
        stock = self.new_product_stock_entry.get()

        if not product_id or not product_name or not price or not stock:
            self.admin_message_label.config(text="All fields are required")
            return

        try:
            price = float(price)
            stock = int(stock)
        except ValueError:
            self.admin_message_label.config(text="Invalid price or stock value")
            return

        try:
            with sqlite3.connect('order_system.db') as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE products SET name = ?, price = ?, stock = ? WHERE product_id = ?",
                               (product_name, price, stock, product_id))
                conn.commit()
            self.admin_message_label.config(text="Product updated successfully")
        except sqlite3.IntegrityError:
            self.admin_message_label.config(text="Error updating product")
        except Exception as e:
            self.admin_message_label.config(text=f"An error occurred: {str(e)}")

    def delete_product(self):
        product_id = self.new_product_id_entry.get()

        if not product_id:
            self.admin_message_label.config(text="Product ID is required")
            return

        try:
            with sqlite3.connect('order_system.db') as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
                conn.commit()
            self.admin_message_label.config(text="Product deleted successfully")
        except Exception as e:
            self.admin_message_label.config(text=f"An error occurred: {str(e)}")

    def submit_order(self):
        order_id = self.order_processor.generate_order_id()
        order_details = {
            "order_id": order_id,
            "customer_name": self.customer_name_entry.get(),
            "product_id": self.product_id_entry.get(),
            "quantity": self.quantity_entry.get()
        }
        self.order_id_label.config(text=f"Order ID: {order_id}")
        logging.info(f"Submitting order: {order_details}")
        self.order_processor.add_order(order_details)

    def cancel_order(self):
        self.order_id_label.config(text="")
        self.customer_name_entry.delete(0, tk.END)
        self.product_id_entry.delete(0, tk.END)
        self.quantity_entry.delete(0, tk.END)
        self.message_label.config(text="Order cancelled. All fields cleared.")

    def generate_sales_report(self):
        df = self.order_processor.generate_sales_report()
        if df is not None:
            report_window = tk.Toplevel(self.master)
            report_window.title("Sales Report")

            frame = ttk.Frame(report_window)
            frame.pack(fill=tk.BOTH, expand=True)

            tree = ttk.Treeview(frame, columns=list(df.columns), show='headings')
            for col in df.columns:
                tree.heading(col, text=col)
                tree.column(col, width=100, anchor=tk.CENTER)
            for index, row in df.iterrows():
                tree.insert("", tk.END, values=list(row))
            tree.pack(fill=tk.BOTH, expand=True)

            export_button = ttk.Button(report_window, text="Export to CSV", command=lambda: self.export_to_csv(df))
            export_button.pack(pady=10)

    def export_to_csv(self, df):
        df.to_csv('sales_report.csv', index=False)
        logging.info("Sales report exported to sales_report.csv")

    def view_sales_summary(self):
        df = self.order_processor.generate_sales_report()
        if df is not None:
            summary = df.groupby('Product Name')['Total Price'].sum()
            plt.figure()
            summary.plot(kind='bar', title='Sales Summary by Product')
            plt.xlabel('Product Name')
            plt.ylabel('Total Sales')
            plt.show()

    def plot_sales_by_product(self):
        df = self.order_processor.generate_sales_report()
        if df is not None:
            plt.figure()
            sns.barplot(x='Product Name', y='Quantity', data=df, estimator=sum)
            plt.title('Total Sales by Product')
            plt.show()

    def plot_monthly_sales(self):
        df = self.order_processor.generate_sales_report()
        if df is not None:
            df['Date'] = pd.to_datetime(df['Date'])
            monthly_sales = df.resample('ME', on='Date').sum()
            plt.figure()
            monthly_sales['Total Price'].plot(kind='line', marker='o')
            plt.title('Monthly Sales')
            plt.xlabel('Month')
            plt.ylabel('Total Sales')
            plt.show()

if __name__ == "__main__":
    init_db()
    alter_orders_table()  # Add this line to modify the existing orders table
    add_initial_products()
    add_initial_users()
    root = tk.Tk()
    login_app = LoginUI(root)
    root.mainloop()
