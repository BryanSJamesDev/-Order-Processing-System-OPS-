# order_processing_system/database.py

import sqlite3
import logging
from config import DATABASE_PATH
from utils import hash_password

def connect_db():
    """
    Establishes and returns a new database connection.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute('PRAGMA foreign_keys = ON')
    return conn

def init_db():
    """
    Initializes the database with necessary tables.
    """
    conn = connect_db()
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
            fraud_flag INTEGER DEFAULT 0,  -- 0 for non-fraud, 1 for fraud
            fraud_prob REAL DEFAULT 0.0    -- Probability of fraud
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
    """
    Alters the orders table to add fraud_flag and fraud_prob columns if they don't exist.
    """
    conn = connect_db()
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE orders ADD COLUMN fraud_flag INTEGER DEFAULT 0")
        c.execute("ALTER TABLE orders ADD COLUMN fraud_prob REAL DEFAULT 0.0")
        logging.info("Added fraud_flag and fraud_prob columns to the orders table.")
    except sqlite3.OperationalError as e:
        logging.warning(f"Columns already exist or failed to add: {e}")
    conn.commit()
    conn.close()

def add_initial_products():
    """
    Populates the products table with initial data.
    """
    products = [
        ('P001', 'Widget A', 19.99, 100),
        ('P002', 'Widget B', 25.99, 200),
        ('P003', 'Widget C', 9.99, 150)
    ]
    try:
        conn = connect_db()
        c = conn.cursor()
        c.executemany('''
            INSERT OR IGNORE INTO products (product_id, name, price, stock)
            VALUES (?, ?, ?, ?)
        ''', products)
        conn.commit()
        conn.close()
        logging.info("Initial products added.")
    except sqlite3.OperationalError as e:
        logging.error(f"Error adding initial products: {e}")

def add_initial_users():
    """
    Populates the users table with initial data.
    """
    users = [
        ('admin', hash_password('password'), 'admin'),
        ('manager', hash_password('password'), 'manager'),
        ('employee', hash_password('password'), 'employee')
    ]
    try:
        conn = connect_db()
        c = conn.cursor()
        c.executemany('''
            INSERT OR IGNORE INTO users (username, password, role)
            VALUES (?, ?, ?)
        ''', users)
        conn.commit()
        conn.close()
        logging.info("Initial users added.")
    except sqlite3.OperationalError as e:
        logging.error(f"Error adding initial users: {e}")
