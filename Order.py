import sqlite3
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# Initialize Database
def init_db():
    conn = sqlite3.connect('order_system.db')
    c = conn.cursor()
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
            status TEXT
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
    conn.close()
