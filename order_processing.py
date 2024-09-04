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
import seaborn as sns  # Ensure seaborn is imported
import time
import openai  # Ensure openai is imported for Chatbot

# Configure logging
logging.basicConfig(filename='order_system.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# OpenAI API Key setup for Chatbot
openai.api_key = 'AP1 Key'  # Replace with your OpenAI API key

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'bryansamjames@gmail.com'
EMAIL_PASSWORD = 'aznh qyep jfvd izel'  # Replace this with your actual app password

# Real-time Dashboard Update Interval (in seconds)
DASHBOARD_UPDATE_INTERVAL = 10

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
            order_status TEXT DEFAULT 'Processing'
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
    logging.info("Database initialized")

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

    def calculate_order_priority(self, order):
        """Calculate order priority based on criteria: order value, customer loyalty, and delivery deadlines"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # Get the total value of the order
        cursor.execute('''
            SELECT SUM(p.price * od.quantity) AS total_order_value
            FROM order_details od
            JOIN products p ON od.product_id = p.product_id
            WHERE od.order_id = ?
        ''', (order['order_id'],))
        total_order_value = cursor.fetchone()[0] or 0

        # Get customer loyalty score (number of previous orders)
        cursor.execute('''
            SELECT COUNT(*) FROM orders WHERE customer_name = ?
        ''', (order['customer_name'],))
        customer_loyalty_score = cursor.fetchone()[0]

        # Deadline priority (For now, an arbitrary fixed value. In future, can be tied to order deadlines)
        deadline_priority = 10  # Example priority based on deadline proximity
        
        # Calculate priority score based on weighted sum of the criteria
        priority_score = (total_order_value * 0.5) + (customer_loyalty_score * 0.3) + (deadline_priority * 0.2)
        
        conn.close()
        return priority_score

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
            
            self.ui.message_label.config(text="Order added to the database.")
            
            # Send order confirmation email
            send_email("Order Confirmation", f"Your order with Order ID {order['order_id']} has been placed successfully.", EMAIL_ADDRESS)
        except sqlite3.IntegrityError as e:
            logging.error(f"Integrity error: {e}")
            self.ui.message_label.config(text=f"Error: {e}")
        except Exception as e:
            logging.error(f"Error adding order: {e}")
            self.ui.message_label.config(text=f"An error occurred: {str(e)}")
        finally:
            conn.close()

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
        
        self.start_dashboard_updates()

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

    def start_dashboard_updates(self):
        self.update_dashboard()
        threading.Timer(DASHBOARD_UPDATE_INTERVAL, self.start_dashboard_updates).start()

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
    add_initial_products()
    add_initial_users()
    root = tk.Tk()
    login_app = LoginUI(root)
    root.mainloop()
