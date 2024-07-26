import sqlite3
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import pandas as pd
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Configure logging
logging.basicConfig(filename='order_system.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'bryansamjames@gmail.com'
EMAIL_PASSWORD = 'abcd efgh ijkl mnop'  # Replace this with your actual app password

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
        ('admin', 'password', 'admin'),
        ('manager', 'password', 'manager'),
        ('employee', 'password', 'employee')
    ]
    try:
        with sqlite3.connect('order_system.db') as conn:
            c = conn.cursor()
            c.executemany('INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)', users)
            conn.commit()
        logging.info("Initial users added")
    except sqlite3.OperationalError as e:
        logging.error(f"Error adding initial users: {e}")

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
        cursor.execute("SELECT role FROM users WHERE username = ? AND password = ?", (username, password))
        result = cursor.fetchone()
        conn.close()
        if result:
            self.user_role = result[0]
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

    def fetch_orders(self):
        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT order_id, customer_name, date, status FROM orders")
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

# GUI Class
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
        self.admin_tab = None
        if self.user_role == 'admin':
            self.admin_tab = ttk.Frame(notebook)
            notebook.add(self.admin_tab, text='Admin')
        notebook.add(self.order_tab, text='Order Entry')
        notebook.add(self.inventory_tab, text='Inventory Management')
        notebook.pack(expand=1, fill="both")

        self.setup_order_tab()
        self.setup_inventory_tab()
        if self.admin_tab:
            self.setup_admin_tab()

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
        self.admin_message_label = ttk.Label(self.admin_tab, text="")
        self.admin_message_label.pack()

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

if __name__ == "__main__":
    init_db()
    add_initial_products()
    add_initial_users()
    root = tk.Tk()
    login_app = LoginUI(root)
    root.mainloop()
