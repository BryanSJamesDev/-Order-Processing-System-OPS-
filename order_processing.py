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
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")


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
            print(f"Current stock for product {product_id}: {current_stock}")
            if current_stock >= int(quantity):
                new_stock = current_stock - int(quantity)
                cursor.execute("UPDATE products SET stock = ? WHERE product_id = ?", (new_stock, product_id))
                cursor.execute("SELECT stock FROM products WHERE product_id = ?", (product_id,))
                updated_result = cursor.fetchone()
                print(f"Updated stock for product {product_id}: {updated_result[0]}")
            else:
                print("Not enough stock available.")
        else:
            print(f"No stock information found for product {product_id}.")

    def add_order(self, order):
        conn = self.connect_db()
        try:
            cursor = conn.cursor()
            print(f"Inserting order: {order}")
            cursor.execute("INSERT INTO orders (order_id, customer_name, date, status) VALUES (?, ?, datetime('now'), 'Processing')",
                           (order['order_id'], order['customer_name']))
            cursor.execute("INSERT INTO order_details (order_id, product_id, quantity) VALUES (?, ?, ?)",
                           (order['order_id'], order['product_id'], order['quantity']))
            
            # Check stock before reducing
            cursor.execute("SELECT stock FROM products WHERE product_id = ?", (order['product_id'],))
            before_stock = cursor.fetchone()
            print(f"Stock for product {order['product_id']} before reduction: {before_stock[0]}")
            
            # Reduce stock within the same transaction
            self.reduce_stock(cursor, order['product_id'], int(order['quantity']))
            
            # Check stock after reducing
            cursor.execute("SELECT stock FROM products WHERE product_id = ?", (order['product_id'],))
            after_stock = cursor.fetchone()
            print(f"Stock for product {order['product_id']} after reduction: {after_stock[0]}")
            
            conn.commit()
            print(f"Order {order['order_id']} added to orders and order_details tables and stock reduced.")
            
            self.ui.message_label.config(text="Order added to the database.")
        except sqlite3.IntegrityError as e:
            self.ui.message_label.config(text=f"Error: {e}")
        except Exception as e:
            self.ui.message_label.config(text=f"An error occurred: {str(e)}")
        finally:
            conn.close()

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
        print("Fetching latest inventory data...")
        conn = None
        try:
            conn = self.connect_db()
            cursor = conn.cursor()
            cursor.execute("SELECT name, stock FROM products")
            data = cursor.fetchall()
            if data:
                products, stock_levels = zip(*data)
                print(f"Latest stock levels: {list(stock_levels)}")  # Debug output to check stock levels
                plt.figure()
                plt.bar(products, stock_levels)
                plt.xlabel('Product Name')
                plt.ylabel('Stock Level')
                plt.title('Inventory Levels')
                plt.show()
            else:
                print("No inventory data available.")
        finally:
            if conn:
                conn.close()
                
# GUI Class
class OrderProcessingUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Order Processing System")
        self.master.geometry("800x600")

        self.order_processor = OrderProcessor(self)
        notebook = ttk.Notebook(master)
        self.order_tab = ttk.Frame(notebook)
        self.inventory_tab = ttk.Frame(notebook)
        notebook.add(self.order_tab, text='Order Entry')
        notebook.add(self.inventory_tab, text='Inventory Management')
        notebook.pack(expand=1, fill="both")

        self.setup_order_tab()
        self.setup_inventory_tab()

    def setup_order_tab(self):
        ttk.Label(self.order_tab, text="Order ID:").pack()
        self.order_id_entry = ttk.Entry(self.order_tab)
        self.order_id_entry.pack()

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

    def submit_order(self):
        order_details = {
            "order_id": self.order_processor.generate_order_id(),
            "customer_name": self.customer_name_entry.get(),
            "product_id": self.product_id_entry.get(),
            "quantity": self.quantity_entry.get()
        }
        print(f"Submitting order: {order_details}")  # Debug statement to check order details
        self.order_processor.add_order(order_details)
        self.message_label.config(text="Order submitted successfully.")

    def cancel_order(self):
        self.order_id_entry.delete(0, tk.END)
        self.customer_name_entry.delete(0, tk.END)
        self.product_id_entry.delete(0, tk.END)
        self.quantity_entry.delete(0, tk.END)
        self.message_label.config(text="Order cancelled. All fields cleared.")

if __name__ == "__main__":
    init_db()
    add_initial_products()  # This ensures initial data is loaded
    root = tk.Tk()
    app = OrderProcessingUI(root)
    root.mainloop()

