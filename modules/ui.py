# order_processing_system/ui.py

import tkinter as tk
from tkinter import ttk
import logging
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from blockchain import Blockchain, Block
from fraud_detection import load_fraud_detection_model, detect_fraud
from database import connect_db
from email_utils import send_email
from chatbot import ChatbotTab
from analytics import (
    train_lstm_model,
    lstm_predict_next_30_days,
    train_arima_model,
    arima_predict_next_30_days,
    train_prophet_model,
    prophet_predict_next_30_days
)
from utils import verify_password, hash_password, get_current_timestamp
from config import EMAIL_ADDRESS, DASHBOARD_UPDATE_INTERVAL

class LoginUI:
    """UI component for user login."""
    def __init__(self, master):
        self.master = master
        self.master.title("Login")
        self.master.geometry("300x200")

        ttk.Label(master, text="Username:").pack(pady=5)
        self.username_entry = ttk.Entry(master)
        self.username_entry.pack(pady=5)

        ttk.Label(master, text="Password:").pack(pady=5)
        self.password_entry = ttk.Entry(master, show="*")
        self.password_entry.pack(pady=5)

        ttk.Button(master, text="Login", command=self.login).pack(pady=10)
        self.message_label = ttk.Label(master, text="")
        self.message_label.pack()

    def login(self):
        """Handles user login."""
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
        """Authenticates the user against the database."""
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT password, role FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        if result and verify_password(result[0], password):
            self.user_role = result[1]
            logging.info(f"User '{username}' authenticated successfully as '{self.user_role}'.")
            return True
        else:
            logging.warning(f"Failed login attempt for user '{username}'.")
            return False

class OrderProcessor:
    """Handles order processing logic."""
    def __init__(self, ui):
        self.ui = ui
        self.fraud_model = load_fraud_detection_model()  # Load fraud detection model
        self.blockchain = Blockchain()

    def generate_order_id(self):
        """Generates a new order ID."""
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(CAST(order_id AS INTEGER)) FROM orders")
        max_id = cursor.fetchone()[0]
        conn.close()
        return int(max_id) + 1 if max_id else 1

    def reduce_stock(self, cursor, product_id, quantity):
        """Reduces stock for a given product."""
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
                    send_email(
                        "Low Stock Alert",
                        f"The stock for product {product_id} is low: {updated_result[0]}",
                        EMAIL_ADDRESS
                    )
            else:
                logging.warning(
                    f"Not enough stock available for product {product_id}. Requested: {quantity}, Available: {current_stock}"
                )
        else:
            logging.warning(f"No stock information found for product {product_id}")

    def add_order(self, order):
        """Adds a new order to the system."""
        if not self.validate_order(order):
            self.ui.message_label.config(text="Invalid order details")
            return

        conn = connect_db()
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
        """
        Calculates the total price of an order.
        """
        cursor.execute('''
            SELECT SUM(od.quantity * p.price) 
            FROM order_details od
            JOIN products p ON od.product_id = p.product_id
            WHERE od.order_id = ?
        ''', (order_id,))
        total_price = cursor.fetchone()[0]
        return total_price if total_price else 0

    def fetch_flagged_orders(self):
        """
        Fetches orders flagged as potentially fraudulent.
        """
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT order_id, customer_name, date, fraud_prob FROM orders WHERE fraud_flag = 1")
        flagged_orders = cursor.fetchall()
        conn.close()
        return flagged_orders

    def validate_order(self, order):
        """
        Validates order details.
        """
        if not order['customer_name'] or not order['product_id'] or not str(order['quantity']).isdigit():
            logging.warning("Validation failed for order: missing or invalid fields")
            return False
        if int(order['quantity']) <= 0:
            logging.warning("Validation failed for order: quantity must be positive")
            return False
        return True

    def update_order_status(self, order_id, new_status):
        """
        Updates the status of an existing order.
        """
        conn = connect_db()
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
        """
        Fetches all orders with their statuses.
        """
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT order_id, customer_name, date, order_status FROM orders")
        orders = cursor.fetchall()
        conn.close()
        return orders

    def fetch_inventory(self):
        """
        Fetches current inventory levels.
        """
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT name, stock FROM products")
        inventory = cursor.fetchall()
        conn.close()
        return inventory

    def plot_inventory_levels(self):
        """
        Plots current inventory levels.
        """
        logging.info("Fetching latest inventory data...")
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT name, stock FROM products")
        data = cursor.fetchall()
        conn.close()
        if data:
            products, stock_levels = zip(*data)
            logging.info(f"Latest stock levels: {list(stock_levels)}")
            plt.figure(figsize=(10,6))
            plt.bar(products, stock_levels, color='skyblue')
            plt.xlabel('Product Name')
            plt.ylabel('Stock Level')
            plt.title('Inventory Levels')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            logging.info("No inventory data available.")

    def generate_sales_report(self):
        """
        Generates a sales report and displays it in a new window.
        """
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
        else:
            self.ui.message_label.config(text="No sales data available.")

    def export_to_csv(self, df):
        """
        Exports the sales report to a CSV file.
        """
        try:
            df.to_csv('sales_report.csv', index=False)
            logging.info("Sales report exported to sales_report.csv")
            self.ui.message_label.config(text="Sales report exported to sales_report.csv")
        except Exception as e:
            logging.error(f"Failed to export sales report: {e}")
            self.ui.message_label.config(text=f"Failed to export report: {e}")

    def view_sales_summary(self):
        """
        Displays a sales summary by product.
        """
        df = self.order_processor.generate_sales_report()
        if df is not None:
            summary = df.groupby('Product Name')['Total Price'].sum()
            plt.figure(figsize=(10,6))
            summary.plot(kind='bar', title='Sales Summary by Product', color='orange')
            plt.xlabel('Product Name')
            plt.ylabel('Total Sales ($)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def plot_sales_by_product(self):
        """
        Plots total sales by product.
        """
        df = self.order_processor.generate_sales_report()
        if df is not None:
            plt.figure(figsize=(10,6))
            sns.barplot(x='Product Name', y='Quantity', data=df, estimator=sum, palette='viridis')
            plt.title('Total Sales by Product')
            plt.xlabel('Product Name')
            plt.ylabel('Total Quantity Sold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def plot_monthly_sales(self):
        """
        Plots monthly sales over time.
        """
        df = self.order_processor.generate_sales_report()
        if df is not None:
            df['Date'] = pd.to_datetime(df['Date'])
            monthly_sales = df.resample('M', on='Date').sum()
            plt.figure(figsize=(10,6))
            monthly_sales['Total Price'].plot(kind='line', marker='o', color='green')
            plt.title('Monthly Sales Over Time')
            plt.xlabel('Month')
            plt.ylabel('Total Sales ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def predict_lstm(self, product_id):
        """
        Predicts sales for the next 30 days using LSTM.
        """
        sales_data = self.fetch_sales_data(product_id)
        if sales_data.empty:
            logging.warning(f"No sales data available for LSTM prediction for product {product_id}.")
            return
        model, scaler = train_lstm_model(sales_data)
        predictions = lstm_predict_next_30_days(sales_data, model, scaler)
        logging.info(f"LSTM Predictions for product {product_id}: {predictions.flatten()}")
        return predictions

    def predict_arima(self, product_id):
        """
        Predicts sales for the next 30 days using ARIMA.
        """
        sales_data = self.fetch_sales_data(product_id)
        if sales_data.empty:
            logging.warning(f"No sales data available for ARIMA prediction for product {product_id}.")
            return
        model_fit = train_arima_model(sales_data)
        predictions = arima_predict_next_30_days(model_fit)
        logging.info(f"ARIMA Predictions for product {product_id}: {predictions}")
        return predictions

    def predict_prophet(self, product_id):
        """
        Predicts sales for the next 30 days using Prophet.
        """
        sales_data = self.fetch_sales_data(product_id)
        if sales_data.empty:
            logging.warning(f"No sales data available for Prophet prediction for product {product_id}.")
            return
        model = train_prophet_model(sales_data)
        predictions = prophet_predict_next_30_days(model)
        logging.info(f"Prophet Predictions for product {product_id}: {predictions}")
        return predictions

    def fetch_sales_data(self, product_id):
        """
        Fetches historical sales data for a given product.
        """
        conn = connect_db()
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

    def predict_stock_requirements(self, product_id):
        """
        Predicts stock requirements based on historical sales data using Linear Regression.
        """
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
        
        predicted_stock = int(np.ceil(future_predictions.sum()))
        logging.info(f"Predicted stock requirements for product {product_id}: {predicted_stock}")
        return predicted_stock
