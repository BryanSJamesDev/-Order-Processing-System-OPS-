# order_processing_system/main.py

import logging
import tkinter as tk
from database import init_db, alter_orders_table, add_initial_products, add_initial_users
from ui import LoginUI
from config import DATABASE_PATH
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='order_system.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def main():
    """
    Initializes the database and launches the UI.
    """
    # Initialize the database and populate initial data
    init_db()
    alter_orders_table()  # Modify the orders table if needed
    add_initial_products()
    add_initial_users()
    
    # Launch the UI
    root = tk.Tk()
    login_app = LoginUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
