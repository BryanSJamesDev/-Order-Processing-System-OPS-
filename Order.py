import tkinter as tk

class OrderProcessor:
    def __init__(self, ui):
        self.order = None
        self.ui = ui  

    def receive_order(self, order):
        self.order = order

    def is_order_valid(self):
        if self.order and all(key in self.order for key in ["order_id", "customer_name", "product_id", "quantity"]):
            return True
        return False

    def check_inventory(self):
        ordered_product_id = self.order.get("product_id")
        ordered_quantity = int(self.order.get("quantity"))
        available_stock = 5

        if ordered_product_id == "ABC123" and ordered_quantity <= available_stock:
            return True
        else:
            return False

    def prepare_sales_invoice(self):
        print("Sales invoice generated for order:", self.order)

    def prepare_purchase_requisition(self):
        print("Purchase requisition generated for order:", self.order)

    def convert_pr_to_po(self):
        print("Purchase order created for order:", self.order)

    def receive_shipment(self):
        print("Shipment received for order:", self.order)

    def prepare_grn(self):
        print("Goods receipt note prepared for order:", self.order)

    def transfer_to_shop(self):
        print("Product transferred to shop for order:", self.order)

    def process_order(self):
        if self.is_order_valid():
            if self.check_inventory():
                self.prepare_sales_invoice()
                self.ui.message_label.config(text="Order processed successfully. Sales invoice generated.")
            else:
                self.prepare_purchase_requisition()
                self.convert_pr_to_po()
                self.receive_shipment()
                self.prepare_grn()
                self.transfer_to_shop()
                self.ui.message_label.config(text="Insufficient inventory. Purchase requisition created and converted to a purchase order.")
        else:
            self.ui.message_label.config(text="Invalid order. Please check the order details.")

class OrderProcessingUI:
    def __init__(self, master):
        self.master = master
        self.order_processor = OrderProcessor(self)

        self.order_id_label = tk.Label(master, text="Order ID:")
        self.order_id_label.pack()

        self.order_id_entry = tk.Entry(master)
        self.order_id_entry.pack()

        self.customer_name_label = tk.Label(master, text="Customer Name:")
        self.customer_name_label.pack()

        self.customer_name_entry = tk.Entry(master)
        self.customer_name_entry.pack()

        self.product_id_label = tk.Label(master, text="Product ID:")
        self.product_id_label.pack()

        self.product_id_entry = tk.Entry(master)
        self.product_id_entry.pack()

        self.quantity_label = tk.Label(master, text="Quantity:")
        self.quantity_label.pack()

        self.quantity_entry = tk.Entry(master)
        self.quantity_entry.pack()

        # Additional Variable Names and Input Boxes
        additional_variables = [
            "Credit Limit", "Order Type", "Date", "Customer", 
            "Credit Balance", "Order Number", "Order Type", 
            "Local Ref #", "Pending DN Amt", "Project No", 
            "Status", "Available Balance", "Delivery Notes", 
            "Quotations", "Ref", "Loc", "Date", "Total", 
            "Discount", "Pending", "Customer", "PO/Ref", 
            "Add. Info", "Customer Price Records", "Cust Price", 
            "Return", "Order", "Number", "Status", "U Price", 
            "UoM", "DN.Line", "Return Order Number", "Item Ref",
            "Item Code", "OH Qty", "DN.Qty", "UoM", "Inv.Qty",
            "Rtn.Qty", "Qty", "Cust Price", "U Price", "Total Price",
            "Serial Num", "Status"
        ]

        for i in range(0, len(additional_variables), 8):
            row_frame = tk.Frame(master)
            row_frame.pack()

            for j in range(8):
                index = i + j
                if index < len(additional_variables):
                    variable_name = additional_variables[index]
                    label = tk.Label(row_frame, text=variable_name)
                    label.grid(row=0, column=j, padx=5)

            for j in range(1, 4):
                for k in range(8):
                    index = i + k
                    if index < len(additional_variables):
                        entry = tk.Entry(row_frame)
                        entry.grid(row=j, column=k, padx=5)

        # Submit and Cancel Buttons
        self.submit_button = tk.Button(master, text="Submit Order", command=self.submit_order)
        self.submit_button.pack()

        self.cancel_button = tk.Button(master, text="Cancel Order", command=self.cancel_order)
        self.cancel_button.pack()

        # Message Label
        self.message_label = tk.Label(master, text="")
        self.message_label.pack()

    def submit_order(self):
        order_id = self.order_id_entry.get()
        customer_name = self.customer_name_entry.get()
        product_id = self.product_id_entry.get()
        quantity = self.quantity_entry.get()

        order = {
            "order_id": order_id,
            "customer_name": customer_name,
            "product_id": product_id,
            "quantity": quantity
        }

        self.order_processor.receive_order(order)
        self.order_processor.process_order()

    def cancel_order(self):
        self.order_id_entry.delete(0, tk.END)
        self.customer_name_entry.delete(0, tk.END)
        self.product_id_entry.delete(0, tk.END)
        self.quantity_entry.delete(0, tk.END)

        self.order_processor.order = None

root = tk.Tk()
root.title("Order Processing System")

app = OrderProcessingUI(root)

root.mainloop()
