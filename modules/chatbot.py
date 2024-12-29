# order_processing_system/chatbot.py

import logging
import openai
from config import OPENAI_API_KEY
import tkinter as tk
from tkinter import ttk

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

def chatbot_response(prompt):
    """
    Generates a response from the chatbot based on the user prompt.
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-003",  # Use the desired OpenAI model
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error with OpenAI API: {e}")
        return "There was an issue connecting to the chatbot service. Please try again."

class ChatbotTab:
    """UI component for interacting with the chatbot."""
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
