# order_processing_system/blockchain.py

import hashlib
import time
import logging

class Block:
    """Represents a single block in the blockchain."""
    def __init__(self, index, timestamp, data, previous_hash=''):
        self.index = index
        self.timestamp = timestamp
        self.data = data  # Order details
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """
        Calculates the SHA-256 hash of the block's contents.
        """
        block_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}".encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    """Manages the chain of blocks."""
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        """
        Creates the genesis (first) block in the blockchain.
        """
        return Block(0, time.time(), "Genesis Block", "0")

    def get_latest_block(self):
        """
        Retrieves the latest block in the blockchain.
        """
        return self.chain[-1]

    def add_block(self, new_block):
        """
        Adds a new block to the blockchain after setting its previous hash.
        """
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)
        logging.info(f"Block {new_block.index} added to blockchain.")

    def is_chain_valid(self):
        """
        Validates the integrity of the blockchain.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                logging.error(f"Block {current_block.index} has invalid hash.")
                return False
            if current_block.previous_hash != previous_block.hash:
                logging.error(f"Block {current_block.index} has invalid previous hash.")
                return False
        logging.info("Blockchain integrity verified.")
        return True
