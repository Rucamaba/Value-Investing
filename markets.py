"""
This script provides utility functions to read ticker symbols from CSV files
for various market indices.
"""

import pandas as pd
import os

def get_tickers_from_csv(file_path):
    """
    Reads tickers and a blacklist from a CSV file.

    The CSV file should contain ticker symbols in the first column.
    The blacklist begins after a line that starts with "Blacklist".

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing two lists: (tickers, blacklist).
               Returns ([], []) if the file cannot be read.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Market file not found at '{file_path}'. Skipping.")
        return [], []

    tickers = []
    blacklist = []
    is_blacklist_section = False

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith('blacklist'):
                    is_blacklist_section = True
                    continue
                
                if is_blacklist_section:
                    blacklist.append(line.split(',')[0])
                else:
                    tickers.append(line.split(',')[0])
        return tickers, blacklist
    except Exception as e:
        print(f"Error reading CSV file '{file_path}': {e}")
        return [], []
