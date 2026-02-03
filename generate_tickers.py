"""
This script generates CSV files containing ticker symbols for various market indices.
In this case, it fetches the list of S&P 500 companies from Wikipedia and saves them to 'data/sp500.csv'.
"""

import pandas as pd
import requests
import os

def generate_sp500_csv():
    """
    Fetches the list of S&P 500 companies from Wikipedia and saves it to a CSV file.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        print(f"Fetching S&P 500 constituents from Wikipedia: {url}")

        # Use a User-Agent to mimic a browser and avoid 403 Forbidden errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Let pandas parse the HTML table from the response
        tables = pd.read_html(response.text)
        df = tables[0]
        
        print("Data fetched successfully. Processing...")

        # The ticker column in this table is named 'Symbol'
        if 'Symbol' not in df.columns:
            print("Error: Could not find the 'Symbol' column in the Wikipedia table.")
            return

        # yfinance often uses dashes instead of dots for certain tickers (e.g., 'BRK-B')
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)

        # We only need the ticker symbol column
        tickers_df = df[['Symbol']].copy()
        # Standardize the column name to 'Ticker' for consistency with our main script
        tickers_df.rename(columns={'Symbol': 'Ticker'}, inplace=True)

        # Ensure the 'data' directory exists
        output_dir = 'data'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, "sp500.csv")
        
        # Save the tickers to the CSV file
        tickers_df.to_csv(output_path, index=False)
        
        print(f"Successfully created '{output_path}' with {len(tickers_df)} tickers.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please ensure you have the required libraries: pip install pandas lxml requests")


if __name__ == "__main__":
    generate_sp500_csv()
