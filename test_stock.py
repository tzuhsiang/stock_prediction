import yfinance as yf
import requests
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import os
import traceback
import json
import pandas as pd

# Disable SSL verification warnings
urllib3.disable_warnings(InsecureRequestWarning)

# Get and print environment variables
print("Environment variables:")
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    print(f"{key}: {os.environ.get(key)}")

# Get proxy settings from environment variables
proxies = {
    'http': os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy'),
    'https': os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
}

print(f"\nUsing proxy settings: {proxies}")

# Headers that might help
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

# Configure session
session = requests.Session()
if proxies:
    session.proxies = proxies
    session.verify = False
    session.headers.update(headers)
    print("Configured session with proxy settings, headers and disabled SSL verification")

# First, test proxy connection
print("\nTesting proxy connection...")
try:
    response = session.get("http://www.google.com", timeout=10)
    print(f"Proxy test successful: {response.status_code}")
except Exception as e:
    print(f"Proxy test failed: {str(e)}")

# Test symbols
symbols = ['0050.TW', '2330.TW']

for symbol in symbols:
    print(f"\nTesting {symbol}")
    try:
        print(f"Attempting to download data for {symbol}...")
        
        # First try to get raw response
        # Calculate timestamps for 1 month of data
        end = int(pd.Timestamp.now().timestamp())
        start = int((pd.Timestamp.now() - pd.Timedelta(days=30)).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start}&period2={end}&interval=1d"
        print(f"Testing direct API access: {url}")
        response = session.get(url, timeout=30)
        print(f"Raw API response status: {response.status_code}")
        print(f"Raw API response headers: {dict(response.headers)}")
        print(f"First 500 characters of response: {response.text[:500]}")
        
        print("\nAttempting to fix and parse JSON...")
        # Fix the JSON string by adding missing commas
        fixed_json = response.text.replace('""', '","')
        parsed_data = json.loads(fixed_json)
        print("Successfully parsed fixed JSON")
        
        print("\nProcessing chart data...")
        result = parsed_data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        
        # Create pandas DataFrame
        data = pd.DataFrame({
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume']
        }, index=pd.to_datetime([pd.Timestamp(ts, unit='s', tz='Asia/Taipei') for ts in timestamps]))
        print(f"Download completed for {symbol}")
        print(f"Got data for {symbol}:")
        print(f"Number of rows: {len(data)}")
        if not data.empty:
            print("Latest data:")
            print(data.tail(1))
        else:
            print("Warning: Empty dataset received")
    except Exception as e:
        print(f"Error with {symbol}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
