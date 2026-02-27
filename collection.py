import pandas as pd
import yfinance as yf
import time
import io
import os
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium_stealth import stealth
from webdriver_manager.chrome import ChromeDriverManager

def clean_currency(value):
    """Helper to convert currency strings to floats safely."""
    if pd.isna(value) or value == '-': 
        return 0.0
    
    clean_val = re.sub(r'[^\d.]', '', str(value))
    try:
        return float(clean_val) if clean_val else 0.0
    except:
        return 0.0

def derive_ticker(name):
    """Safely extracts a possible NSE ticker from a Company Name string."""
   
    if pd.isna(name) or not isinstance(name, str):
        return None
    
    
    clean = re.sub(r'Limited|Ltd|LTD|LIMITED|\(.*\)', '', name, flags=re.IGNORECASE).strip()
    
  
    words = clean.split()
    if words:
        return f"{words[0].upper()}.NS"
    return None

def get_collection():
    url = "https://www.chittorgarh.com/report/ipo-performance-report-listing-current-gain/125/all/?year=2026"
    
    options = Options()
    options.add_argument("--headless") 
    options.add_argument("--disable-gpu")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )

    try:
        print("Bypassing protection and fetching 2026 IPO data...")
        driver.get(url)
        time.sleep(10) 
        
        html_content = driver.page_source
        df_list = pd.read_html(io.StringIO(html_content))
        
        if not df_list:
            print(" No tables found.")
            return None

      
        df = max(df_list, key=len)
        
       
        df.columns = [str(col).replace('▲▼', '').strip() for col in df.columns]

       
        df = df.dropna(subset=['Company'])

        
        symbol_col = next((c for c in ['Symbol', 'NSE Symbol', 'Ticker'] if c in df.columns), None)
        
        if symbol_col:
            df['YF_Ticker'] = df[symbol_col].apply(lambda x: f"{str(x).strip()}.NS" if pd.notnull(x) else None)
        elif 'Company' in df.columns:
            print("🔍 Deriving Tickers from Company names safely...")
            df['YF_Ticker'] = df['Company'].apply(derive_ticker)
            
        print(f" Captured {len(df)} IPO entries.")
        return df

    except Exception as e:
        print(f" Scraping Error: {e}")
        return None
    finally:
        driver.quit()

def live_prices(df):
    if df is None or 'YF_Ticker' not in df.columns:
        return df

    print("Fetching live prices ")
    
    
    price_cols = [c for c in df.columns if 'Price' in c or 'Rs' in c]
    for col in price_cols:
        df[col] = df[col].apply(clean_currency)


    valid_tickers_df = df[df['YF_Ticker'].notna()].head(10) 
    
    for idx, row in valid_tickers_df.iterrows():
        ticker = row['YF_Ticker']
        try:
            stock = yf.Ticker(ticker)
            price = stock.fast_info['last_price']
            if price and price > 0:
                df.at[idx, 'Live_Price'] = round(price, 2)
                print(f"    {ticker}: ₹{round(price, 2)}")
        except:
            continue
            
    return df 

if __name__=="__main__":
    os.makedirs("data_processed", exist_ok=True)
    raw_data = get_collection()
    
    if raw_data is not None:
        final_data = live_prices(raw_data)
        
        
        issue_col = 'Issue Price (Rs.)'
        listing_col = 'Close Price on Listing (Rs.)'

        if issue_col in final_data.columns and listing_col in final_data.columns:
            
            final_data = final_data[final_data[issue_col] > 0]
            final_data['Listing_Gain_Pct'] = ((final_data[listing_col] - final_data[issue_col]) / final_data[issue_col]) * 100

        output_path = "data_processed/ipo_tracker_cleaned.csv"
        final_data.to_csv(output_path, index=False)
        print(f" Phase 1 & Preprocessing Complete. Saved {len(final_data)} rows to {output_path}")
    else:
        print(" Critical Error: No data collected.")