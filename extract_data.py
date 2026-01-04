
import os
import pandas as pd
import numpy as np

# Configuration
SYMBOLS = ['TSLA', 'AAPL', 'SNDK', 'RKLB']
DATA_ROOT = '/Volumes/ssd/us_stock_data/1d'
LOOKBACK_DAYS = 90  # Read enough days for indicators

def get_kdj(df, n=9, m1=3, m2=3):
    low_list = df['low'].rolling(window=n, min_periods=1).min()
    high_list = df['high'].rolling(window=n, min_periods=1).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    
    # KDJ smoothing generally uses Wilder's smoothing or EMA. 
    # Standard KDJ uses: K = 2/3*PrevK + 1/3*RSV
    # This is equivalent to EMA with alpha=1/3 -> com=2
    k = rsv.ewm(com=m1-1, adjust=False).mean()
    d = k.ewm(com=m2-1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def get_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = 2 * (dif - dea)
    return dif, dea, macd_hist

def main():
    # 1. Get List of Date Directories
    try:
        all_dirs = sorted([d for d in os.listdir(DATA_ROOT) if d.isdigit() and len(d) == 8])
    except FileNotFoundError:
        print(f"Error: Directory {DATA_ROOT} not found.")
        return

    # Take the last N days
    target_dirs = all_dirs[-LOOKBACK_DAYS:]
    
    data_map = {s: [] for s in SYMBOLS}
    
    print(f"Reading data from {len(target_dirs)} dates...")
    
    for date_str in target_dirs:
        file_path = os.path.join(DATA_ROOT, date_str, f"{date_str}.csv")
        if not os.path.exists(file_path):
            continue
            
        try:
            # Optimize: Read only relevant columns
            # Columns: exchange,symbol,open,high,low,close,amount,volume,bob,eob,type,sequence
            # Pandas read_csv can be slow, but for 90 files it should be okay. 
            # Ideally we'd grep, but mixed types are annoying.
            # Let's filter after reading.
            df = pd.read_csv(file_path, usecols=['symbol', 'open', 'high', 'low', 'close', 'volume'])
            
            # Filter for our symbols
            subset = df[df['symbol'].isin(SYMBOLS)].copy()
            subset['date'] = date_str
            
            for _, row in subset.iterrows():
                data_map[row['symbol']].append(row)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # 2. Process each symbol
    for symbol in SYMBOLS:
        rows = data_map[symbol]
        if not rows:
            print(f"\nNo data found for {symbol}")
            continue
            
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate Indicators
        df['diff'], df['dea'], df['macd'] = get_macd(df)
        df['k'], df['d'], df['j'] = get_kdj(df)
        
        # MA
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        # Output Last 5 rows for context
        print(f"\n=== ANALYSIS DATA FOR {symbol} ===")
        print(df[['date', 'open', 'high', 'low', 'close', 'volume', 'macd', 'diff', 'dea', 'k', 'd', 'j', 'ma5', 'ma10', 'ma20']].tail(5).to_string())
        
        # Determine "Status" strings for the prompt consumption
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # Basic Logic for "Key K-Line" etc. (Just for my internal 'thought' process, the printed dataframe is enough)

if __name__ == "__main__":
    main()
