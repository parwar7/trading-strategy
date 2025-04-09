import ccxt
import pandas as pd
import ta
import lightgbm as lgb
import logging
from time import sleep

# Logging setup
logging.basicConfig(filename='trading_log.txt', level=logging.INFO)

# Bitget API credentials
api_key = 'your_api_key'
api_secret = 'your_api_secret'
passphrase = 'your_passphrase'

# Initialize exchange
exchange = ccxt.bitget({
    'apiKey': api_key,
    'secret': api_secret,
    'password': passphrase,
})

def fetch_data(symbol='BTC/USDT', timeframe='5m', limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def generate_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    df['boll_z'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    df['returns'] = df['close'].pct_change()
    df.dropna(inplace=True)
    return df

def place_order(symbol, side, amount):
    try:
        if side == 'buy':
            order = exchange.create_market_buy_order(symbol, amount)
        else:
            order = exchange.create_market_sell_order(symbol, amount)
        logging.info(f"{side.upper()} order placed: {amount} {symbol}")
        return order
    except Exception as e:
        logging.error(f"Order Error: {e}")
        return None

# Load trained model
model = lgb.Booster(model_file='model.txt')  # Ensure you upload model.txt too

# Infinite loop to trade live
while True:
    df = fetch_data()
    df = generate_features(df)

    features = ['rsi', 'macd', 'boll_z', 'returns']
    latest_data = df[features].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest_data)

    signal = 'buy' if prediction[0] > 0.5 else 'sell'
    logging.info(f"Prediction: {prediction[0]:.4f}, Signal: {signal}")

    # Example trade amount (adjust to your balance)
    amount = 0.001

    place_order('BTC/USDT', signal, amount)
    sleep(300)  # Wait for 5 minutes
