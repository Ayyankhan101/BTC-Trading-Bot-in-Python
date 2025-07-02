import ccxt.async_support as ccxt # type: ignore
import pandas as pd
import numpy as np # type: ignore
import asyncio
import os # For environment variables
import logging

# --- Notification Function ---

async def send_notification(message, send_notifications_enabled: bool):
    """
    Sends a notification. This is a placeholder function.
    In a real bot, this could send messages to Telegram, email, Discord, etc.
    """
    if send_notifications_enabled:
        logging.info(f"NOTIFICATION: {message}")
        # Example for Telegram (requires python-telegram-bot library and bot token/chat ID)
        # try:
        #     from telegram import Bot
        #     from telegram.error import TelegramError
        #     TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        #     TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
        #     if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        #         bot = Bot(token=TELEGRAM_BOT_TOKEN)
        #         await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        #     else:
        #         logging.warning("Telegram bot token or chat ID not set. Skipping Telegram notification.")
        # except ImportError:
        #     logging.warning("python-telegram-bot not installed. Skipping Telegram notification.")
        # except TelegramError as e:
        #     logging.error(f"Telegram notification failed: {e}")
        pass

# --- Helper Functions ---

async def fetch_ohlcv_live(exchange, symbol, timeframe, limit=1):
    """Fetches the latest OHLCV data from the exchange."""
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

async def fetch_ohlcv_backtest(exchange, symbol, timeframe, since, limit):
    """Fetches historical OHLCV data for backtesting."""
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_indicators(df, fast_ma_period, slow_ma_period, rsi_period, long_ma_period):
    """Calculates MA, RSI, and a long-period MA for trend filtering."""
    df['fast_ma'] = df['close'].rolling(window=fast_ma_period).mean()
    df['slow_ma'] = df['close'].rolling(window=slow_ma_period).mean()
    df['long_ma'] = df['close'].rolling(window=long_ma_period).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

async def execute_trade(exchange, symbol, trade_type, amount, price=None, order_type='market'):
    """
    Executes a trade on the exchange. This function is critical for live trading.
    
    WARNINGS:
    1.  **REAL MONEY INVOLVED**: This function will place actual orders on the exchange.
        Ensure your API keys have appropriate permissions and that you understand
        the risks involved with live trading.
    2.  **ERROR HANDLING**: Robust error handling is crucial. Network issues,
        exchange errors (e.g., insufficient balance, invalid order), and rate limits
        must be handled gracefully to prevent unexpected behavior or losses.
    3.  **ORDER TYPES**: This example now supports 'market' and 'limit' orders.
        Limit orders require more complex logic (e.g., order placement, cancellation, and monitoring).
    4.  **FEES & SLIPPAGE**: Account for exchange fees and potential slippage,
        especially with market orders on volatile assets.
    5.  **POSITION MANAGEMENT**: This function only places orders. A complete bot
        needs to track open positions, average entry prices, and manage multiple
        orders (e.g., stop-loss, take-profit orders placed directly on the exchange).
    """
    logging.info(f"--- LIVE TRADE EXECUTION ---")
    logging.info(f"Attempting to place {order_type} {trade_type} order for {amount} {symbol} at price {price if price else 'market'}")

    try:
        order = None
        if order_type == 'market':
            if trade_type == 'buy':
                # Real: order = await exchange.create_market_buy_order(symbol, amount)
                logging.info(f"Simulating MARKET BUY order for {amount} {symbol}.")
                order = {'id': 'simulated_market_buy', 'status': 'closed', 'amount': amount, 'filled': amount, 'price': price, 'type': 'market'}
            elif trade_type == 'sell':
                # Real: order = await exchange.create_market_sell_order(symbol, amount)
                logging.info(f"Simulating MARKET SELL order for {amount} {symbol}.")
                order = {'id': 'simulated_market_sell', 'status': 'closed', 'amount': amount, 'filled': amount, 'price': price, 'type': 'market'}
        elif order_type == 'limit':
            if price is None:
                logging.error("Limit orders require a price.")
                return None
            if trade_type == 'buy':
                # Real: order = await exchange.create_limit_buy_order(symbol, amount, price)
                logging.info(f"Simulating LIMIT BUY order for {amount} {symbol} at {price}.")
                order = {'id': f'simulated_limit_buy_{price}', 'status': 'open', 'amount': amount, 'filled': 0, 'price': price, 'type': 'limit'}
            elif trade_type == 'sell':
                # Real: order = await exchange.create_limit_sell_order(symbol, amount, price)
                logging.info(f"Simulating LIMIT SELL order for {amount} {symbol} at {price}.")
                order = {'id': f'simulated_limit_sell_{price}', 'status': 'open', 'amount': amount, 'filled': 0, 'price': price, 'type': 'limit'}
        
        if order:
            logging.info(f"Order simulated: {order}")
            return order
        else:
            logging.error(f"Order simulation failed for {trade_type} {order_type} order.")
            return None

    except ccxt.NetworkError as e:
        logging.error(f"Network error during trade execution: {e}")
        return None
    except ccxt.ExchangeError as e:
        logging.error(f"Exchange error during trade execution: {e}. Check balance, minimum order size, etc.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during trade execution: {e}")
        return None


# --- Backtesting Function ---

async def backtest_strategy(df, fast_ma_period, slow_ma_period, rsi_period, rsi_oversold, rsi_overbought, stop_loss_percent, take_profit_percent, long_ma_period):
    """
    Backtests the MA Crossover + RSI strategy with Stop-Loss and Take-Profit, and a long MA trend filter.
    """
    df = calculate_indicators(df.copy(), fast_ma_period, slow_ma_period, rsi_period, long_ma_period)
    df['signal'] = 0  # 0: Hold, 1: Buy, -1: Sell

    in_position = False
    buy_price = 0
    trades = []
    
    df.dropna(inplace=True)

    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]

        if in_position:
            stop_loss_price = buy_price * (1 - stop_loss_percent / 100)
            take_profit_price = buy_price * (1 + take_profit_percent / 100)

            if current_price <= stop_loss_price:
                sell_price = current_price
                profit = (sell_price - buy_price) / buy_price * 100
                trades.append({'buy_price': buy_price, 'sell_price': sell_price, 'profit_percent': profit, 'exit_reason': 'Stop-Loss'})
                in_position = False
                logging.info(f"Stop-Loss hit at {df.index[i]}: Price = {sell_price:.2f}, Profit = {profit:.2f}%")
                continue
            elif current_price >= take_profit_price:
                sell_price = current_price
                profit = (sell_price - buy_price) / buy_price * 100
                trades.append({'buy_price': buy_price, 'sell_price': sell_price, 'profit_percent': profit, 'exit_reason': 'Take-Profit'})
                in_position = False
                logging.info(f"Take-Profit hit at {df.index[i]}: Price = {sell_price:.2f}, Profit = {profit:.2f}%")
                continue

        # Trend filter: Only buy if price is above long MA
        if (df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i] and
            df['fast_ma'].iloc[i-1] <= df['slow_ma'].iloc[i-1] and
            df['rsi'].iloc[i] < rsi_overbought and
            current_price > df['long_ma'].iloc[i] and # Trend filter
            not in_position):
            
            df.loc[df.index[i], 'signal'] = 1
            buy_price = current_price
            in_position = True
            logging.info(f"Buy signal at {df.index[i]}: Price = {buy_price:.2f}")

        # Trend filter: Only sell if price is below long MA (for MA crossover exit)
        elif (df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i] and
              df['fast_ma'].iloc[i-1] >= df['slow_ma'].iloc[i-1] and
              current_price < df['long_ma'].iloc[i] and # Trend filter
              in_position):
            
            df.loc[df.index[i], 'signal'] = -1
            sell_price = current_price
            profit = (sell_price - buy_price) / buy_price * 100
            trades.append({'buy_price': buy_price, 'sell_price': sell_price, 'profit_percent': profit, 'exit_reason': 'MA Crossover'})
            in_position = False
            logging.info(f"Sell signal (MA Crossover) at {df.index[i]}: Price = {sell_price:.2f}, Profit = {profit:.2f}%")

    # Calculate additional metrics
    if not trades:
        return [], {}

    profits = [t['profit_percent'] for t in trades]
    
    # Cumulative Profit/Loss
    cumulative_profit = pd.Series(np.cumsum(profits))

    # Maximum Drawdown
    peak = cumulative_profit.cummax()
    drawdown = peak - cumulative_profit
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

    # Average Win/Loss
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p <= 0]
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0

    # Profit Factor (Gross Profit / Gross Loss)
    total_gross_profit = sum(p for p in profits if p > 0)
    total_gross_loss = abs(sum(p for p in profits if p < 0))
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss != 0 else np.inf

    metrics = {
        'max_drawdown': max_drawdown,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }

    return trades, metrics

# --- Live Trading Function ---

async def live_trade(exchange, symbol, timeframe, fast_ma_period, slow_ma_period, rsi_period, rsi_oversold, rsi_overbought, stop_loss_percent, take_profit_percent, long_ma_period, capital, risk_per_trade_percent, send_notifications_enabled: bool):
    """
    Executes the trading strategy in a live environment.
    WARNING: This is a simplified example. Use with extreme caution.
    """
    logging.info(f"--- Starting Live Trading for {symbol} on {timeframe} ---")
    logging.info(f"Strategy Parameters: Fast MA={fast_ma_period}, Slow MA={slow_ma_period}, RSI Period={rsi_period}, Long MA={long_ma_period}")
    logging.info(f"RSI Oversold={rsi_oversold}, RSI Overbought={rsi_overbought}")
    logging.info(f"Stop Loss={stop_loss_percent}%, Take Profit={take_profit_percent}%")
    logging.info(f"Initial Capital: {capital}, Risk per Trade: {risk_per_trade_percent}%")

    # Maintain a history of candles for indicator calculation
    # Fetch enough historical data to calculate initial indicators
    logging.info("Fetching initial historical data for indicator calculation...")
    initial_limit = max(slow_ma_period, rsi_period, long_ma_period) * 2 # Fetch enough for indicators
    ohlcv_history = pd.DataFrame()
    try:
        ohlcv_history = await fetch_ohlcv_backtest(exchange, symbol, timeframe, None, initial_limit)
        if ohlcv_history.empty:
            logging.warning("Could not fetch initial historical data. Exiting live trade.")
            return
        logging.info(f"Fetched {len(ohlcv_history)} initial candles.")
    except Exception as e:
        logging.error(f"Error fetching initial historical data: {e}")
        return

    in_position = False
    buy_price = 0
    open_order = None # To track open limit orders

    while True:
        try:
            # Fetch the latest candle
            latest_candle_df = await fetch_ohlcv_live(exchange, symbol, timeframe)
            if latest_candle_df.empty:
                logging.info("No new candle data. Waiting...")
                await asyncio.sleep(60) # Wait for a minute if no data
                continue

            # Append the latest candle to history and keep only necessary length
            ohlcv_history = pd.concat([ohlcv_history, latest_candle_df]).drop_duplicates().sort_index()
            ohlcv_history = ohlcv_history.iloc[-initial_limit:] # Keep only the last 'initial_limit' candles

            # Calculate indicators on the updated history
            df_with_indicators = calculate_indicators(ohlcv_history.copy(), fast_ma_period, slow_ma_period, rsi_period, long_ma_period)
            
            # Ensure we have enough data for indicator calculation
            if len(df_with_indicators.dropna()) < 2: # Need at least 2 data points for MA/RSI comparison
                logging.info("Not enough data for indicator calculation yet. Waiting...")
                await asyncio.sleep(60)
                continue

            current_data = df_with_indicators.iloc[-1]
            previous_data = df_with_indicators.iloc[-2]

            current_price = current_data['close']
            current_fast_ma = current_data['fast_ma']
            current_slow_ma = current_data['slow_ma']
            current_rsi = current_data['rsi']
            current_long_ma = current_data['long_ma']

            previous_fast_ma = previous_data['fast_ma']
            previous_slow_ma = previous_data['slow_ma']

            logging.info(f"[{current_data.name}] Price: {current_price:.2f}, Fast MA: {current_fast_ma:.2f}, Slow MA: {current_slow_ma:.2f}, Long MA: {current_long_ma:.2f}, RSI: {current_rsi:.2f}")

            # --- Order Management Logic ---
            if open_order:
                # In a real bot, you would fetch the order status from the exchange:
                # order_status = await exchange.fetch_order(open_order['id'], symbol)
                # For simulation, we'll assume it fills if price crosses the limit
                if open_order['type'] == 'limit':
                    if open_order['trade_type'] == 'buy' and current_price <= open_order['price']:
                        open_order['status'] = 'closed'
                        open_order['filled'] = open_order['amount']
                        logging.info(f"Simulated BUY limit order filled at {open_order['price']}")
                    elif open_order['trade_type'] == 'sell' and current_price >= open_order['price']:
                        open_order['status'] = 'closed'
                        open_order['filled'] = open_order['amount']
                        logging.info(f"Simulated SELL limit order filled at {open_order['price']}")

                if open_order['status'] == 'closed':
                    if open_order['trade_type'] == 'buy':
                        in_position = True
                        buy_price = open_order['price'] # Use the filled price
                        logging.info(f"Position opened at {buy_price:.2f}")
                    elif open_order['trade_type'] == 'sell':
                        in_position = False
                        buy_price = 0 # Reset buy price after selling
                        logging.info("Position closed.")
                    open_order = None # Clear the open order
                else:
                    logging.info(f"Order {open_order['id']} is still {open_order['status']}. Waiting for fill...")
                    # If there's an open order, don't place new ones this cycle
                    await asyncio.sleep(60 * int(timeframe[:-1])) # Wait for the next candle interval
                    continue # Skip strategy logic if waiting for order fill

            # Check for exit conditions if in a position
            if in_position:
                stop_loss_price = buy_price * (1 - stop_loss_percent / 100)
                take_profit_price = buy_price * (1 + take_profit_percent / 100)

                if current_price <= stop_loss_price:
                    logging.info(f"[{current_data.name}] STOP LOSS HIT! Selling at {current_price:.2f}")
                    # Calculate amount to sell (should be the amount of the current position)
                    # For simplicity, we'll use the initial trade_amount. In a real bot, track actual position size.
                    amount_to_sell = (capital / buy_price) if buy_price else 0 # Assuming full position exit
                    order_result = await execute_trade(exchange, symbol, 'sell', amount_to_sell, current_price, order_type='market')
                    if order_result and order_result['status'] == 'closed':
                        in_position = False
                        buy_price = 0
                        await send_notification(f"STOP LOSS HIT! Sold {symbol} at {current_price:.2f}")
                elif current_price >= take_profit_price:
                    logging.info(f"[{current_data.name}] TAKE PROFIT HIT! Selling at {current_price:.2f}")
                    # Calculate amount to sell
                    amount_to_sell = (capital / buy_price) if buy_price else 0 # Assuming full position exit
                    order_result = await execute_trade(exchange, symbol, 'sell', amount_to_sell, current_price, order_type='market')
                    if order_result and order_result['status'] == 'closed':
                        in_position = False
                        buy_price = 0
                        await send_notification(f"TAKE PROFIT HIT! Sold {symbol} at {current_price:.2f}")

            # Buy signal with trend filter
            if (current_fast_ma > current_slow_ma and
                previous_fast_ma <= previous_slow_ma and
                current_rsi < rsi_overbought and
                current_price > current_long_ma and # Trend filter
                not in_position):
                
                logging.info(f"[{current_data.name}] BUY SIGNAL! Price: {current_price:.2f}")
                
                # Calculate trade amount based on position sizing
                risk_amount = capital * (risk_per_trade_percent / 100)
                # Assuming stop loss is at a fixed percentage below entry for position sizing
                # In a real scenario, stop loss placement would be dynamic based on market structure
                potential_stop_loss_price = current_price * (1 - stop_loss_percent / 100)
                price_difference = current_price - potential_stop_loss_price

                if price_difference > 0:
                    calculated_trade_amount = risk_amount / price_difference
                    logging.info(f"Calculated trade amount: {calculated_trade_amount:.6f} {symbol}")
                    # Place a limit buy order at the current price
                    open_order = await execute_trade(exchange, symbol, 'buy', calculated_trade_amount, current_price, order_type='limit')
                    if open_order: # If order was successfully placed (simulated)
                        logging.info(f"Placed BUY limit order: {open_order['id']} at {open_order['price']}")
                        await send_notification(f"Placed BUY limit order for {calculated_trade_amount:.6f} {symbol} at {current_price:.2f}")
                    else:
                        logging.error("Failed to place BUY limit order.")
                        await send_notification(f"Failed to place BUY limit order for {symbol}.")
                else:
                    logging.warning("Price difference for stop loss is zero or negative, cannot calculate trade amount. Skipping trade.")
                    await send_notification(f"Skipping BUY signal for {symbol}: Price difference for SL is zero or negative.")

            # Sell signal (MA Crossover, if not already exited by SL/TP) with trend filter
            elif (current_fast_ma < current_slow_ma and
                  previous_fast_ma >= previous_slow_ma and
                  current_price < current_long_ma and # Trend filter
                  in_position):
                
                logging.info(f"[{current_data.name}] SELL SIGNAL (MA Crossover)! Price: {current_price:.2f}")
                # Calculate amount to sell
                amount_to_sell = (capital / buy_price) if buy_price else 0 # Assuming full position exit
                open_order = await execute_trade(exchange, symbol, 'sell', amount_to_sell, current_price, order_type='limit')
                if open_order: # If order was successfully placed (simulated)
                    logging.info(f"Placed SELL limit order: {open_order['id']} at {open_order['price']}")
                    await send_notification(f"Placed SELL limit order for {amount_to_sell:.6f} {symbol} at {current_price:.2f}")
                else:
                    logging.error("Failed to place SELL limit order.")
                    await send_notification(f"Failed to place SELL limit order for {symbol}.")

            # Wait for the next candle interval
            await asyncio.sleep(60 * int(timeframe[:-1])) # Convert '5m' to 300 seconds

        except ccxt.NetworkError as e:
            logging.error(f"Network error: {e}. Retrying in 60 seconds...")
            await send_notification(f"Network error: {e}. Retrying...")
            await asyncio.sleep(60)
        except ccxt.ExchangeError as e:
            logging.error(f"Exchange error: {e}. Check API keys/permissions. Retrying in 60 seconds...")
            await send_notification(f"Exchange error: {e}. Retrying...")
            await asyncio.sleep(60)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}. Retrying in 60 seconds...")
            await send_notification(f"Unexpected error: {e}. Retrying...")
            await asyncio.sleep(60)

# --- Main Execution ---

async def main(send_notifications: bool = False):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    # Set to 'backtest' or 'live'
    MODE = 'live' 

    # Exchange and Symbol
    exchange_id = 'bybit'
    symbol = 'BTC/USDT'
    timeframe = '5m' # 5-minute candles

    # Strategy parameters (tuned for backtesting, may need re-tuning for live)
    fast_ma_period = 30
    slow_ma_period = 60
    long_ma_period = 120 # New parameter for trend filtering
    rsi_period = 14
    rsi_oversold = 40
    rsi_overbought = 60
    stop_loss_percent = 0.8
    take_profit_percent = 5.0

    # Live Trading specific parameters
    # WARNING: Do NOT hardcode API keys in production. Use environment variables.
    # Example: api_key = os.getenv('BYBIT_API_KEY')
    # Example: secret = os.getenv('BYBIT_SECRET')
    api_key = os.getenv('BYBIT_API_KEY')
    secret = os.getenv('BYBIT_SECRET')

    # Position Sizing Parameters
    capital = 1000.0 # Starting capital in USDT
    risk_per_trade_percent = 1.0 # Percentage of capital to risk per trade
    # trade_amount will be calculated dynamically based on position sizing

    # --- Exchange Initialization ---
    exchange_config = {'enableRateLimit': True}
    if MODE == 'live':
        exchange_config['apiKey'] = api_key
        exchange_config['secret'] = secret

    exchange = getattr(ccxt, exchange_id)(exchange_config)

    if MODE == 'backtest':
        # Fetch data for backtesting (e.g., last 30 days)
        since = exchange.parse8601('2024-06-01T00:00:00Z')
        limit = 1000 # Fetch 1000 candles at a time

        all_ohlcv = pd.DataFrame()
        logging.info(f"Fetching historical data for backtesting {symbol} on {timeframe}...")
        while True:
            data = await fetch_ohlcv_backtest(exchange, symbol, timeframe, since, limit)
            if data.empty:
                break
            all_ohlcv = pd.concat([all_ohlcv, data])
            since = data.index[-1].value // 10**6 + 1
            if len(data) < limit:
                break
            await asyncio.sleep(exchange.rateLimit / 1000)

        await exchange.close()

        if all_ohlcv.empty:
            logging.warning("No data fetched for backtesting. Exiting.")
            return

        logging.info(f"Fetched {len(all_ohlcv)} candles for {symbol} on {timeframe} timeframe for backtesting.")

        trades, metrics = await backtest_strategy(all_ohlcv.copy(), fast_ma_period, slow_ma_period, rsi_period, rsi_oversold, rsi_overbought, stop_loss_percent, take_profit_percent, long_ma_period)

        if trades:
            total_profit_percent = sum(trade['profit_percent'] for trade in trades)
            winning_trades = [t for t in trades if t['profit_percent'] > 0]
            losing_trades = [t for t in trades if t['profit_percent'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

            logging.info("\n--- Backtest Results ---")
            logging.info(f"Total Trades: {len(trades)}")
            logging.info(f"Winning Trades: {len(winning_trades)}")
            logging.info(f"Losing Trades: {len(losing_trades)}")
            logging.info(f"Win Rate: {win_rate:.2f}%")
            logging.info(f"Total Profit/Loss: {total_profit_percent:.2f}%")
            
            sl_trades = [t for t in trades if t['exit_reason'] == 'Stop-Loss']
            tp_trades = [t for t in trades if t['exit_reason'] == 'Take-Profit']
            ma_trades = [t for t in trades if t['exit_reason'] == 'MA Crossover']

            logging.info(f"Trades exited by Stop-Loss: {len(sl_trades)}")
            logging.info(f"Trades exited by Take-Profit: {len(tp_trades)}")
            logging.info(f"Trades exited by MA Crossover: {len(ma_trades)}")
            
            logging.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            logging.info(f"Average Winning Trade: {metrics['avg_win']:.2f}%")
            logging.info(f"Average Losing Trade: {metrics['avg_loss']:.2f}%")
            logging.info(f"Profit Factor: {metrics['profit_factor']:.2f}")

        else:
            logging.info("No trades were executed during the backtest period.")

    elif MODE == 'live':
        if api_key is None or secret is None:
            logging.warning("API keys not found in environment variables. Please set BYBIT_API_KEY and BYBIT_SECRET.")
            return

        await live_trade(exchange, symbol, timeframe, fast_ma_period, slow_ma_period, rsi_period, rsi_oversold, rsi_overbought, stop_loss_percent, take_profit_percent, long_ma_period, capital, risk_per_trade_percent, send_notifications)
    
    await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())