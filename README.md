# BTC Trading Bot

This is a Python-based trading bot for BTC/USDT on the Bybit exchange. It uses a trading strategy based on Moving Average (MA) crossovers, the Relative Strength Index (RSI), and a long-period MA for trend filtering.

**Disclaimer:** This is a simplified example and should not be used for live trading without further development and testing. Trading cryptocurrencies involves significant risk.

## Features

- **Trading Strategy:**
    - **Entry:** Buys when the fast MA crosses above the slow MA, the RSI is not overbought, and the price is above the long MA (indicating an uptrend).
    - **Exit:** Sells when the fast MA crosses below the slow MA, or when a stop-loss or take-profit level is hit.
- **Backtesting:** The bot can be run in backtesting mode to evaluate the strategy's performance on historical data.
- **Live Trading:** The bot can be run in live trading mode to execute trades on the Bybit exchange. **(Use with extreme caution!)**
- **Notifications:** The bot can send notifications for significant events (e.g., trades, errors). (This is a placeholder and needs to be configured).

## Getting Started

### Prerequisites

- Python 3.7+
- The required Python libraries, listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd btc_trading_bot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API keys:**

    -   **DO NOT hardcode your API keys in the script.**
    -   Create a file named `.env` in the project directory.
    -   Add your Bybit API key and secret to the `.env` file:
        ```
        BYBIT_API_KEY="YOUR_API_KEY"
        BYBIT_SECRET="YOUR_SECRET_KEY"
        ```
    -   The bot will load these keys from the environment variables.

### Usage

1.  **Configure the bot:**

    -   Open `trading_bot.py` and adjust the parameters in the `main` function:
        -   `MODE`: Set to `'backtest'` or `'live'`.
        -   `symbol`: The trading pair (e.g., `'BTC/USDT'`).
        -   `timeframe`: The candle timeframe (e.g., `'5m'`, `'1h'`).
        -   Strategy parameters: `fast_ma_period`, `slow_ma_period`, `rsi_period`, etc.

2.  **Run the bot:**
    ```bash
    python trading_bot.py
    ```

## How it Works

The bot uses the `ccxt` library to interact with the Bybit exchange. It fetches OHLCV (Open, High, Low, Close, Volume) data, calculates technical indicators, and executes trades based on the defined strategy.

### Backtesting

In backtesting mode, the bot fetches historical data and simulates trades. It then prints a summary of the results, including:

-   Total trades
-   Win rate
-   Total profit/loss
-   Maximum drawdown

### Live Trading

In live trading mode, the bot connects to your Bybit account and executes trades in real-time. **Be aware of the risks involved and start with a small amount of capital.**

## Disclaimer

This trading bot is for educational purposes only. The author is not responsible for any financial losses you may incur. Always do your own research and trade responsibly.
