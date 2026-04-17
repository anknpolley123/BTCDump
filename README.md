# BTC Dump - Professional Bitcoin Price Prediction Tool

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A professional-grade Bitcoin price prediction tool using ensemble machine learning models (XGBoost, Random Forest, and Gradient Boosting) with technical analysis indicators.

## Features

- **Ensemble ML Models**: Combines XGBoost, Random Forest, and Gradient Boosting for robust predictions
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages
- **Real-time Data**: Fetches live BTC/USDT data from Binance API
- **Multiple Timeframes**: 5min, 30min, 1h, 4h, 1d
- **Live Auto Mode**: Continuous prediction with customizable refresh intervals
- **Visualization**: Interactive price charts with technical indicators
- **Signal Generation**: STRONG BUY/BUY/SELL/STRONG SELL/HOLD signals

## Installation

```bash
# Clone the repository
git clone https://github.com/anknpolley123/BTCDump.git
cd BTCDump

# Install required dependencies
python3 -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

### Basic Usage
Run the tool:

```bash
python3 BTCDump.py
```

### Menu Options

1. **Select Timeframe** - Choose prediction interval (5min to 1day)
2. **Train & Predict** - Fetch data, train models, and get prediction
3. **Show Live Chart** - Display price chart with technical indicators
4. **Refresh Data Only** - Update market data without retraining
5. **Show Last Prediction** - Display previous analysis results
6. **Auto Live Mode** - Continuous predictions with custom refresh interval

## Technical Analysis Indicators

The tool calculates the following technical indicators:

- **Moving Averages**: MA5, MA20, MA50
- **Relative Strength Index (RSI)**: 14-period RSI
- **MACD**: 12/26 EMA with 9-period signal line
- **Bollinger Bands**: 20-period SMA with 2 standard deviations
- **Volume Analysis**: Volume SMA and ratio

## Machine Learning Models

The tool uses a weighted average of three models:

1. **XGBoost Regressor**
   - n_estimators: 200
   - learning_rate: 0.05
   - max_depth: 6

2. **Random Forest Regressor**
   - n_estimators: 200
   - max_depth: 10
   - min_samples_split: 5

3. **Gradient Boosting Regressor**
   - n_estimators: 200
   - learning_rate: 0.05
   - max_depth: 5

### Features Used for Prediction

- Current price and volume
- RSI, MACD values
- Moving averages (MA5, MA20, MA50)
- Bollinger Band positions
- Volume ratios

## Signal Generation Logic

| Condition | Signal |
|-----------|--------|
| Change > 1.5% AND RSI < 70 AND MACD > Signal AND Price > MA20 | STRONG BUY |
| Change > 0.5% AND RSI < 65 AND MACD > Signal | BUY |
| Change < -1.5% AND RSI > 30 AND MACD < Signal AND Price < MA20 | STRONG SELL |
| Change < -0.5% AND RSI > 35 AND MACD < Signal | SELL |
| Otherwise | HOLD |

## Configuration

### Binance API

The tool uses Binance's public API endpoint:
```
https://api.binance.com/api/v3/klines
```

No API key required for public market data.

### Timeframe Options

| Option | Timeframe | Description |
|--------|-----------|-------------|
| 1 | 5m | 5 Minutes |
| 2 | 30m | 30 Minutes |
| 3 | 1h | 1 Hour |
| 4 | 4h | 4 Hours |
| 5 | 1d | 1 Day |


## Output Example

```
╔════════════════════════════════════════════════════════════════════╗
║  TOOL      : DUMP BTC                                              ║
║  AUTHOR    : Ankon Polley                                                ║
║  INSTAGRAM : itz_vorzhevik_volnyshev                                           ║
╚════════════════════════════════════════════════════════════════════╝

BTC ANALYSIS
======================================================================
Current Price:  $43,250.00
AI Prediction:  $44,100.00
Change:         +1.97%
RSI:            58.3
Signal:         STRONG BUY
Model Error:    2.34%
Timeframe:      1h
Last Update:    2026-03-19 14:30:00
======================================================================
```

## Architecture

```
BTCPredictorPro
├── fetch_data()      → Get BTC/USDT data from Binance
├── indicators()      → Calculate technical indicators
├── features()        → Create feature matrix
├── train()           → Train ensemble models
├── predict()         → Generate price prediction
├── signal()          → Generate trading signal
├── chart()           → Display price chart
├── auto_live()       → Continuous predictions
└── main_menu()       → User interface
```

## Performance Notes

- **Model Error**: Typically 1-3% MAPE on test data
- **Training Time**: ~10-30 seconds per model
- **Prediction Speed**: <1 second per prediction
- **Data Requirements**: Minimum 100 candles for training

## Disclaimer

⚠️ **This tool is for educational and research purposes only.**

- Cryptocurrency trading involves significant risk
- Past performance does not guarantee future results
- Do not use this tool for actual trading decisions
- The author assumes no liability for financial losses

## Author

- **GitHub**: [@anknpolley123](https://github.com/anknpolley123)
- **Instagram**: [@itz_vorzhevik_volnyshev](https://www.instagram.com/itz_vorzhevik_volnyshev?igsh=OTNqMndzdDFzZ293)

## Acknowledgments

- Binance API for real-time market data
- Scikit-learn, XGBoost, and Pandas communities
- Technical analysis principles from traditional finance

