import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

BANNER = r"""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║  ██████╗ ██╗   ██╗███╗   ███╗██████╗     ██████╗ ████████╗ ██████╗ ║
║  ██╔══██╗██║   ██║████╗ ████║██╔══██╗    ██╔══██╗╚══██╔══╝██╔════╝ ║
║  ██║  ██║██║   ██║██╔████╔██║██████╔╝    ██████╔╝   ██║   ██║      ║
║  ██║  ██║██║   ██║██║╚██╔╝██║██╔═══╝     ██╔══██╗   ██║   ██║      ║
║  ██████╔╝╚██████╔╝██║ ╚═╝ ██║██║         ██████╔╝   ██║   ╚██████╗ ║
║  ╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚═╝         ╚═════╝    ╚═╝    ╚═════╝ ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║  TOOL      : DUMP BTC                                              ║
║  AUTHOR    : Ankon Polley                                                ║
║  INSTAGRAM : itz_vorzhevik_volnyshev                                           ║
╚════════════════════════════════════════════════════════════════════╝
"""

BINANCE_API = "https://api.binance.com/api/v3/klines"

TIMEFRAMES = {
    "1": ("5m", "5 Minutes"),
    "2": ("30m", "30 Minutes"),
    "3": ("1h", "1 Hour"),
    "4": ("4h", "4 Hours"),
    "5": ("1d", "1 Day")
}

WHITE = "\033[97m"
RESET = "\033[0m"

class BTCPredictorPro:

    def __init__(self):
        self.interval = "1h"
        self.df = None
        self.models = None
        self.scaler = StandardScaler()
        self.last_prediction = None
        self.last_signal = None
        self.last_update = None

    def clear(self):
        os.system("cls" if os.name == "nt" else "clear")

    def banner(self):
        self.clear()
        print(WHITE + BANNER + RESET)
        print(WHITE + "BTC Dump v2.0 - Professional Edition" + RESET)
        print(WHITE + "Engine: XGBoost + RandomForest + GradientBoosting" + RESET)
        print(
            WHITE + f"Last Update: {self.last_update or 'Never'} | Interval: {self.interval}" + RESET)
        print(WHITE + "-" * 60 + RESET)

    def fetch_data(self, interval="1h"):  
        try:  
            params = {  
                "symbol": "BTCUSDT",  
                "interval": interval,  
                "limit": 1000  
            }  

            headers = {"User-Agent": "Mozilla/5.0"}  

            response = requests.get(BINANCE_API, params=params, headers=headers, timeout=10)  

            if response.status_code != 200:  
                print("API Error:", response.status_code)  
                return None  

            data = response.json()  

            df = pd.DataFrame(data, columns=[  
                "time", "open", "high", "low", "close", "volume",  
                "ct", "qav", "trades", "tb_base", "tb_quote", "ignore"  
            ])  

            numeric_cols = ["open", "high", "low", "close", "volume"]  

            for col in numeric_cols:  
                df[col] = pd.to_numeric(df[col], errors="coerce")  

            df["time"] = pd.to_datetime(df["time"], unit="ms")  

            df = df.sort_values("time").reset_index(drop=True)  

            self.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  

            return df  

        except requests.exceptions.RequestException as e:  
            print("Network Error:", e)  
            return None  

    def indicators(self, df):  
        df = df.copy()  

        df["ma5"] = df["close"].rolling(5).mean()  
        df["ma20"] = df["close"].rolling(20).mean()  
        df["ma50"] = df["close"].rolling(50).mean()  

        delta = df["close"].diff()  
        gain = delta.where(delta > 0, 0).rolling(14).mean()  
        loss = -delta.where(delta < 0, 0).rolling(14).mean()  

        rs = gain / loss  
        df["RSI"] = 100 - (100 / (1 + rs))  

        exp1 = df["close"].ewm(span=12).mean()  
        exp2 = df["close"].ewm(span=26).mean()  

        df["MACD"] = exp1 - exp2  
        df["MACD_signal"] = df["MACD"].ewm(span=9).mean()  

        df["BB_middle"] = df["close"].rolling(20).mean()  
        bb_std = df["close"].rolling(20).std()  

        df["BB_upper"] = df["BB_middle"] + (bb_std * 2)  
        df["BB_lower"] = df["BB_middle"] - (bb_std * 2)  

        df["volume_sma"] = df["volume"].rolling(20).mean()  
        df["volume_ratio"] = df["volume"] / df["volume_sma"]  

        df = df.dropna()  
        return df  

    def features(self, df):  
        feature_cols = [  
            "close", "volume", "RSI", "MACD",  
            "volume_ratio", "ma5", "ma20",  
            "ma50", "BB_upper", "BB_lower"  
        ]  

        X = []  
        y = []  

        window_size = 20  

        for i in range(window_size, len(df) - 1):  
            window = df[feature_cols].iloc[i-window_size:i].values.flatten()  
            X.append(window)  
            y.append(df["close"].iloc[i+1])  

        return np.array(X), np.array(y)  

    def train(self, X, y):  
        split = int(len(X) * 0.85)  

        X_train, X_test = X[:split], X[split:]  
        y_train, y_test = y[:split], y[split:]  

        X_train_scaled = self.scaler.fit_transform(X_train)  
        X_test_scaled = self.scaler.transform(X_test)  

        xgb = XGBRegressor(  
            n_estimators=200,  
            learning_rate=0.05,  
            max_depth=6,  
            subsample=0.8,  
            random_state=42  
        )  

        rf = RandomForestRegressor(  
            n_estimators=200,  
            max_depth=10,  
            min_samples_split=5,  
            random_state=42  
        )  

        gb = GradientBoostingRegressor(  
            n_estimators=200,  
            learning_rate=0.05,  
            max_depth=5,  
            random_state=42  
        )  

        xgb.fit(X_train_scaled, y_train)  
        rf.fit(X_train_scaled, y_train)  
        gb.fit(X_train_scaled, y_train)  

        pred_xgb = xgb.predict(X_test_scaled)  
        pred_rf = rf.predict(X_test_scaled)  
        pred_gb = gb.predict(X_test_scaled)  

        ensemble_pred = np.mean([pred_xgb, pred_rf, pred_gb], axis=0)  

        error = mean_absolute_percentage_error(y_test, ensemble_pred)  

        return {  
            "xgb": xgb,  
            "rf": rf,  
            "gb": gb,  
            "error": error  
        }  

    def predict(self, models, features):  
        features_scaled = self.scaler.transform(features.reshape(1, -1))  

        p1 = models["xgb"].predict(features_scaled)[0]  
        p2 = models["rf"].predict(features_scaled)[0]  
        p3 = models["gb"].predict(features_scaled)[0]  

        return np.mean([p1, p2, p3])  

    def signal(self, current, prediction, rsi, macd, macd_signal):  
        change_pct = ((prediction - current) / current) * 100  

        if (change_pct > 1.5 and rsi < 70 and macd > macd_signal and current > self.df["ma20"].iloc[-1]):  
            return "STRONG BUY"  

        elif (change_pct > 0.5 and rsi < 65 and macd > macd_signal):  
            return "BUY"  

        elif (change_pct < -1.5 and rsi > 30 and macd < macd_signal and current < self.df["ma20"].iloc[-1]):  
            return "STRONG SELL"  

        elif (change_pct < -0.5 and rsi > 35 and macd < macd_signal):  
            return "SELL"  

        else:  
            return "HOLD"  

    def chart(self):
        self.df = self.fetch_data(self.interval)

        if self.df is None or len(self.df) < 50:
            print("Failed to load chart data")
            return

        self.df = self.indicators(self.df)

        plt.style.use("dark_background")

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 10),
            gridspec_kw={"height_ratios": [3, 1]}
        )

        ax1.plot(self.df["time"], self.df["close"], label="BTC Price", linewidth=2)
        ax1.plot(self.df["time"], self.df["ma20"], label="MA20", linewidth=1)

        current_price = self.df["close"].iloc[-1]

        if self.last_prediction is not None:
            ax1.scatter(
                self.df["time"].iloc[-1],
                self.last_prediction,
                s=120,
                label="Prediction"
            )

            ax1.plot(
                [self.df["time"].iloc[-1], self.df["time"].iloc[-1]],
                [current_price, self.last_prediction],
                linestyle="--",
                label="Move"
            )

        ax1.legend()
        
        ax2.plot(self.df["time"], self.df["RSI"], label="RSI")
        ax2.axhline(70, linestyle="--")
        ax2.axhline(30, linestyle="--")
        ax2.legend()

        plt.tight_layout()
        plt.show()
        plt.close()

    def auto_live(self, delay=60):
        while True:
            try:
                self.banner()
                print("\nAuto Live Mode Running")
                print("-" * 50)

                self.df = self.fetch_data(self.interval)

                if self.df is None or len(self.df) < 100:
                    print("Data error, retrying...")
                    time.sleep(delay)
                    continue

                self.df = self.indicators(self.df)

                X, y = self.features(self.df)

                self.models = self.train(X, y)

                last_features = X[-1]

                self.last_prediction = self.predict(self.models, last_features)

                current_price = self.df["close"].iloc[-1]

                rsi = self.df["RSI"].iloc[-1]
                macd = self.df["MACD"].iloc[-1]
                macd_signal = self.df["MACD_signal"].iloc[-1]

                self.last_signal = self.signal(
                    current_price,
                    self.last_prediction,
                    rsi,
                    macd,
                    macd_signal
                )

                print(f"Time       : {datetime.now().strftime('%H:%M:%S')}")
                print(f"Price      : {current_price}")
                print(f"Prediction : {self.last_prediction}")
                print(f"Signal     : {self.last_signal}")
                print("-" * 50)

                time.sleep(delay)

            except KeyboardInterrupt:
                print("\nStopped Auto Live Mode")
                break

    def main_menu(self):  
        while True:  
            self.banner()  

            print("1. Select Timeframe")  
            print("2. Train & Predict")  
            print("3. Show Live Chart")  
            print("4. Refresh Data Only")  
            print("5. Show Last Prediction")
            print("6. Auto Live Mode")  
            print("0. Exit")  

            choice = input("\nChoice: ").strip()  

            if choice == "0":
                print("\nGoodbye!")  
                break  

            elif choice == "1":
                print("\nAvailable Timeframes:")
                for k, v in TIMEFRAMES.items():
                    print(f"  {k}. {v[1]}")
                
                sel = input("\nSelect timeframe: ").strip()
                if sel in TIMEFRAMES:
                    self.interval = TIMEFRAMES[sel][0]
                    print(f"Timeframe set to {TIMEFRAMES[sel][1]}")
                else:
                    print("Invalid selection")
                input("\n⏎ Press Enter...") 

            elif choice == "2":
                print("\nFetching fresh market data...")
                self.df = self.fetch_data(self.interval)
                
                if self.df is None or len(self.df) < 100:
                    print("Insufficient data for training")
                    input("\n⏎ Press Enter...")
                    continue
                
                print("Computing technical indicators...")
                self.df = self.indicators(self.df)
                
                print("Training our models...")
                X, y = self.features(self.df)
                self.models = self.train(X, y)  

                last_features = X[-1]
                self.last_prediction = self.predict(self.models, last_features)
                current_price = self.df["close"].iloc[-1]
                change_pct = ((self.last_prediction - current_price) / current_price) * 100
                rsi = self.df["RSI"].iloc[-1]
                macd = self.df["MACD"].iloc[-1]
                macd_signal = self.df["MACD_signal"].iloc[-1]
                self.last_signal = self.signal(current_price, self.last_prediction, rsi, macd, macd_signal)
                
                print("\n" + "="*70)
                print("BTC ANALYSIS")
                print("="*70)
                print(f"Current Price:  ${current_price:,.2f}")
                print(f"AI Prediction:  ${self.last_prediction:,.2f}")
                print(f"Change:         {change_pct:+.2f}%")
                print(f"RSI:            {rsi:.1f}")
                print(f"Signal:         {self.last_signal}")
                print(f"Model Error:   {self.models['error']*100:.2f}%")
                print(f"Timeframe:      {self.interval}")
                print(f"Last Update:    {self.last_update}")
                print("="*70)
                
                input("\n⏎ Press Enter to continue...")  

            elif choice == "3":
                 self.chart()

            elif choice == "4":
                print("\nRefreshing data...")
                self.df = self.fetch_data(self.interval)
                if self.df is not None:
                    self.df = self.indicators(self.df)
                    print("Data refreshed successfully!")
                input("\n⏎ Press Enter...")  

            elif choice == "5":
                if self.last_prediction:
                    print(f"\nLast Analysis:")
                    print(f"Prediction: ${self.last_prediction:.2f}")
                    print(f"Signal: {self.last_signal}")
                else:
                    print("No previous analysis")
                input("\n⏎ Press Enter...")
         
            elif choice == "6":
                try:
                    delay = int(input("\nEnter refresh time (seconds): ").strip())
                except:
                    delay = 60

                self.auto_live(delay)

def main():
    app = BTCPredictorPro()
    app.main_menu()

if __name__ == "__main__":
    main()
