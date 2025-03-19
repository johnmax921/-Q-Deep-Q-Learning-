import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import yfinance as yf
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import time
import datetime
import random
import math
import warnings
import matplotlib as mpl

# 設置matplotlib中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
plt.rcParams['font.size'] = 12  # 設置默認字體大小

# 檢查可用的中文字體
def check_chinese_fonts():
    print("系統中可用的中文字體:")
    chinese_fonts = []
    for f in fm.fontManager.ttflist:
        if any(keyword in f.name for keyword in ['微軟', 'Microsoft', '宋體', 'SimSun', 'SimHei', '黑體']):
            chinese_fonts.append(f.name)
            print(f"  - {f.name}")
    return chinese_fonts

# 暫時註釋掉 reverb 相關的導入
# try:
#     import reverb
# except ImportError:
#     print("reverb 庫未安裝，某些功能可能無法使用")

# 檢查 TensorFlow 版本並適應不同版本的 API
tf_version = tf.__version__
print(f"TensorFlow 版本: {tf_version}")

# 避免 TensorFlow 2.18+ 中的 Keras 相容性問題
if tf_version >= "2.15.0":
    # 對於較新版本的 TensorFlow，使用獨立的 Keras 套件
    import keras
    print(f"使用獨立的 Keras 版本: {keras.__version__}")
else:
    # 對於較舊版本的 TensorFlow，使用內建的 Keras
    from tensorflow import keras
    print(f"使用 TensorFlow 內建的 Keras")

from tf_agents.environments import tf_py_environment
from tf_agents.utils import common
# 暫時註釋掉 reverb 相關的導入
# from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.policies import py_tf_eager_policy, random_tf_policy, policy_saver
from tf_agents.train.utils import strategy_utils
from tf_agents.drivers import dynamic_step_driver
# import reverb
from tqdm import tqdm

# 台灣股市特定常數
TRADING_DAYS_YEAR = 252  # 台灣股市一年約252個交易日
TRADE_COSTS_PERCENT = 0.003  # 台灣股市交易稅0.3%
RISK_FREE_RATE = 0.015  # 假設無風險利率為1.5%

# 交易環境常數
CAPITAL = 1000000  # 初始資金100萬
STATE_LEN = 30  # 觀察窗口長度
DISCOUNT = 0.99  # 折扣因子
REWARD_CLIP = 1.0  # 獎勵裁剪範圍

# 動作空間
ACT_HOLD = 0
ACT_LONG = 1
ACT_SHORT = -1  # 修改為負數，表示做空

# 資料相關設定
DATA_DIR = "./data"
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
SPLIT_DATE = "2022-01-01"  # 訓練和測試數據的分割日期
VALIDATION_ITERS = 5  # 驗證迭代次數

# 台灣市場相關指數
TARGET = "2330.TW"  # 台積電作為目標股票
MARKET_INDEX = "^TWII"  # 台灣加權指數
VOLATILITY_INDEX = "^VIX"  # 仍使用VIX作為波動指標
RATES_INDEX = "^TNX"  # 10年期美國國債收益率，可替換為台灣相關利率
ELECTRONICS_INDEX = "0050.TW"  # 台灣50ETF，代表電子業
USD_TWD = "TWD=X"  # 美元兌台幣匯率

# 特徵設定
MACRO_FEATURES = [MARKET_INDEX, VOLATILITY_INDEX, RATES_INDEX, ELECTRONICS_INDEX, USD_TWD]
TA_FEATURES = ['MACD', 'MACD_HIST', 'MACD_SIG', 'ATR', 'EMA_SHORT', 'EMA_MID', 'EMA_LONG']
HLOC_FEATURES = ["Close", "High", "Low", "Open", "Volume"]
FEATURES = ['Price Returns', 'Price Delta', 'Close Position', 'Volume']
TARGET_FEATURE = "Close"  # 修改為使用 Close 作為目標特徵

# 訓練設定
TRAIN_EPISODES = 1000

def get_tickerdata(tickers_symbols, start=START_DATE, end=END_DATE, interval="1d", data_dir=DATA_DIR):
    """
    獲取台灣股市的股票數據，並進行基本的數據處理
    """
    tickers = {}
    earliest_end = datetime.strptime(end, '%Y-%m-%d')
    latest_start = datetime.strptime(start, '%Y-%m-%d')
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"嘗試獲取以下股票/指數數據: {tickers_symbols}")
    
    for symbol in tickers_symbols:
        cached_file_path = f"{data_dir}/{symbol.replace('^', '').replace('.', '_')}-{start}-{end}-{interval}.parquet"
        csv_file_path = cached_file_path.replace('.parquet', '.csv')
        
        try:
            # 首先嘗試從CSV讀取（因為我們現在保存為CSV）
            if os.path.exists(csv_file_path):
                print(f"從CSV快取讀取 {symbol} 數據")
                df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
                assert len(df) > 0
            # 然後嘗試從parquet讀取（向後兼容）
            elif os.path.exists(cached_file_path):
                print(f"從Parquet快取讀取 {symbol} 數據")
                df = pd.read_parquet(cached_file_path)
                df.index = pd.to_datetime(df.index)
                assert len(df) > 0
            else:
                print(f"從 Yahoo Finance 下載 {symbol} 數據")
                # 使用yfinance下載台灣股票數據
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    progress=False,
                    interval=interval,
                )
                print(f"獲取到 {symbol} 數據，長度: {len(df)}")
                assert len(df) > 0
                
                # 保存為 CSV
                df.to_csv(csv_file_path)
                print(f"已保存 {symbol} 數據到 {csv_file_path}")
            
            # 基本數據統計（使用安全的方式計算）
            try:
                min_date = df.index.min()
                max_date = df.index.max()
                nan_count = df["Close"].isnull().sum()
                
                # 安全計算偏度和峰度
                close_values = df["Close"].dropna().values
                skewness = float(skew(close_values))
                skewness = round(float(skewness), 2)
                kurt = float(kurtosis(close_values))
                kurt = round(float(kurt), 2)
                
                outliers_count = (df["Close"] > df["Close"].mean() + (3 * df["Close"].std())).sum()
                
                print(
                    f"{symbol} => min_date: {min_date}, max_date: {max_date}, "
                    f"kurt:{kurt}, skewness:{skewness}, outliers_count:{outliers_count}, "
                    f"nan_count: {nan_count}"
                )
            except Exception as e:
                print(f"計算 {symbol} 統計數據時出錯: {e}，但將繼續使用數據")
            
            # 無論統計計算是否成功，都添加數據到tickers字典
            tickers[symbol] = df
            
            if min_date > latest_start:
                latest_start = min_date
            if max_date < earliest_end:
                earliest_end = max_date
                
        except Exception as e:
            print(f"獲取 {symbol} 數據時出錯: {e}")
            import traceback
            traceback.print_exc()
    
    if not tickers:
        raise ValueError("無法獲取任何股票數據，請檢查網絡連接或股票代碼")
    
    return tickers, latest_start, earliest_end

# 定義要獲取的股票和指數
TICKER_SYMBOLS = [TARGET, MARKET_INDEX, VOLATILITY_INDEX, RATES_INDEX, ELECTRONICS_INDEX, USD_TWD]

class TaiwanStockTradingEnv(py_environment.PyEnvironment):
    """
    台灣股市交易環境，考慮了台灣股市的特點：
    - 漲跌幅限制(±10%)
    - 交易稅(0.3%)
    - 台灣特定的宏觀經濟指標
    """
    def __init__(self, data, features=FEATURES, money=CAPITAL, state_length=STATE_LEN, 
                 transaction_cost=TRADE_COSTS_PERCENT, market_costs=0.001, 
                 reward_discount=DISCOUNT, price_limit=0.1):
        """
        初始化交易環境
        """
        super(TaiwanStockTradingEnv, self).__init__()
        assert data is not None
        
        self.data_dim = len(features)
        self.features = features
        self.state_length = state_length
        self.current_step = self.state_length
        self.reward_discount = reward_discount
        self.balance = money
        self.initial_balance = money
        self.transaction_cost = transaction_cost
        self.price_limit = price_limit  # 台灣股市漲跌幅限制(±10%)
        self.epsilon = max(market_costs, np.finfo(float).eps)  # 市場波動成本
        self.total_shares = 0
        self._episode_ended = False
        
        # 定義動作和觀察空間
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=ACT_SHORT, maximum=ACT_LONG, name='action')
        
        # 修改觀察空間的定義，確保形狀是固定的
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.state_length * len(self.features),), 
            dtype=np.float32, 
            minimum=-10.0,  # 設定合理的最小值
            maximum=10.0,   # 設定合理的最大值
            name='observation')
        
        self.data = self.preprocess_data(data.copy())
        self.reset()
    
    @property
    def batched(self):
        return False
    
    @property
    def batch_size(self):
        return None
    
    def preprocess_data(self, df):
        """
        預處理數據，包括添加特徵、標準化和填充缺失值
        """
        print("預處理數據...")
        print(f"原始數據列: {df.columns.tolist()}")
        print(f"原始數據形狀: {df.shape}")
        
        # 處理 MultiIndex DataFrame
        # 將 MultiIndex 轉換為單一索引
        if isinstance(df.columns, pd.MultiIndex):
            print("檢測到 MultiIndex DataFrame，進行轉換...")
            # 創建新的列名
            new_columns = []
            for col in df.columns:
                if col[1]:  # 如果第二級不為空
                    new_columns.append(f"{col[0]}_{col[1]}")
                else:
                    new_columns.append(col[0])
            
            # 創建新的 DataFrame
            new_df = pd.DataFrame(df.values, index=df.index, columns=new_columns)
            df = new_df
            print(f"轉換後的列: {df.columns.tolist()}")
        
        # 確保所有必要的列都存在
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            matching_cols = [c for c in df.columns if c.startswith(col)]
            if not matching_cols:
                raise ValueError(f"數據中缺少必要的列: {col}")
            if len(matching_cols) > 1:
                # 如果有多個匹配的列，使用第一個
                print(f"發現多個 {col} 列: {matching_cols}，使用 {matching_cols[0]}")
            
            # 如果列名不完全匹配，重命名為標準名稱
            if matching_cols[0] != col:
                df[col] = df[matching_cols[0]]
        
        # 添加基本特徵
        df['Price Returns'] = df['Close'].pct_change().fillna(0)
        df['Price Delta'] = df['High'] - df['Low']
        
        # 安全地計算 Close Position
        df['Close Position'] = 0.5  # 默認值
        mask = df['Price Delta'] != 0
        if mask.any():
            df.loc[mask, 'Close Position'] = (df.loc[mask, 'Close'] - df.loc[mask, 'Low']).abs() / df.loc[mask, 'Price Delta']
        
        # 應用漲跌幅限制
        df['Price Returns'] = df['Price Returns'].clip(-self.price_limit, self.price_limit)
        
        # 標準化特徵
        for col in self.features:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if isinstance(col_min, (int, float)) and isinstance(col_max, (int, float)) and col_min != col_max:
                    df[col] = (df[col] - col_min) / (col_max - col_min)
                else:
                    df[col] = 0.
        
        # 填充缺失值
        df = df.ffill().bfill()
        
        # 添加交易相關列
        df['Position'] = 0
        df['Action'] = ACT_HOLD
        df['Cash'] = self.initial_balance
        df['Holdings'] = 0.
        df['Money'] = df['Cash']
        df['Returns'] = 0.
        df['Reward'] = 0.
        df['Sharpe'] = 0.
        
        print(f"處理後數據列: {df.columns.tolist()}")
        print(f"處理後數據形狀: {df.shape}")
        
        return df
    
    def action_spec(self):
        """提供動作空間的規格"""
        return self._action_spec
    
    def observation_spec(self):
        """提供觀察空間的規格"""
        return self._observation_spec
    
    def _reset(self):
        """重置環境狀態，準備新的交易回合"""
        self.balance = self.initial_balance
        self.current_step = self.state_length
        self._episode_ended = False
        self.total_shares = 0
        
        self.data['Reward'] = 0.
        self.data['Sharpe'] = 0.
        self.data['Position'] = 0
        self.data['Action'] = ACT_HOLD
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(self.balance)
        self.data['Money'] = self.data.iloc[0]['Holdings'] + self.data.iloc[0]['Cash']
        self.data['Returns'] = 0.
        
        return ts.restart(self._next_observation())
    
    def _next_observation(self):
        """根據當前步驟和歷史長度生成下一個觀察"""
        start_idx = max(0, self.current_step - self.state_length + 1)
        end_idx = self.current_step + 1
        
        # 確保我們有足夠的數據
        if end_idx - start_idx < self.state_length:
            # 如果數據不足，用第一個數據填充
            pad_size = self.state_length - (end_idx - start_idx)
            obs_values = []
            
            # 填充缺少的數據
            for _ in range(pad_size):
                for feature in self.features:
                    if feature in self.data.columns:
                        obs_values.append(self.data[feature].iloc[start_idx])
                    else:
                        obs_values.append(0.0)  # 如果特徵不存在，用0填充
            
            # 添加實際數據
            for i in range(start_idx, end_idx):
                for feature in self.features:
                    if feature in self.data.columns:
                        obs_values.append(self.data[feature].iloc[i])
                    else:
                        obs_values.append(0.0)  # 如果特徵不存在，用0填充
        else:
            # 如果有足夠的數據
            obs_values = []
            for i in range(start_idx, end_idx):
                for feature in self.features:
                    if feature in self.data.columns:
                        obs_values.append(self.data[feature].iloc[i])
                    else:
                        obs_values.append(0.0)  # 如果特徵不存在，用0填充
        
        # 確保觀察向量的長度正確
        expected_length = self.state_length * len(self.features)
        if len(obs_values) < expected_length:
            # 如果長度不足，填充0
            obs_values.extend([0.0] * (expected_length - len(obs_values)))
        elif len(obs_values) > expected_length:
            # 如果長度過長，截斷
            obs_values = obs_values[:expected_length]
        
        return np.array(obs_values, dtype=np.float32)
    
    def _step(self, action):
        """
        執行交易動作，更新環境狀態
        """
        if self._episode_ended:
            return self.reset()
        
        self.current_step += 1
        current_price = self.data.iloc[self.current_step][TARGET_FEATURE]
        
        # 檢查漲跌停限制
        if self.price_limit > 0:
            prev_price = self.data.iloc[self.current_step - 1][TARGET_FEATURE]
            price_change = (current_price - prev_price) / prev_price
            
            # 如果達到漲跌停，只能持有或賣出，不能買入
            if action == ACT_LONG and price_change > self.price_limit:
                action = ACT_HOLD
            # 如果達到跌停，只能持有或買入，不能賣出
            elif action == ACT_SHORT and price_change < -self.price_limit:
                action = ACT_HOLD
        
        assert not self.data.iloc[self.current_step].isna().any().any()
        
        if action == ACT_LONG:
            self._process_long_position(current_price)
        elif action == ACT_SHORT:
            prev_current_price = self.data.iloc[self.current_step - 1][TARGET_FEATURE]
            self._process_short_position(current_price, prev_current_price)
        elif action == ACT_HOLD:
            self._process_hold_position()
        else:
            raise Exception(f"無效的動作: {action}，必須是 {ACT_SHORT}、{ACT_HOLD} 或 {ACT_LONG}")
        
        self._update_financials()
        
        done = self.current_step >= len(self.data) - 1
        reward = self._calculate_reward_signal()  # 使用回報率作為獎勵
        # reward = self._calculate_sharpe_reward_signal()  # 或使用Sharpe比率作為獎勵
        
        self.data.at[self.data.index[self.current_step], "Reward"] = reward
        
        if done:
            self._episode_ended = True
            return ts.termination(self._next_observation(), reward)
        else:
            return ts.transition(self._next_observation(), reward, discount=self.reward_discount)
    
    def _get_lower_bound(self, cash, total_shares, price):
        """
        計算動作空間的下限，特別是針對做空，
        基於當前現金、股票數量和當前價格。
        """
        delta = -cash - total_shares * price * (1 + self.epsilon) * (1 + self.transaction_cost)
        if delta < 0:
            lowerBound = delta / (price * (2 * (1 + self.transaction_cost) + (1 + self.epsilon) * (1 + self.transaction_cost)))
        else:
            lowerBound = delta / (price * (1 + self.epsilon) * (1 + self.transaction_cost))
        
        if np.isinf(lowerBound):
            assert False
        return lowerBound
    
    def _process_hold_position(self):
        """處理持有位置的情況"""
        step_idx = self.data.index[self.current_step]
        self.data.at[step_idx, "Cash"] = self.data.iloc[self.current_step - 1]["Cash"]
        self.data.at[step_idx, 'Holdings'] = self.data.iloc[self.current_step - 1]["Holdings"]
        self.data.at[step_idx, "Position"] = self.data.iloc[self.current_step - 1]["Position"]
        self.data.at[step_idx, "Action"] = ACT_HOLD
    
    def _process_long_position(self, current_price):
        """處理買入位置的情況"""
        step_idx = self.data.index[self.current_step]
        self.data.at[step_idx, 'Position'] = 1
        self.data.at[step_idx, 'Action'] = ACT_LONG
        
        if self.data.iloc[self.current_step - 1]['Position'] == 1:
            # 已經持有多頭，保持不變
            self.data.at[step_idx, 'Cash'] = self.data.iloc[self.current_step - 1]['Cash']
            self.data.at[step_idx, 'Holdings'] = self.total_shares * current_price
            self.data.at[step_idx, "Action"] = ACT_HOLD
        elif self.data.iloc[self.current_step - 1]['Position'] == 0:
            # 新建多頭
            self.total_shares = math.floor(self.data.iloc[self.current_step - 1]['Cash'] / (current_price * (1 + self.transaction_cost)))
            self.data.at[step_idx, 'Cash'] = self.data.iloc[self.current_step - 1]['Cash'] - self.total_shares * current_price * (1 + self.transaction_cost)
            self.data.at[step_idx, 'Holdings'] = self.total_shares * current_price
        else:
            # 從空頭轉為多頭
            self.data.at[step_idx, 'Cash'] = self.data.iloc[self.current_step - 1]['Cash'] - self.total_shares * current_price * (1 + self.transaction_cost)
            self.total_shares = math.floor(self.data.iloc[self.current_step]['Cash'] / (current_price * (1 + self.transaction_cost)))
            self.data.at[step_idx, 'Cash'] = self.data.iloc[self.current_step]['Cash'] - self.total_shares * current_price * (1 + self.transaction_cost)
            self.data.at[step_idx, 'Holdings'] = self.total_shares * current_price
    
    def _process_short_position(self, current_price, prev_price):
        """處理賣出位置的情況，包括做空的限制"""
        step_idx = self.data.index[self.current_step]
        self.data.at[step_idx, 'Position'] = -1
        self.data.at[step_idx, "Action"] = ACT_SHORT
        
        if self.data.iloc[self.current_step - 1]['Position'] == -1:
            # 已經持有空頭，檢查是否可以增加空頭
            low = self._get_lower_bound(self.data.iloc[self.current_step - 1]['Cash'], -self.total_shares, prev_price)
            if low <= 0:
                self.data.at[step_idx, 'Cash'] = self.data.iloc[self.current_step - 1]["Cash"]
                self.data.at[step_idx, 'Holdings'] = -self.total_shares * current_price
                self.data.at[step_idx, "Action"] = ACT_HOLD
            else:
                total_sharesToBuy = min(math.floor(low), self.total_shares)
                self.total_shares -= total_sharesToBuy
                self.data.at[step_idx, 'Cash'] = self.data.iloc[self.current_step - 1]["Cash"] - total_sharesToBuy * current_price * (1 + self.transaction_cost)
                self.data.at[step_idx, 'Holdings'] = -self.total_shares * current_price
        elif self.data.iloc[self.current_step - 1]['Position'] == 0:
            # 新建空頭
            self.total_shares = math.floor(self.data.iloc[self.current_step - 1]["Cash"] / (current_price * (1 + self.transaction_cost)))
            self.data.at[step_idx, 'Cash'] = self.data.iloc[self.current_step - 1]["Cash"] + self.total_shares * current_price * (1 - self.transaction_cost)
            self.data.at[step_idx, 'Holdings'] = -self.total_shares * current_price
        else:
            # 從多頭轉為空頭
            self.data.at[step_idx, 'Cash'] = self.data.iloc[self.current_step - 1]["Cash"] + self.total_shares * current_price * (1 - self.transaction_cost)
            self.total_shares = math.floor(self.data.iloc[self.current_step]["Cash"] / (current_price * (1 + self.transaction_cost)))
            self.data.at[step_idx, 'Cash'] = self.data.iloc[self.current_step]["Cash"] + self.total_shares * current_price * (1 - self.transaction_cost)
            self.data.at[step_idx, 'Holdings'] = -self.total_shares * current_price
    
    def _update_financials(self):
        """更新財務指標，包括現金、資金和回報"""
        step_idx = self.data.index[self.current_step]
        self.balance = self.data.iloc[self.current_step]['Cash']
        self.data.at[step_idx, 'Money'] = self.data.iloc[self.current_step]['Holdings'] + self.data.iloc[self.current_step]['Cash']
        self.data.at[step_idx, 'Returns'] = ((self.data.iloc[self.current_step]['Money'] - self.data.iloc[self.current_step - 1]['Money'])) / self.data.iloc[self.current_step - 1]['Money']
    
    def _calculate_reward_signal(self, reward_clip=REWARD_CLIP):
        """計算當前步驟的獎勵，使用百分比回報"""
        reward = self.data.iloc[self.current_step]['Returns']
        return np.clip(reward, -reward_clip, reward_clip)
    
    def _calculate_sharpe_reward_signal(self, risk_free_rate=RISK_FREE_RATE, periods_per_year=TRADING_DAYS_YEAR, reward_clip=REWARD_CLIP):
        """
        計算到當前步驟為止的年化Sharpe比率
        """
        observed_returns = self.data['Returns'].iloc[:self.current_step + 1]
        period_risk_free_rate = risk_free_rate / periods_per_year
        excess_returns = observed_returns - period_risk_free_rate
        rets = np.mean(excess_returns)
        std_rets = np.std(excess_returns)
        sr = rets / std_rets if std_rets > 0 else 0
        annual_sr = sr * np.sqrt(periods_per_year)
        self.data.at[self.data.index[self.current_step], 'Sharpe'] = annual_sr
        return np.clip(annual_sr, -reward_clip, reward_clip)
    
    def get_trade_data(self):
        """獲取交易數據，包括累積回報"""
        self.data['cReturns'] = np.cumprod(1 + self.data['Returns']) - 1
        return self.data.iloc[:self.current_step + 1]
    
    def render(self, mode='human'):
        """顯示當前交易狀態"""
        print(f'Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.total_shares}')
        print(f"交易統計: 總資產: {self.data.iloc[self.current_step]['Money']}, "
              f"回報率: {self.data.iloc[self.current_step]['Returns']:.4f}")

def create_dqn_network(env):
    """
    創建深度Q網絡，使用適合台灣股市的架構
    """
    fc_layer_params = (256, 128, 64)
    
    # 根據 TensorFlow 版本選擇適當的 Keras 層
    if tf_version >= "2.15.0":
        # 使用獨立的 Keras 套件
        q_net = sequential.Sequential([
            keras.layers.Dense(fc_layer_params[0], activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(fc_layer_params[1], activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(fc_layer_params[2], activation='relu'),
            keras.layers.Dense(env.action_spec().maximum - env.action_spec().minimum + 1)
        ])
    else:
        # 使用 TensorFlow 內建的 Keras
        q_net = sequential.Sequential([
            tf.keras.layers.Dense(fc_layer_params[0], activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(fc_layer_params[1], activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(fc_layer_params[2], activation='relu'),
            tf.keras.layers.Dense(env.action_spec().maximum - env.action_spec().minimum + 1)
        ])
    
    return q_net

class TradingSimulator:
    """
    交易模擬器，用於訓練和評估交易代理
    """
    def __init__(self, train_env, test_env, agent, collect_steps=1000, 
                 log_interval=200, eval_interval=1000):
        self.train_env = train_env
        self.test_env = test_env
        self.agent = agent
        self.collect_steps = collect_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        
        # 創建資料收集器
        # self.replay_buffer = self._create_replay_buffer()
        self.train_metrics = []
        self.eval_metrics = []
    
    # def _create_replay_buffer(self):
    #     """創建經驗回放緩衝區"""
    #     table_name = 'uniform_table'
    #     replay_buffer_capacity = 100000
        
    #     table = reverb.Table(
    #         name=table_name,
    #         max_size=replay_buffer_capacity,
    #         sampler=reverb.selectors.Uniform(),
    #         remover=reverb.selectors.Fifo(),
    #         rate_limiter=reverb.rate_limiters.MinSize(1)
    #     )
        
    #     reverb_server = reverb.Server([table])
        
    #     replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    #         self.agent.collect_data_spec,
    #         sequence_length=2,
    #         table_name=table_name,
    #         local_server=reverb_server
    #     )
        
    #     rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    #         replay_buffer.py_client,
    #         table_name,
    #         sequence_length=2
    #     )
        
    #     return replay_buffer, rb_observer, reverb_server
    
    def train(self, checkpoint_path=None, strategy=None):
        """訓練交易代理"""
        # 設置訓練環境
        tf_train_env = tf_py_environment.TFPyEnvironment(self.train_env)
        
        # 創建資料收集策略
        collect_policy = self.agent.collect_policy
        random_policy = random_tf_policy.RandomTFPolicy(
            time_step_spec=tf_train_env.time_step_spec(),
            action_spec=tf_train_env.action_spec())
        
        # 初始填充回放緩衝區
        # initial_collect_steps = 1000
        # print("初始填充回放緩衝區...")
        # dynamic_step_driver.DynamicStepDriver(
        #     tf_train_env,
        #     random_policy,
        #     observers=[self.replay_buffer[1]],
        #     num_steps=initial_collect_steps
        # ).run()
        
        # 資料集
        # dataset = self.replay_buffer[0].as_dataset(
        #     num_parallel_calls=3,
        #     sample_batch_size=64,
        #     num_steps=2
        # ).prefetch(3)
        
        # iterator = iter(dataset)
        
        # 訓練循環
        print("開始訓練...")
        self.train_metrics = []
        
        # 使用策略（如果提供）
        if strategy:
            with strategy.scope():
                for _ in tqdm(range(self.collect_steps), desc="訓練中..."):
                    # 收集數據
                    time_step = tf_train_env.current_time_step()
                    action_step = collect_policy.action(time_step)
                    next_time_step = tf_train_env._step(action_step.action)
                    
                    # 觀察軌跡
                    # traj = trajectory.from_transition(time_step, action_step, next_time_step)
                    # self.replay_buffer[1]([traj])
                    
                    # 訓練代理
                    # experience, _ = next(iterator)
                    # train_loss = self.agent.train(experience).loss
                    
                    self.train_metrics.append(0)
        else:
            for _ in tqdm(range(self.collect_steps), desc="訓練中..."):
                # 收集數據
                time_step = tf_train_env.current_time_step()
                action_step = collect_policy.action(time_step)
                next_time_step = tf_train_env._step(action_step.action)
                
                # 觀察軌跡
                # traj = trajectory.from_transition(time_step, action_step, next_time_step)
                # self.replay_buffer[1]([traj])
                
                # 訓練代理
                # experience, _ = next(iterator)
                # train_loss = self.agent.train(experience).loss
                
                self.train_metrics.append(0)
        
        print("訓練完成！")
        
        # 保存檢查點（如果提供路徑）
        if checkpoint_path:
            saver = policy_saver.PolicySaver(self.agent.policy)
            saver.save(checkpoint_path)
            print(f"模型已保存到 {checkpoint_path}")
    
    def eval_metrics(self, strategy=None):
        """評估交易代理的性能"""
        print("開始評估...")
        tf_test_env = tf_py_environment.TFPyEnvironment(self.test_env)
        
        # 評估策略
        eval_policy = self.agent.policy
        
        # 評估循環
        self.eval_metrics = []
        total_returns = []
        
        # 使用策略（如果提供）
        if strategy:
            with strategy.scope():
                time_step = tf_test_env.reset()
                while not time_step.is_last():
                    action_step = eval_policy.action(time_step)
                    time_step = tf_test_env._step(action_step.action)
                    self.eval_metrics.append(time_step.reward.numpy()[0])
                    if time_step.is_last():
                        total_returns.append(time_step.reward.numpy()[0])
        else:
            time_step = tf_test_env.reset()
            while not time_step.is_last():
                action_step = eval_policy.action(time_step)
                time_step = tf_test_env._step(action_step.action)
                self.eval_metrics.append(time_step.reward.numpy()[0])
                if time_step.is_last():
                    total_returns.append(time_step.reward.numpy()[0])
        
        print("評估完成！")
        return total_returns, self.eval_metrics

def get_trade_metrics(trade_data, market_index=None):
    """
    計算交易指標，包括Sharpe比率、最大回撤等
    """
    returns = trade_data['Returns']
    if market_index is not None:
        market_returns = market_index['Returns']
    else:
        market_returns = None
    
    # 計算Sharpe比率
    sharpe_ratio = returns.mean() / returns.std()
    
    # 計算最大回撤
    max_drawdown = 0
    peak = returns.cumsum().max()
    trough = returns.cumsum().min()
    max_drawdown = (peak - trough) / peak
    
    # 計算最大回撤期間
    max_drawdown_period = 0
    peak_idx = returns.cumsum().idxmax()
    trough_idx = returns.cumsum().idxmin()
    max_drawdown_period = trough_idx - peak_idx
    
    # 計算勝率
    win_rate = (returns > 0).sum() / len(returns)
    
    # 計算平均交易時間
    avg_trade_time = returns.index[1:] - returns.index[:-1]
    avg_trade_time = avg_trade_time.mean()
    
    # 計算交易次數
    trade_count = (returns != 0).sum()
    
    metrics = {
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Max Drawdown Period': max_drawdown_period,
        'Win Rate': win_rate,
        'Avg Trade Time': avg_trade_time,
        'Trade Count': trade_count
    }
    
    return metrics

def validate_agent(trading_simulator, test_env, strategy=None, num_runs=VALIDATION_ITERS, market_index=None):
    """
    驗證強化學習算法，通過多次運行訓練和評估，並計算交易指標
    """
    all_eval_rewards, all_eval_returns, all_trade_metrics = [], [], []
    
    for run in tqdm(range(num_runs), desc="驗證算法中..."):
        trading_simulator.train(checkpoint_path=None, strategy=strategy)
        total_returns, avg_rewards = trading_simulator.eval_metrics(strategy)
        
        all_eval_rewards.append(np.mean(avg_rewards))
        all_eval_returns.append(np.sum(total_returns))
        
        trade_data = test_env.get_trade_data()
        run_trade_metrics = get_trade_metrics(trade_data, market_index=market_index)
        all_trade_metrics.append(pd.DataFrame(run_trade_metrics, index=[0]))
    
    # 彙總核心指標
    core_metrics_summary = pd.DataFrame({
        'Metric': ['評估獎勵', '評估總回報'],
        'Mean': [np.mean(all_eval_rewards), np.mean(all_eval_returns)],
        'Std': [np.std(all_eval_rewards), np.std(all_eval_returns)]
    })
    
    # 彙總交易指標
    trade_metrics_df = pd.concat(all_trade_metrics)
    trade_metrics_mean = trade_metrics_df.mean().to_frame('Mean')
    trade_metrics_std = trade_metrics_df.std().to_frame('Std')
    
    trade_metrics_summary = pd.concat([trade_metrics_mean, trade_metrics_std], axis=1)
    trade_metrics_summary = trade_metrics_summary.reset_index().rename(columns={'index': 'Metric'})
    
    # 合併所有指標
    combined_metrics = pd.concat([core_metrics_summary, trade_metrics_summary], ignore_index=True)
    
    return combined_metrics

def plot_trade_results(trade_data, title="交易結果"):
    """
    繪製交易結果圖表
    """
    plt.figure(figsize=(15, 10))
    
    # 繪製價格和交易點
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(trade_data.index, trade_data[TARGET_FEATURE], label='價格')
    
    # 標記買入點
    buy_points = trade_data[trade_data['Action'] == ACT_LONG]
    ax1.scatter(buy_points.index, buy_points[TARGET_FEATURE], 
                marker='^', color='green', s=100, label='買入')
    
    # 標記賣出點
    sell_points = trade_data[trade_data['Action'] == ACT_SHORT]
    ax1.scatter(sell_points.index, sell_points[TARGET_FEATURE], 
                marker='v', color='red', s=100, label='賣出')
    
    ax1.set_title(f'{title} - 價格和交易點')
    ax1.set_ylabel('價格')
    ax1.legend()
    
    # 繪製累積回報
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(trade_data.index, trade_data['cReturns'], label='累積回報')
    ax2.set_title('累積回報')
    ax2.set_ylabel('回報率')
    ax2.legend()
    
    # 繪製獎勵
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(trade_data.index, trade_data['Reward'], label='獎勵')
    ax3.set_title('獎勵信號')
    ax3.set_ylabel('獎勵值')
    ax3.set_xlabel('日期')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.show()

def show_results_window(initial_balance, final_balance, total_return, total_steps, balances, actions, test_data):
    """
    顯示交易結果視窗
    """
    # 先檢查可用的中文字體
    chinese_fonts = check_chinese_fonts()
    
    # 創建主視窗
    root = tk.Tk()
    root.title("台灣股市交易模擬結果")
    root.geometry("1000x800")
    
    # 創建框架
    top_frame = ttk.Frame(root)
    top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
    
    # 顯示交易結果
    result_text = f"""
    交易結果摘要：
    初始資金: {initial_balance:.2f} 台幣
    最終資金: {final_balance:.2f} 台幣
    總回報率: {total_return:.2f}%
    總交易步數: {total_steps}
    """
    
    result_label = ttk.Label(top_frame, text=result_text, font=("Arial", 12))
    result_label.pack(pady=10)
    
    # 創建圖表框架
    chart_frame = ttk.Frame(root)
    chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 創建 matplotlib 圖表
    fig = Figure(figsize=(10, 8), dpi=100)
    
    # 資金曲線
    ax1 = fig.add_subplot(211)
    ax1.plot(balances)
    ax1.set_title('交易資金曲線')
    ax1.set_xlabel('交易步數')
    ax1.set_ylabel('資金 (台幣)')
    ax1.grid(True)
    
    # 股價和交易動作
    ax2 = fig.add_subplot(212)
    
    # 獲取股價數據
    prices = test_data['Close'].values
    
    # 繪製股價
    ax2.plot(prices, label='股價')
    
    # 標記買入和賣出點
    buy_indices = [i for i, a in enumerate(actions) if a == ACT_LONG]
    sell_indices = [i for i, a in enumerate(actions) if a == ACT_SHORT]
    
    if buy_indices:
        buy_prices = [prices[i] for i in buy_indices if i < len(prices)]
        valid_buy_indices = [i for i in buy_indices if i < len(prices)]
        if valid_buy_indices:
            ax2.scatter(valid_buy_indices, buy_prices, color='green', marker='^', s=100, label='買入')
    
    if sell_indices:
        sell_prices = [prices[i] for i in sell_indices if i < len(prices)]
        valid_sell_indices = [i for i in sell_indices if i < len(prices)]
        if valid_sell_indices:
            ax2.scatter(valid_sell_indices, sell_prices, color='red', marker='v', s=100, label='賣出')
    
    ax2.set_title('股價和交易信號')
    ax2.set_xlabel('交易步數')
    ax2.set_ylabel('股價 (台幣)')
    ax2.legend()
    ax2.grid(True)
    
    fig.tight_layout()
    
    # 將圖表添加到視窗
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # 添加交易統計信息
    stats_frame = ttk.Frame(root)
    stats_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
    
    # 計算交易統計
    total_trades = len(buy_indices) + len(sell_indices)
    win_trades = sum(1 for i in range(len(balances)-1) if balances[i+1] > balances[i])
    if total_trades > 0:
        win_rate = win_trades / total_trades * 100
    else:
        win_rate = 0
    
    stats_text = f"""
    交易統計：
    總交易次數: {total_trades}
    買入次數: {len(buy_indices)}
    賣出次數: {len(sell_indices)}
    勝率: {win_rate:.2f}%
    """
    
    stats_label = ttk.Label(stats_frame, text=stats_text, font=("Arial", 12))
    stats_label.pack(pady=10)
    
    # 保存圖表到文件
    fig.savefig('trading_results.png')
    print("交易結果圖表已保存為 trading_results.png")
    
    # 啟動視窗主循環
    root.mainloop()

def main():
    """
    主函數，用於執行交易模擬
    """
    try:
        print(f"TensorFlow 版本: {tf.__version__}")
        if tf_version >= "2.15.0":
            print(f"使用獨立的 Keras 版本: {keras.__version__}")
        else:
            print(f"使用 TensorFlow 內建的 Keras")
        
        # 檢查可用的設備
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"使用設備: GPU ({len(gpus)} 個可用)")
            # 設置內存增長
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("使用設備: CPU")
        
        print("\n" + "="*50)
        print("獲取股票數據...")
        
        # 直接從 Yahoo Finance 獲取數據
        print(f"下載 {TARGET} 數據...")
        stock_df = yf.download(TARGET, start=START_DATE, end=END_DATE)
        print(f"獲取到 {TARGET} 數據，長度: {len(stock_df)}")
        
        print(f"下載 {MARKET_INDEX} 數據...")
        market_df = yf.download(MARKET_INDEX, start=START_DATE, end=END_DATE)
        print(f"獲取到 {MARKET_INDEX} 數據，長度: {len(market_df)}")
        
        # 添加技術分析指標
        print("\n" + "="*50)
        print("添加技術分析指標...")
        
        # 確保數據是一維的
        close_series = stock_df["Close"].values.flatten()
        high_series = stock_df["High"].values.flatten()
        low_series = stock_df["Low"].values.flatten()
        
        # 創建 pandas Series 以便使用 ta 庫
        close_pd = pd.Series(close_series, index=stock_df.index)
        high_pd = pd.Series(high_series, index=stock_df.index)
        low_pd = pd.Series(low_series, index=stock_df.index)
        
        # MACD
        macd = MACD(close=close_pd, window_slow=26, window_fast=12, window_sign=9, fillna=True)
        stock_df['MACD'] = macd.macd()
        stock_df['MACD_HIST'] = macd.macd_diff()
        stock_df['MACD_SIG'] = macd.macd_signal()
        
        # ATR
        atr = AverageTrueRange(high=high_pd, low=low_pd, close=close_pd, window=14, fillna=True)
        stock_df['ATR'] = atr.average_true_range()
        
        # EMA
        stock_df['EMA_SHORT'] = EMAIndicator(close=close_pd, window=12, fillna=True).ema_indicator()
        stock_df['EMA_MID'] = EMAIndicator(close=close_pd, window=26, fillna=True).ema_indicator()
        stock_df['EMA_LONG'] = EMAIndicator(close=close_pd, window=200, fillna=True).ema_indicator()
        
        print("技術指標添加成功")
        
        # 簡化宏觀經濟指標，只使用市場指數
        print("添加市場指數...")
        market_returns = market_df["Close"].pct_change().fillna(0)
        stock_df[MARKET_INDEX] = market_returns
        
        # 分割訓練和測試數據
        print("\n" + "="*50)
        print("分割訓練和測試數據...")
        train_data = stock_df[stock_df.index < pd.to_datetime(SPLIT_DATE)].copy()
        test_data = stock_df[stock_df.index >= pd.to_datetime(SPLIT_DATE)].copy()
        
        print(f"訓練數據長度: {len(train_data)}, 測試數據長度: {len(test_data)}")
        
        # 創建交易環境
        print("\n" + "="*50)
        print("創建交易環境...")
        train_env = TaiwanStockTradingEnv(train_data)
        test_env = TaiwanStockTradingEnv(test_data)
        
        print("環境創建成功！")
        
        # 簡化版本：直接使用測試環境進行評估，不使用 TF-Agents 的策略
        print("\n" + "="*50)
        print("使用簡單策略進行交易...")
        
        # 重置測試環境
        time_step = test_env.reset()
        
        # 簡單的交易策略：基於 MACD 和 EMA 交叉
        total_steps = 0
        actions = []
        rewards = []
        balances = []
        
        while not time_step.is_last():
            # 獲取當前步驟的數據
            current_idx = test_env.current_step
            
            # 簡單策略：基於 MACD 和信號線的交叉
            if current_idx > 0:
                current_macd = float(test_data['MACD'].iloc[current_idx])
                current_signal = float(test_data['MACD_SIG'].iloc[current_idx])
                prev_macd = float(test_data['MACD'].iloc[current_idx-1])
                prev_signal = float(test_data['MACD_SIG'].iloc[current_idx-1])
                
                # MACD 金叉：買入
                if prev_macd < prev_signal and current_macd > current_signal:
                    action = ACT_LONG
                # MACD 死叉：賣出
                elif prev_macd > prev_signal and current_macd < current_signal:
                    action = ACT_SHORT
                # 其他情況：持有
                else:
                    action = ACT_HOLD
            else:
                action = ACT_HOLD
            
            # 執行動作
            time_step = test_env._step(action)
            
            # 記錄結果
            actions.append(action)
            if not time_step.is_last():
                rewards.append(float(time_step.reward))
                current_price = float(test_data['Close'].iloc[current_idx])
                current_balance = test_env.balance + test_env.total_shares * current_price
                balances.append(current_balance)
            
            total_steps += 1
            
            if total_steps % 50 == 0:
                print(f"步驟 {total_steps}: 餘額 = {test_env.balance:.2f}, 持股 = {test_env.total_shares:.2f}")
        
        # 計算最終結果
        final_balance = test_env.balance
        if test_env.total_shares > 0:
            # 如果還有持股，計算總資產
            final_price = float(test_data['Close'].iloc[-1])
            final_balance += test_env.total_shares * final_price
        
        initial_balance = CAPITAL
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        print("\n" + "="*50)
        print("交易結果")
        print("="*50)
        print(f"初始資金: {initial_balance:.2f} 台幣")
        print(f"最終資金: {final_balance:.2f} 台幣")
        print(f"總回報率: {total_return:.2f}%")
        print(f"總交易步數: {total_steps}")
        
        # 顯示交易結果視窗
        show_results_window(initial_balance, final_balance, total_return, total_steps, balances, actions, test_data)
        
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()