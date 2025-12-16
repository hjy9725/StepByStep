import akshare as ak
import pandas as pd
import numpy as np
import time
import datetime
import os
import sys
import threading
import tkinter as tk
import warnings
from collections import deque

# 深度学习库
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 声音库
try:
    import winsound
except ImportError:
    winsound = None

# ================= 1. 参数控制台 (请在这里修改参数) =================
class Config:
    # --- 基础设置 ---
    STOCK_CODE = "002463"       # [修改] 你的股票代码
    
    # --- AI 个性设置 ---
    # HISTORY_DAYS = 500          # 训练用多少天的数据
    HISTORY_DAYS = 100          # 训练用多少天的数据
    RISK_FACTOR = 1.05           # [修改] 贪婪系数
                                # 1.0 = 相信AI; 1.2 = 比AI更保守(跌更深才买); 0.8 = 激进
    
    # --- 盘中动态修正 (新功能) ---
    ENABLE_DYNAMIC_ADJUST = True # 是否开启盘中修正
    PANIC_SENSITIVITY = 0.5      # [修改] 恐慌敏感度
                                 # 如果1分钟内跌幅超过 0.5%，系统会认为主力在砸盘
                                 # 此时阈值会自动下移，避开锋芒
    
    # --- 监控频率 ---
    REALTIME_INTERVAL = 3        # 3秒刷一次

# ================= 2. 强报警系统 =================
alarm_active = False

def play_alarm_loop():
    global alarm_active
    while alarm_active:
        if winsound:
            winsound.Beep(2500, 100) # 更加急促的声音
            time.sleep(0.05)
            winsound.Beep(2500, 100)
            time.sleep(0.5)
        else:
            print('\a'); time.sleep(1)

def show_force_alert_window(msg, current_price):
    global alarm_active
    if not alarm_active:
        alarm_active = True
        t = threading.Thread(target=play_alarm_loop, daemon=True)
        t.start()
    
    root = tk.Tk()
    root.title(f"⚡ 动态狙击信号")
    w, h = 600, 550
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.attributes('-topmost', True)
    root.configure(bg='red')
    
    tk.Label(root, text="🚀 AI 捕捉到买点 🚀", font=("黑体", 32, "bold"), bg='red', fg='yellow').pack(pady=20)
    tk.Label(root, text=f"股票: {Config.STOCK_CODE}", font=("微软雅黑", 20), bg='red', fg='white').pack()
    tk.Label(root, text=f"现价: {current_price}", font=("微软雅黑", 36, "bold"), bg='red', fg='white').pack(pady=10)
    tk.Label(root, text=msg, font=("微软雅黑", 14), bg='red', fg='white', wraplength=550).pack(pady=10)
    
    def stop_alarm():
        global alarm_active
        alarm_active = False
        root.destroy()

    tk.Button(root, text="我已处理，停止报警", font=("微软雅黑", 20, "bold"), 
              command=stop_alarm, bg='white', fg='red').pack(pady=30)
    root.mainloop()

# ================= 3. AI 大脑 (训练部分) =================
class AIBrain:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def fetch_and_train(self):
        print(f"\n🧠 [AI] 正在连接神经网络...")
        print(f"📡 [AI] 拉取 {Config.STOCK_CODE} 历史数据...")
        
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=Config.HISTORY_DAYS*1.5)).strftime("%Y%m%d")
        
        try:
            df = ak.stock_zh_a_hist(symbol=Config.STOCK_CODE, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low"})
        except Exception as e:
            print(f"❌ 数据拉取失败: {e}")
            return None

        # 计算最大下杀幅度
        df['max_drop_pct'] = (df['low'] - df['open']) / df['open'] * 100
        
        # 训练集
        data_set = df[['open', 'close', 'high', 'low', 'max_drop_pct']].values
        scaled_data = self.scaler.fit_transform(data_set)
        
        X, y = [], []
        time_step = 30
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i-time_step:i, :])
            y.append(scaled_data[i, 4])
            
        X, y = np.array(X), np.array(y)
        
        print(f"🔥 [AI] 正在重训模型 (适应最新股性)...")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, batch_size=32, epochs=5, verbose=0) # 快速训练5轮
        
        # 预测今日基础阈值
        last_30 = np.array([scaled_data[-time_step:]])
        pred_scaled = model.predict(last_30, verbose=0)
        
        dummy = np.zeros((1, 5))
        dummy[0, 4] = pred_scaled[0][0]
        base_threshold = self.scaler.inverse_transform(dummy)[0, 4]
        
        # 兜底逻辑：如果AI预测跌幅太小（比如预测涨），强制给一个最小值
        if base_threshold > -1.5: base_threshold = -1.5
            
        return base_threshold

# ================= 4. 实时监控层 (含恐慌传感器) =================
def run_sniper():
    # 1. 初始化 AI
    ai = AIBrain()
    base_threshold = ai.fetch_and_train()
    
    if base_threshold is None: return

    # 2. 初始化价格缓存 (用于计算瞬时跌速)
    # 队列长度20，存最近60秒的价格 (3秒一次 * 20 = 60秒)
    price_history = deque(maxlen=20) 
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*60)
    print(f"🤖 AI 动态狙击手 Pro | 目标: {Config.STOCK_CODE}")
    print(f"📉 AI 预测今日支撑位: {base_threshold:.2f}%")
    print(f"🛡️ 基础报警阈值: {base_threshold * Config.RISK_FACTOR:.2f}%")
    print(f"🌪️ 盘中恐慌修正: {'已开启' if Config.ENABLE_DYNAMIC_ADJUST else '未开启'}")
    print("="*60)

    while True:
        try:
            spot = ak.stock_zh_a_spot_em()
            target = spot[spot['代码'] == Config.STOCK_CODE]
            
            if target.empty:
                time.sleep(3); continue
                
            current_price = float(target.iloc[0]['最新价'])
            open_price = float(target.iloc[0]['今开'])
            
            # 存入历史记录
            price_history.append(current_price)
            
            # --- 核心：计算动态阈值 ---
            current_threshold = base_threshold * Config.RISK_FACTOR
            panic_msg = ""
            
            if Config.ENABLE_DYNAMIC_ADJUST and len(price_history) >= 2:
                # 计算最近1分钟的跌速
                price_1min_ago = price_history[0]
                drop_speed = (current_price - price_1min_ago) / price_1min_ago * 100
                
                # 如果1分钟内跌幅超过恐慌值 (比如 -0.5%)，说明正在砸盘
                if drop_speed < -Config.PANIC_SENSITIVITY:
                    # 动态下移阈值：跌得越快，阈值越低
                    # 比如：原本 -3%，现在瞬间跌了 1%，阈值临时调整为 -3% + (-1%) = -4%
                    adjustment = drop_speed 
                    current_threshold += adjustment
                    panic_msg = f"⚠️ 检测到急跌({drop_speed:.2f}%)，阈值已自动下移至 {current_threshold:.2f}%"

            # 计算当前累计跌幅
            drop_from_open = (current_price - open_price) / open_price * 100
            
            # 打印面板
            now = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\r[{now}] 现价:{current_price} | 跌幅:{drop_from_open:.2f}% | 动态阈值:{current_threshold:.2f}% {panic_msg}", end=" "*10)
            
            # 触发判断
            if drop_from_open <= current_threshold:
                print("\n")
                full_msg = (f"当前跌幅 {drop_from_open:.2f}% 击穿动态阈值 {current_threshold:.2f}%\n"
                            f"原始AI预测: {base_threshold:.2f}%\n"
                            f"{panic_msg}")
                show_force_alert_window(full_msg, current_price)
                # 报警后清空历史，防止连续触发
                price_history.clear()
                
            time.sleep(Config.REALTIME_INTERVAL)
            
        except Exception as e:
            print(f"\nRunning... {e}") # 简化报错
            time.sleep(3)

if __name__ == "__main__":
    run_sniper()