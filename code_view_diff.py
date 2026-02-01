import os
import sys
import time
import json
import random
import datetime
import traceback
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# 数据处理
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import RobustScaler

# 深度学习
import tensorflow as tf
from tensorflow.keras import layers, models

# UI 库
import tkinter as tk
from tkinter import ttk
from colorama import init, Fore, Style

# 导入配置
try:
    import token_stock_list_config as cfg
except ImportError:
    print("❌ 错误: 找不到 token_stock_list_config.py 文件。")
    sys.exit(1)

# 初始化设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
init(autoreset=True)

# ==========================================
# 模块 A: 数据管理 (增强版：资金流+大盘)
# ==========================================
class DataManager:
    def __init__(self):
        self.cache_dir = "./stock_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.anchor_date = datetime.datetime.now().strftime("%Y%m%d")

    def _get_tencent_code(self, code):
        if code.startswith('6'): return f"sh{code}"
        elif code.startswith('0') or code.startswith('3'): return f"sz{code}"
        return code # 指数通常自带前缀

    def fetch_indices_snapshot(self):
        """获取大盘指数: 上证, 深证, 创业板"""
        # sh000001:上证, sz399001:深证, sz399006:创业板
        url = "http://qt.gtimg.cn/q=s_sh000001,s_sz399001,s_sz399006"
        indices = {"sh": 0, "sz": 0, "cyb": 0}
        try:
            resp = requests.get(url, timeout=3)
            lines = resp.text.split(';')
            # 腾讯简版接口: v_s_sh000001="1~上证指数~3200.50~-10.20~-0.32~..."
            # Index 3:涨跌额, Index 5:涨跌幅(%)
            if len(lines) >= 3:
                indices['sh'] = float(lines[0].split('~')[5])
                indices['sz'] = float(lines[1].split('~')[5])
                indices['cyb'] = float(lines[2].split('~')[5])
        except:
            pass
        return indices

    def fetch_fund_flow(self, code):
        """
        获取资金流向 (主力净流入)
        接口: http://qt.gtimg.cn/q=ff_sh600519
        返回: code~主力流入~主力流出~主力净流入~主力净流入占比...
        """
        symbol = self._get_tencent_code(code)
        url = f"http://qt.gtimg.cn/q=ff_{symbol}"
        data = {
            "main_net": 0.0, # 主力净流入(万)
            "main_pct": 0.0, # 主力净占比
            "retail_net": 0.0 # 散户净流入(万)
        }
        try:
            resp = requests.get(url, timeout=3)
            # 格式: v_ff_sh600519="sh600519~30353.50~34977.00~-4623.50~-7.08~..."
            # Index 3: 主力净流入(万), Index 4: 主力净占比(%)
            items = resp.text.split('"')[1].split('~')
            if len(items) > 10:
                data['main_net'] = float(items[3])
                data['main_pct'] = float(items[4])
                # 腾讯这个接口 散户数据通常在后面，简单起见我们重点看主力
                # 若主力净流入为负，散户通常为正
                data['retail_net'] = -data['main_net'] 
        except:
            pass
        return data

    def fetch_tencent_history(self, code):
        """获取历史K线 (保持不变)"""
        symbol = self._get_tencent_code(code)
        url = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {"param": f"{symbol},day,,,320,qfq"}
        proxies = {"http": None, "https": None}
        try:
            res = requests.get(url, params=params, proxies=proxies, timeout=5)
            data = res.json()
            if 'data' not in data or symbol not in data['data']: return pd.DataFrame()
            stock_data = data['data'][symbol]
            k_lines = stock_data.get('qfqday') or stock_data.get('day')
            if not k_lines: return pd.DataFrame()
            cleaned_data = [row[:6] for row in k_lines]
            df = pd.DataFrame(cleaned_data, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
            cols = ['open', 'close', 'high', 'low', 'volume']
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            return df
        except: return pd.DataFrame()

    def get_history_data(self, code):
        file_path = os.path.join(self.cache_dir, f"{code}_{self.anchor_date}.csv")
        if os.path.exists(file_path):
            try: return pd.read_csv(file_path)
            except: pass 
        time.sleep(random.uniform(0.05, 0.1)) 
        df = self.fetch_tencent_history(code)
        if not df.empty:
            df.to_csv(file_path, index=False)
            return df
        return pd.DataFrame()

    def get_realtime_snapshot(self, stock_list):
        qt_codes = [self._get_tencent_code(c) for c in stock_list]
        results = {}
        batch_size = 60
        proxies = {"http": None, "https": None} 
        for i in range(0, len(qt_codes), batch_size):
            batch = qt_codes[i:i+batch_size]
            url = f"http://qt.gtimg.cn/q={','.join(batch)}"
            try:
                resp = requests.get(url, proxies=proxies, timeout=3)
                lines = resp.text.split(';')
                for line in lines:
                    if len(line) < 10: continue
                    try:
                        var_name = line.split('=')[0]
                        code = var_name.split('_')[-1][2:] 
                        content = line.split('=')[1].strip('"')
                        data = content.split('~')
                        if len(data) < 40: continue
                        price = float(data[3])
                        pre_close = float(data[4])
                        results[code] = {
                            'name': data[1],
                            'price': price,
                            'pre_close': pre_close,
                            'volume': float(data[6]) * 100,
                            'amount': float(data[37]) * 10000,
                            'pct': (price - pre_close) / pre_close * 100 if pre_close > 0 else 0
                        }
                    except: continue
            except: pass
        return results

# ==========================================
# 模块 B: 特征工程 (保持不变)
# ==========================================
class AlphaFactors:
    @staticmethod
    def process(df):
        if df.empty or len(df) < 30: return df
        df = df.sort_values('date').reset_index(drop=True)
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['Bias20'] = (df['close'] - df['MA20']) / df['MA20']
        df['tr'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
        df['ATR'] = df['tr'].rolling(window=14).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        std = df['close'].rolling(20).std()
        df['BOLL_POS'] = (df['close'] - (df['MA20'] - 2*std)) / (4*std + 0.0001)
        
        # 计算历史 MA 趋势斜率 (简单线性回归)
        # 取最近5天的 MA20 计算斜率
        y = df['MA20'].iloc[-5:].values
        x = np.arange(len(y))
        if len(y) == 5:
            slope, _ = np.polyfit(x, y, 1)
            # 将这个斜率存储在最后一行，供后续读取
            df.loc[df.index[-1], 'MA_SLOPE'] = slope
        else:
             df.loc[df.index[-1], 'MA_SLOPE'] = 0

        df.dropna(inplace=True)
        return df

# ==========================================
# 模块 C: 预测模型 (保持不变)
# ==========================================
class EnsembleBrain:
    def __init__(self, stock_id):
        self.stock_id = stock_id
        self.seq_len = getattr(cfg, 'SEQ_LEN', 180) 
        self.scaler = RobustScaler()
        self.is_trained = False
        self.model = self._build_model()
    
    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.seq_len, 5)), 
            layers.LSTM(32, return_sequences=False),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_on_fly(self, df):
        if len(df) < self.seq_len + 5: return
        features = ['close', 'Bias20', 'RSI', 'BOLL_POS', 'ATR']
        data = df[features].values
        try:
            data_scaled = self.scaler.fit_transform(data)
            X, y = [], []
            for i in range(self.seq_len, len(data) - 1):
                X.append(data_scaled[i-self.seq_len:i])
                y.append((data[i+1, 0] - data[i, 0]) * 100)
            if len(X) > 5:
                self.model.fit(np.array(X), np.array(y), epochs=2, batch_size=32, verbose=0)
                self.is_trained = True
        except: pass
    
    def predict_score(self, recent_df):
        if not self.is_trained or len(recent_df) < self.seq_len: return 50.0 
        features = ['close', 'Bias20', 'RSI', 'BOLL_POS', 'ATR']
        try:
            raw = recent_df[features].values[-self.seq_len:]
            scaled = self.scaler.transform(raw)
            pred = self.model.predict(scaled.reshape(1, self.seq_len, 5), verbose=0)
            return max(0, min(100, 50 + float(pred[0][0]) * 10))
        except: return 50.0

# ==========================================
# 模块 D: 双核 LLM (重写：分批策略 Prompt)
# ==========================================
class DualAdvisor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.ds_key = getattr(cfg, 'DEEPSEEK_API_KEY', "")

    def _call_deepseek(self, prompt):
        print(f"\n{Fore.YELLOW}------ [LOG] >>> Prompt Sent ------")
        print(f"{Fore.CYAN}{prompt[:]}")
        
        if not self.ds_key or "sk-" not in self.ds_key: 
            return {"provider": "DeepSeek", "action": "WAIT", "plan": []}
        
        headers = {"Authorization": f"Bearer {self.ds_key}", "Content-Type": "application/json"}
        payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        try:
            resp = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, proxies={"http": None, "https": None}, timeout=20)
            content = resp.json()['choices'][0]['message']['content']
            print(f"\n{Fore.GREEN}------ [LOG] <<< DeepSeek Response ------")
            print(f"{content}")
            return self._parse_json(content, "DeepSeek")
        except Exception as e:
            print(f"DeepSeek Error: {e}")
            return {"provider": "DeepSeek", "action": "ERROR", "plan": []}

    def _call_qwen(self, prompt):
        time.sleep(0.5)
        # 模拟分批策略返回
        return {
            "provider": "Qwen(Mock)", 
            "action": "EXECUTE", 
            "score": 85,
            "plan": ["现价买入30%底仓", "若回调至MA20(27.5)加仓30%", "突破前高28.8加仓40%"],
            "reason": "多头排列，主力资金持续流入，大盘配合，建议金字塔建仓。"
        }

    def _parse_json(self, text, provider):
        try:
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            data['provider'] = provider
            # 兼容性处理：如果模型没返回 plan，把 suggest_price 转为 plan
            if 'plan' not in data:
                price = data.get('suggest_price', 'Market')
                data['plan'] = [f"单一价格操作: {price}"]
            return data
        except:
            return {"provider": provider, "action": "MANUAL", "plan": ["JSON解析失败，请人工判断"]}

    def consult(self, stock, price, direction, d, indices, funds):
        action_cn = "低吸买入 (BUY)" if direction == "BUY" else "高抛止盈 (SELL)"
        
        prompt = f"""
        你是一个精通A股日内T+0和波段交易的顶级交易员。当前触发【{action_cn}】信号。
        请结合大盘环境、资金流向和个股走势，给出**分批阶梯交易策略**，防止卖飞或被套。

        【市场环境 (Indices)】
        上证: {indices['sh']:.2f}% | 深证: {indices['sz']:.2f}% | 创业板: {indices['cyb']:.2f}%
        
        【个股信息: {stock}】
        现价: {price} (涨跌幅: {d['pct']:.2f}%)
        成交量: {d['volume']/100:.0f}手
        
        【资金博弈 (Capital Flow)】
        主力净流入: {funds['main_net']:.1f}万 (正数代表主力买入，负数代表流出)
        主力净占比: {funds['main_pct']:.2f}% (重要参考！)

        【技术指标详解】
        1. 均价(VWAP): {d['vwap']:.2f}
        2. 乖离率(Bias): {d['bias']:.2f}% (触发阈值: {d['threshold']:.2f}%)
        3. 分时斜率(Intraday Slope): {d['intraday_slope']:.4f} (当下分钟级别的冲高/杀跌力度)
        4. 日线趋势斜率(MA Slope): {d['ma_slope']:.4f} (0附近震荡，正数上升趋势)

        【任务要求】
        不要只给一个价格！请制定“分批操作计划”。
        - 如果是买入：考虑分批建仓（底仓、加仓点、止损点）。
        - 如果是卖出：考虑分批止盈（锁定利润、预留仓位博涨停、防踏空）。
        
        必须返回纯 JSON 格式：
        {{
            "action": "EXECUTE" 或 "WAIT",
            "score": 0-100 (信心分),
            "reason": "简短分析(包含对大盘和资金的看法)",
            "plan": [
                "第一步: 现价卖出30%锁定利润",
                "第二步: 若冲高至28.8元再卖出40%",
                "第三步: 剩余30%若跌破均价线清仓，否则持有博涨停"
            ]
        }}
        """
        f1 = self.executor.submit(self._call_deepseek, prompt)
        f2 = self.executor.submit(self._call_qwen, prompt)
        try:
            return [f1.result(timeout=20), f2.result(timeout=20)]
        except:
            return []

# ==========================================
# 模块 F: UI (高级版：显示大盘/资金/策略)
# ==========================================
class PopupManager:
    def __init__(self):
        self.root = None
    
    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        
    def _run(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.mainloop()
        
    def show(self, stock, price, direction, analysis, detailed_data, indices, funds):
        if self.root:
            self.root.after(0, lambda: self._create_win(stock, price, direction, analysis, detailed_data, indices, funds))
            
    def _create_win(self, stock, price, direction, analysis, d, idx, funds):
        win = tk.Toplevel(self.root)
        win.title(f"{direction} 策略 - {stock}")
        win.attributes("-topmost", True)
        
        # 主题色：买绿卖红
        bg_color = "#005500" if direction == "BUY" else "#8B0000" 
        fg_color = "white"
        win.configure(bg=bg_color)
        win.geometry("600x750") # 进一步加大窗口
        
        # 字体
        font_title = ("Microsoft YaHei", 14, "bold")
        font_big = ("Arial", 32, "bold")
        font_norm = ("Microsoft YaHei", 10)
        font_small = ("Microsoft YaHei", 9)
        
        # === 1. 顶部大盘环境 ===
        idx_color = "#CCCCCC"
        idx_frame = tk.Frame(win, bg="#222222", pady=5) # 深色顶栏
        idx_frame.pack(fill="x")
        idx_str = f"🌏 大盘环境: 上证 {idx['sh']}%  |  深证 {idx['sz']}%  |  创业板 {idx['cyb']}%"
        tk.Label(idx_frame, text=idx_str, font=font_small, bg="#222222", fg="#00FF00" if idx['sh']>0 else "#FF4444").pack()

        # === 2. 股票信息 ===
        tk.Label(win, text=f"⚡ {direction} 信号: {stock}", font=font_title, bg=bg_color, fg="#FFD700").pack(pady=(10,0))
        
        price_frame = tk.Frame(win, bg=bg_color)
        price_frame.pack()
        tk.Label(price_frame, text=f"{price}", font=font_big, bg=bg_color, fg=fg_color).pack(side="left")
        pct_color = "#00FF00" if d['pct'] < 0 else "#FF4500"
        tk.Label(price_frame, text=f" {d['pct']:.2f}%", font=("Arial", 18, "bold"), bg=bg_color, fg=pct_color).pack(side="left", padx=10)

        # === 3. 资金博弈 (新增) ===
        fund_frame = tk.Frame(win, bg=bg_color, pady=5)
        fund_frame.pack(fill="x", padx=20)
        
        # 主力净流入可视化
        fund_val = funds['main_net']
        fund_str = f"主力净流入: {int(fund_val)}万"
        fund_fg = "#FF3333" if fund_val > 0 else "#33FF33" # 红进绿出
        tk.Label(fund_frame, text=fund_str, font=("Microsoft YaHei", 12, "bold"), bg=bg_color, fg=fund_fg).pack()
        
        ratio_str = f"主力占比: {funds['main_pct']}%  (博弈强度)"
        tk.Label(fund_frame, text=ratio_str, font=font_small, bg=bg_color, fg="#DDDDDD").pack()

        ttk.Separator(win, orient="horizontal").pack(fill="x", padx=20, pady=5)

        # === 4. 技术指标矩阵 ===
        info_frame = tk.Frame(win, bg=bg_color)
        info_frame.pack(fill="x", padx=30)
        
        # 使用 Grid 布局对齐
        tk.Label(info_frame, text="均价(VWAP):", font=font_norm, bg=bg_color, fg="#AAA").grid(row=0, column=0, sticky="w")
        tk.Label(info_frame, text=f"{d['vwap']:.2f}", font=font_norm, bg=bg_color, fg="white").grid(row=0, column=1, sticky="e")
        
        tk.Label(info_frame, text="分时斜率:", font=font_norm, bg=bg_color, fg="#AAA").grid(row=0, column=2, sticky="w", padx=(20,0))
        tk.Label(info_frame, text=f"{d['intraday_slope']:.4f}", font=font_norm, bg=bg_color, fg="white").grid(row=0, column=3, sticky="e")

        tk.Label(info_frame, text="当前乖离:", font=font_norm, bg=bg_color, fg="#AAA").grid(row=1, column=0, sticky="w")
        tk.Label(info_frame, text=f"{d['bias']:.2f}%", font=("Arial", 11, "bold"), bg=bg_color, fg="#FFD700").grid(row=1, column=1, sticky="e")
        
        tk.Label(info_frame, text="触发阈值:", font=font_norm, bg=bg_color, fg="#AAA").grid(row=1, column=2, sticky="w", padx=(20,0))
        tk.Label(info_frame, text=f"{d['threshold']:.2f}%", font=font_norm, bg=bg_color, fg="white").grid(row=1, column=3, sticky="e")

        # === 5. AI 策略展示 (分批计划) ===
        tk.Label(win, text="🤖 智能阶梯策略 (避免卖飞/深套)", font=font_title, bg=bg_color, fg="#ADD8E6", anchor="w").pack(fill="x", padx=20, pady=(15, 5))
        
        ai_container = tk.Frame(win, bg=bg_color)
        ai_container.pack(fill="both", expand=True, padx=15, pady=5)
        
        for res in analysis:
            card = tk.Frame(ai_container, bg=bg_color, bd=1, relief="groove")
            card.pack(fill="x", pady=5)
            
            # 标题行
            act = res.get('action', 'WAIT')
            score = res.get('score', 0)
            header_color = "#00FF00" if act == "EXECUTE" else "#AAAAAA"
            tk.Label(card, text=f"[{res.get('provider')}] {act} (信心:{score})", font=("Consolas", 11, "bold"), bg=bg_color, fg=header_color, anchor="w").pack(fill="x")
            
            # 理由
            reason = res.get('reason', '无')
            tk.Label(card, text=f"💡 分析: {reason}", font=font_small, bg=bg_color, fg="#EEE", wraplength=520, justify="left", anchor="w").pack(fill="x", pady=2)
            
            # 策略计划列表 (重点！)
            plans = res.get('plan', [])
            if plans:
                tk.Label(card, text="📋 操作计划:", font=("Microsoft YaHei", 9, "bold"), bg=bg_color, fg="#FFD700", anchor="w").pack(fill="x", pady=(5,0))
                for step in plans:
                    tk.Label(card, text=f"  • {step}", font=font_small, bg=bg_color, fg="white", anchor="w").pack(fill="x")
            
            tk.Label(card, text="-"*80, bg=bg_color, fg="#444").pack()

# ==========================================
# 主程序
# ==========================================
class MonitorApp:
    def __init__(self):
        raw_list = cfg.STOCK_LIST + cfg.SHORT_STUDIED_LIST
        self.stocks = list(dict.fromkeys(raw_list))
        print(f"{Fore.CYAN}=== 系统启动: 监控 {len(self.stocks)} 只股票 ===")
        
        self.dm = DataManager()
        self.runtime = {}
        self.advisor = DualAdvisor()
        self.ui = PopupManager()
        self.ui.start()
        
        self._init_models()

    def _init_models(self):
        print(f"{Fore.GREEN}=== 初始化数据与模型 ===")
        for code in self.stocks:
            # 读取历史数据计算 MA Slope
            df = self.dm.get_history_data(code)
            ma_slope = 0
            if not df.empty:
                try:
                    df = AlphaFactors.process(df)
                    if 'MA_SLOPE' in df.columns:
                        ma_slope = df['MA_SLOPE'].iloc[-1]
                except: pass

            self.runtime[code] = {
                'price_q': deque(maxlen=cfg.PRICE_WINDOW_SIZE), 
                'last_alert': 0,
                'ma_slope': ma_slope # 存储日线趋势
            }
        print("模型就绪。")

    def run(self):
        print(f"{Fore.GREEN}=== 开始实时监控 ===")
        while True:
            try:
                # 1. 轮询大盘指数
                indices = self.dm.fetch_indices_snapshot()
                
                # 2. 轮询个股
                snapshot = self.dm.get_realtime_snapshot(self.stocks)
                log_line = [f"SH:{indices['sh']}%"] # 日志加上大盘
                
                for code, data in snapshot.items():
                    rt = self.runtime.get(code)
                    if not rt: continue
                    
                    price = data['price']
                    rt['price_q'].append(price)
                    
                    vwap = data['amount'] / data['volume'] if data['volume'] > 0 else price
                    bias = (price - vwap) / vwap * 100
                    
                    # 计算分时斜率 (Intraday Slope)
                    intraday_slope = 0
                    if len(rt['price_q']) >= 5:
                        y = list(rt['price_q'])
                        x = np.arange(len(y))
                        slope, _ = np.polyfit(x, y, 1)
                        intraday_slope = slope * 100
                    
                    # 动态阈值 (使用分时斜率调整)
                    thresh_buy = -cfg.BASE_THRESHOLD_PCT + (intraday_slope * 0.1 if intraday_slope < 0 else 0)
                    
                    log_line.append(f"{data['name']}:{data['pct']:.1f}%")
                    
                    direction = None
                    if time.time() - rt['last_alert'] > cfg.AI_COOLDOWN_SECONDS:
                        if bias < thresh_buy: direction = "BUY"
                        elif bias > cfg.SELL_THRESHOLD_PCT: direction = "SELL"
                            
                        if direction:
                            print(f"\n{Fore.MAGENTA}⚡ {direction}: {data['name']} (Bias:{bias:.2f}%)")
                            rt['last_alert'] = time.time()
                            
                            # 获取资金流向 (仅触发时获取，节省资源)
                            funds = self.dm.fetch_fund_flow(code)
                            
                            d = {
                                'price': price, 'vwap': vwap, 'bias': bias,
                                'intraday_slope': intraday_slope, # 分时斜率
                                'ma_slope': rt['ma_slope'],       # 日线斜率
                                'threshold': thresh_buy if direction=="BUY" else cfg.SELL_THRESHOLD_PCT,
                                'volume': data['volume'], 'pct': data['pct']
                            }
                            
                            # 咨询 AI (带分批策略)
                            analysis = self.advisor.consult(data['name'], price, direction, d, indices, funds)
                            
                            # 弹窗
                            self.ui.show(data['name'], price, direction, analysis, d, indices, funds)
                            
                print(f"\r[{datetime.datetime.now().strftime('%H:%M:%S')}] {' '.join(log_line[:5])}...", end="")
                time.sleep(cfg.REALTIME_INTERVAL)
                
            except KeyboardInterrupt: break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                time.sleep(3)

if __name__ == "__main__":
    app = MonitorApp()
    app.run()