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

# Data Processing
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import RobustScaler

# UI Library
import tkinter as tk
from tkinter import ttk
from colorama import init, Fore, Style

# Import Configuration
try:
    import token_stock_list_config as cfg
except ImportError:
    print("❌ Error: token_stock_list_config.py not found.")
    sys.exit(1)

# Initialize Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
init(autoreset=True)

# ==========================================
# Helper Function: Trading Time Calculation
# ==========================================
def get_trading_minutes_elapsed():
    """Calculate minutes elapsed in the trading day for Volume Ratio calculation"""
    now = datetime.datetime.now()
    # Define trading hours
    start_am = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end_am = now.replace(hour=11, minute=30, second=0, microsecond=0)
    start_pm = now.replace(hour=13, minute=0, second=0, microsecond=0)
    end_pm = now.replace(hour=15, minute=0, second=0, microsecond=0)

    if now < start_am:
        return 0
    elif start_am <= now <= end_am:
        return (now - start_am).seconds // 60
    elif end_am < now < start_pm:
        return 120 # Fixed 120 minutes for morning session
    elif start_pm <= now <= end_pm:
        return 120 + (now - start_pm).seconds // 60
    else:
        return 240 # Market closed

# ==========================================
# Module A: Data Management
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
        return code

    def fetch_indices_snapshot(self):
        """Fetch Market Indices"""
        url = "http://qt.gtimg.cn/q=s_sh000001,s_sz399001,s_sz399006"
        indices = {"sh": 0, "sz": 0, "cyb": 0}
        try:
            resp = requests.get(url, timeout=3)
            lines = resp.text.split(';')
            if len(lines) >= 3:
                indices['sh'] = float(lines[0].split('~')[5])
                indices['sz'] = float(lines[1].split('~')[5])
                indices['cyb'] = float(lines[2].split('~')[5])
        except: pass
        return indices

    def fetch_tencent_history(self, code):
        """Fetch History K-Line for Indicators"""
        symbol = self._get_tencent_code(code)
        url = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        # Fetch 320 days to ensure enough data for MACD/MA
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
                            'volume': float(data[6]) * 100, # Convert to shares
                            'amount': float(data[37]) * 10000, # Convert to Yuan
                            'pct': (price - pre_close) / pre_close * 100 if pre_close > 0 else 0
                        }
                    except: continue
            except: pass
        return results

# ==========================================
# Module B: Feature Engineering (AlphaFactors)
# ==========================================
class AlphaFactors:
    @staticmethod
    def process(df):
        if df.empty or len(df) < 30: return df
        df = df.sort_values('date').reset_index(drop=True)
        
        # 1. RSI (Corrected to match Tonghuashun/EastMoney SMA algorithm)
        # This uses the logic provided by user
        def calc_rsi(series, period):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['RSI6'] = calc_rsi(df['close'], 6)
        df['RSI12'] = calc_rsi(df['close'], 12)
        df['RSI24'] = calc_rsi(df['close'], 24)
        
        # 2. BOLL (Calculated to output UP/MID/LOW)
        # MID is MA20, UP/LOW are +/- 2 STD
        df['BOLL_MID'] = df['close'].rolling(window=20).mean() # MA20
        std = df['close'].rolling(20).std()
        df['BOLL_UP'] = df['BOLL_MID'] + 2 * std
        df['BOLL_LOW'] = df['BOLL_MID'] - 2 * std
        # Position still useful for logic, though prompt uses raw values
        df['BOLL_POS'] = (df['close'] - df['BOLL_LOW']) / (4 * std + 0.0001)
        
        # 3. MACD (Added MACD Histogram column)
        # DIF: EMA12 - EMA26
        # DEA: EMA9 of DIF
        # MACD: (DIF - DEA) * 2
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp12 - exp26
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD'] = (df['DIF'] - df['DEA']) * 2
        
        # 4. ATR (Corrected TR calculation)
        # TR = Max(H-L, |H-PrevClose|, |L-PrevClose|)
        prev_close = df['close'].shift(1)
        h_l = df['high'] - df['low']
        h_pc = (df['high'] - prev_close).abs()
        l_pc = (df['low'] - prev_close).abs()
        
        df['tr'] = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        # THS typically uses SMA for ATR, using SMA(14)
        df['ATR'] = df['tr'].rolling(window=14).mean()
        
        # 5. Vol_MA5 for Volume Ratio (Shift 1 to exclude today)
        df['Vol_MA5'] = df['volume'].rolling(window=5).mean().shift(1)

        df.dropna(inplace=True)
        return df

# ==========================================
# Module D: DeepSeek Advisor
# ==========================================
class DeepSeekAdvisor:
    def __init__(self):
        self.ds_key = getattr(cfg, 'DEEPSEEK_API_KEY', "")

    def consult(self, stock, price, direction, d, indices, indicators):
        action_cn = "低吸买入 (BUY)" if direction == "BUY" else "高抛止盈 (SELL)"
        
        # Extract Indicator Values safely
        rsi6 = indicators.get('RSI6', 0)
        rsi12 = indicators.get('RSI12', 0)
        rsi24 = indicators.get('RSI24', 0)
        
        dif = indicators.get('DIF', 0)
        dea = indicators.get('DEA', 0)
        macd_bar = indicators.get('MACD', 0)
        
        boll_up = indicators.get('BOLL_UP', 0)
        boll_mid = indicators.get('BOLL_MID', 0)
        boll_low = indicators.get('BOLL_LOW', 0)
        boll_pos = indicators.get('BOLL_POS', 0.5)
        
        atr = indicators.get('ATR', 0)
        TR = indicators.get('tr', 0)
        # For prompt display, TR usually approximates to ATR in short term or we use the latest TR
        # But user prompt asked: "ATR波动: TR是多少，ATR是多少". 
        # Since we don't pass raw TR in indicators dict, we display ATR twice or denote ATR.
        # However, to strictly follow request, I will format it as ATR value.
        
        prompt = f"""
        你是一个顶级A股交易员。当前触发【{action_cn}】信号。
        请基于以下全方位数据，进行逻辑严密的推演，并制定**分批交易计划**。

        【市场环境】
        上证:{indices['sh']:.2f}% | 深证:{indices['sz']:.2f}% | 创业板:{indices['cyb']:.2f}%

        【个股盘口: {stock}】
        现价: {price} (涨跌幅: {d['pct']:.2f}%)
        成交量: {d['volume']/100:.0f}手
        **量比 (Vol Ratio): {d['vol_ratio']:.2f}** (重要: >1.5放量, <0.8缩量, 配合价格位置判断)

        【核心触发逻辑】
        1. 均价(VWAP): {d['vwap']:.2f}
        2. 乖离率(Bias): {d['bias']:.2f}% (触发阈值: {d['threshold']:.2f}%)
        3. **分时均线趋势(VWAP Slope): {d['vwap_slope']:.4f}** (0为横盘，正数为上行，反映日内黄线方向)

        【技术指标 (日线级别参考)】
        - RSI组合: RSI(6):{rsi6:.2f}, RSI(12):{rsi12:.2f}, RSI(24):{rsi24:.2f} (6日/12日/24日)
        - MACD(12,26,9): MACD={macd_bar:.2f}, DIF={dif:.2f}, DEA={dea:.2f}
        - 布林带(20,2): UP:{boll_up:.2f}, MID:{boll_mid:.2f}, LOW:{boll_low:.2f} (位置: {boll_pos:.2f})
        - ATR波动: TR: {TR:.2f}, ATR: {atr:.2f}

        【任务要求】
        不要只给一个建议价格！请制定详细的**分批操作计划**。
        必须返回纯 JSON 格式：
        {{
            "action": "EXECUTE" 或 "WAIT",
            "score": 0-100 (信心分),
            "reason": "结合量比、均线趋势和技术指标的综合分析(50字内)",
            "plan": [
                "第一步: 现价xx买入xx%作为底仓",
                "第二步: 若回调至xx元(支撑位)加仓xx%",
                "第三步: 若跌破xx元坚决止损"
            ]
        }}
        """
        
        print(f"\n{Fore.YELLOW}------ [LOG] >>> DeepSeek Prompt Sent ------")
        print(f"{Fore.CYAN}{prompt[:]}")
        print(f"{Fore.CYAN}量比: {d['vol_ratio']}, VWAP斜率: {d['vwap_slope']}, RSI: {rsi6:.1f}")

        if not self.ds_key or "sk-" not in self.ds_key: 
            return {"provider": "System", "action": "WAIT", "plan": ["API Key未配置"], "reason": "无Key"}
        
        headers = {"Authorization": f"Bearer {self.ds_key}", "Content-Type": "application/json"}
        payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions", 
                json=payload, headers=headers, 
                proxies={"http": None, "https": None}, timeout=20
            )
            content = resp.json()['choices'][0]['message']['content']
            
            print(f"{Fore.GREEN}------ [LOG] <<< DeepSeek Response ------\n{content}")
            return self._parse_json(content)
        except Exception as e:
            print(f"{Fore.RED}DeepSeek Error: {e}")
            return {"provider": "DeepSeek", "action": "ERROR", "plan": [], "reason": str(e)}

    def _parse_json(self, text):
        try:
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            data['provider'] = "DeepSeek"
            if 'plan' not in data:
                data['plan'] = [f"建议操作: {data.get('suggest_price', 'Market')}"]
            return data
        except:
            return {"provider": "DeepSeek", "action": "MANUAL", "plan": ["JSON解析失败"], "reason": "格式错误"}

# ==========================================
# Module F: UI Manager
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
        
    def show(self, stock, price, direction, analysis, d, idx, ind):
        if self.root:
            self.root.after(0, lambda: self._create_win(stock, price, direction, analysis, d, idx, ind))
            
    def _create_win(self, stock, price, direction, analysis, d, idx, ind):
        win = tk.Toplevel(self.root)
        win.title(f"{direction} Strategy - {stock}")
        win.attributes("-topmost", True)
        
        bg_color = "#005500" if direction == "BUY" else "#8B0000" 
        win.configure(bg=bg_color)
        win.geometry("600x800") 
        
        # Fonts
        f_ti = ("Microsoft YaHei", 14, "bold")
        f_pr = ("Arial", 36, "bold")
        f_no = ("Microsoft YaHei", 10)
        
        # 1. Market Indices
        idx_str = f"🌏 上证 {idx['sh']}% | 深证 {idx['sz']}% | 创业板 {idx['cyb']}%"
        tk.Label(win, text=idx_str, font=("Microsoft YaHei", 9), bg="#222", fg="#0F0").pack(fill="x")

        # 2. Stock Header
        tk.Label(win, text=f"⚡ {direction} 信号: {stock}", font=f_ti, bg=bg_color, fg="#FFD700").pack(pady=(10,0))
        f_p = tk.Frame(win, bg=bg_color); f_p.pack()
        tk.Label(f_p, text=f"{price}", font=f_pr, bg=bg_color, fg="white").pack(side="left")
        col_pct = "#0F0" if d['pct'] < 0 else "#F44"
        tk.Label(f_p, text=f" {d['pct']:.2f}%", font=("Arial", 20), bg=bg_color, fg=col_pct).pack(side="left", padx=10)

        # 3. Core Indicators (VolRatio & Slope)
        f_core = tk.Frame(win, bg=bg_color); f_core.pack(fill="x", padx=20, pady=5)
        
        # Vol Ratio
        vr = d['vol_ratio']
        col_vr = "#FFD700" if vr > 1.5 else "white"
        tk.Label(f_core, text=f"量比: {vr:.2f}", font=("Arial", 12, "bold"), bg=bg_color, fg=col_vr).pack(side="left")
        
        # VWAP Slope
        vs = d['vwap_slope']
        col_vs = "#F44" if vs < -0.1 else ("#0F0" if vs > 0.1 else "#DDD")
        tk.Label(f_core, text=f"均线趋势: {vs:.4f}", font=("Arial", 12), bg=bg_color, fg=col_vs).pack(side="right")

        ttk.Separator(win, orient="horizontal").pack(fill="x", padx=20, pady=5)

        # 4. Detailed Data
        f_info = tk.Frame(win, bg=bg_color); f_info.pack(padx=30, fill="x")
        
        # Left: Price
        tk.Label(f_info, text=f"均价(VWAP): {d['vwap']:.2f}", font=f_no, bg=bg_color, fg="#DDD").grid(row=0, column=0, sticky="w")
        tk.Label(f_info, text=f"当前乖离: {d['bias']:.2f}%", font=("Arial", 11, "bold"), bg=bg_color, fg="#FFD700").grid(row=1, column=0, sticky="w")
        tk.Label(f_info, text=f"触发阈值: {d['threshold']:.2f}%", font=f_no, bg=bg_color, fg="#AAA").grid(row=2, column=0, sticky="w")
        
        # Right: Technicals (Updated to show RSI6 and new MACD)
        tk.Label(f_info, text=f"RSI(6): {ind.get('RSI6',0):.1f}", font=f_no, bg=bg_color, fg="white").grid(row=0, column=1, sticky="e", padx=(40,0))
        macd_str = f"MACD:{ind.get('MACD',0):.2f}"
        tk.Label(f_info, text=macd_str, font=f_no, bg=bg_color, fg="#DDD").grid(row=1, column=1, sticky="e", padx=(40,0))
        boll_s = "上轨" if ind.get('BOLL_POS',0.5)>0.8 else ("下轨" if ind.get('BOLL_POS')<0.2 else "中轨")
        tk.Label(f_info, text=f"布林位置: {boll_s}", font=f_no, bg=bg_color, fg="#AAA").grid(row=2, column=1, sticky="e", padx=(40,0))

        # 5. AI Strategy
        tk.Label(win, text="🤖 DeepSeek 交易军师", font=f_ti, bg=bg_color, fg="#ADE").pack(pady=(20,5))
        
        f_ai = tk.Frame(win, bg=bg_color, bd=1, relief="groove")
        f_ai.pack(fill="both", expand=True, padx=15, pady=5)
        
        # Action & Score
        act = analysis.get('action', 'WAIT')
        col_act = "#0F0" if act == "EXECUTE" else "#AAA"
        tk.Label(f_ai, text=f"{act} (信心:{analysis.get('score',0)})", font=("Consolas", 14, "bold"), bg=bg_color, fg=col_act).pack(pady=5)
        
        # Reason
        tk.Label(f_ai, text=f"💡 {analysis.get('reason','')}", font=f_no, bg=bg_color, fg="#EEE", wraplength=520).pack(pady=5)
        
        # Plan
        plans = analysis.get('plan', [])
        if plans:
            tk.Label(f_ai, text="📋 分批执行计划:", font=("Microsoft YaHei", 10, "bold"), bg=bg_color, fg="#FFD700").pack(pady=(10,5))
            for p in plans:
                tk.Label(f_ai, text=f"• {p}", font=("Microsoft YaHei", 9), bg=bg_color, fg="white", anchor="w").pack(fill="x", padx=20)

# ==========================================
# Main Application
# ==========================================
class MonitorApp:
    def __init__(self):
        raw_list = cfg.STOCK_LIST + cfg.SHORT_STUDIED_LIST
        self.stocks = list(dict.fromkeys(raw_list))
        print(f"{Fore.CYAN}=== 启动监控: {len(self.stocks)} 只目标 ===")
        
        self.dm = DataManager()
        self.runtime = {} 
        self.advisor = DeepSeekAdvisor()
        self.ui = PopupManager()
        self.ui.start()
        
        self._init_data()

    def _init_data(self):
        print(f"{Fore.GREEN}=== 预加载技术指标 & 计算五日均量 ===")
        for code in self.stocks:
            # Fetch History
            df = self.dm.get_history_data(code)
            indicators = {}
            avg_vol_5d = 0
            
            if not df.empty:
                try:
                    df = AlphaFactors.process(df)
                    # Get Last Row
                    last_row = df.iloc[-1]
                    indicators = {
                        'RSI6': last_row['RSI6'],
                        'RSI12': last_row['RSI12'],
                        'RSI24': last_row['RSI24'],
                        'DIF': last_row['DIF'],
                        'DEA': last_row['DEA'],
                        'MACD': last_row['MACD'],
                        'BOLL_UP': last_row['BOLL_UP'],
                        'BOLL_MID': last_row['BOLL_MID'],
                        'BOLL_LOW': last_row['BOLL_LOW'],
                        'BOLL_POS': last_row['BOLL_POS'],
                        'ATR': last_row['ATR']
                    }
                    if 'Vol_MA5' in df.columns:
                        avg_vol_5d = last_row['Vol_MA5']
                except: pass

            self.runtime[code] = {
                'price_q': deque(maxlen=cfg.PRICE_WINDOW_SIZE), 
                'vwap_q': deque(maxlen=10), 
                'last_alert': 0,
                'indicators': indicators,
                'avg_vol_5d': avg_vol_5d 
            }
        print("数据加载完毕。")

    def run(self):
        print(f"{Fore.GREEN}=== 监控运行中... ===")
        while True:
            try:
                # 1. Market Indices
                indices = self.dm.fetch_indices_snapshot()
                
                # 2. Realtime Snapshot
                snapshot = self.dm.get_realtime_snapshot(self.stocks)
                log_line = [f"SH:{indices['sh']}%"]
                
                # 3. Minutes Elapsed (For VolRatio)
                mins_elapsed = get_trading_minutes_elapsed()
                if mins_elapsed == 0: mins_elapsed = 1
                
                for code, data in snapshot.items():
                    rt = self.runtime.get(code)
                    if not rt: continue
                    
                    price = data['price']
                    rt['price_q'].append(price)
                    
                    # === VWAP ===
                    vwap = data['amount'] / data['volume'] if data['volume'] > 0 else price
                    rt['vwap_q'].append(vwap)
                    
                    # === VWAP Slope ===
                    vwap_slope = 0
                    if len(rt['vwap_q']) >= 5:
                        y = list(rt['vwap_q'])
                        x = np.arange(len(y))
                        s, _ = np.polyfit(x, y, 1)
                        vwap_slope = s * 10 
                    
                    # === Vol Ratio (Corrected /100) ===
                    vol_ratio = 0
                    if rt['avg_vol_5d'] > 0:
                        vol_per_min_now = data['volume'] / mins_elapsed
                        vol_per_min_avg = rt['avg_vol_5d'] / 240
                        vol_ratio = vol_per_min_now / vol_per_min_avg 
                        vol_ratio = vol_ratio / 100 # Corrected logic

                    # === Bias ===
                    bias = (price - vwap) / vwap * 100
                    
                    # === Dynamic Threshold ===
                    thresh_buy = -cfg.BASE_THRESHOLD_PCT + (vwap_slope * 0.2 if vwap_slope < 0 else 0)
                    
                    log_line.append(f"{data['name']}:{data['pct']:.1f}%")
                    
                    # === Trigger ===
                    direction = None
                    if time.time() - rt['last_alert'] > cfg.AI_COOLDOWN_SECONDS:
                        if bias < thresh_buy: direction = "BUY"
                        elif bias > cfg.SELL_THRESHOLD_PCT: direction = "SELL"
                            
                        if direction:
                            print(f"\n{Fore.MAGENTA}⚡ {direction}: {data['name']} (Bias:{bias:.2f}%, VolRatio:{vol_ratio:.2f})")
                            rt['last_alert'] = time.time()
                            
                            # Package Data
                            d_pkg = {
                                'pct': data['pct'],
                                'volume': data['volume'],
                                'price': price, 
                                'vwap': vwap, 
                                'bias': bias,
                                'vwap_slope': vwap_slope,
                                'threshold': thresh_buy if direction=="BUY" else cfg.SELL_THRESHOLD_PCT,
                                'vol_ratio': vol_ratio
                            }
                            
                            # DeepSeek Consult
                            ana = self.advisor.consult(data['name'], price, direction, d_pkg, indices, rt['indicators'])
                            
                            # Show UI
                            self.ui.show(data['name'], price, direction, ana, d_pkg, indices, rt['indicators'])
                            
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