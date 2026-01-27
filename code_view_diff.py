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
import json
import logging
import re
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler

# === 深度学习 & 机器学习 ===
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import RobustScaler

# === 大模型 API ===
from openai import OpenAI
import dashscope 

# === 导入本地配置 ===
try:
    import token_stock_list_config as cfg
except ImportError:
    print("❌ 错误：未找到配置文件 'token_stock_list_config.py'。")
    print("请先创建该文件并配置 API Key 和股票列表。")
    sys.exit(1)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import winsound
except ImportError:
    winsound = None

# ================= 0. 日志系统 & UI 工具 =================
class LogSystem:
    def __init__(self):
        self.today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        self.base_dir = os.path.join(os.getcwd(), "logs", self.today_str)
        if not os.path.exists(self.base_dir): os.makedirs(self.base_dir)
        self.simple_fmt = logging.Formatter('%(asctime)s - %(message)s')
        
        self.sys_logger = self._get_logger("system", "system.log", self.simple_fmt)
        self.llm_logger = self._get_logger("llm", "llm_dialog.log", self.simple_fmt)

    def _get_logger(self, name, filename, formatter):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_path = os.path.join(self.base_dir, filename)
            handler = RotatingFileHandler(file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def log_system(self, msg): self.sys_logger.info(msg)
    def log_llm(self, msg): self.llm_logger.info(msg)

logger = LogSystem()

class ConsoleUI:
    """控制台可视化增强工具"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    # Windows CMD 颜色支持适配
    os.system('color')

    @staticmethod
    def print_status(index, total, code, name, status, source="未知", error=None):
        """初始化阶段：打印单只股票状态"""
        progress = f"[{index}/{total}]"
        
        if status == "SUCCESS":
            color = ConsoleUI.OKGREEN
            icon = "✅"
            msg = f"{source}"
        elif status == "CACHE":
            color = ConsoleUI.OKCYAN
            icon = "📂"
            msg = "本地极速加载"
        else:
            color = ConsoleUI.FAIL
            icon = "❌"
            msg = f"失败: {str(error)[:20]}..."

        print(f"{progress} {icon} {ConsoleUI.BOLD}{code}{ConsoleUI.ENDC} | {name:<6} | {color}{msg}{ConsoleUI.ENDC}")

    @staticmethod
    def print_heartbeat(count, market_sh, latency, error_count=0):
        """运行时：打印底部动态心跳"""
        now = datetime.datetime.now().strftime('%H:%M:%S')
        lat_color = ConsoleUI.OKGREEN if latency < 1.0 else (ConsoleUI.WARNING if latency < 3.0 else ConsoleUI.FAIL)
        
        status_str = f"正常" if error_count == 0 else f"{ConsoleUI.FAIL}异常({error_count}){ConsoleUI.ENDC}"
        
        print(f"\r{ConsoleUI.OKBLUE}[{now}] 📡 {status_str}{ConsoleUI.ENDC} | "
              f"监控: {ConsoleUI.BOLD}{count}{ConsoleUI.ENDC}只 | "
              f"大盘: {market_sh:+.2f}% | "
              f"延迟: {lat_color}{latency:.2f}s{ConsoleUI.ENDC}    ", end="")

    @staticmethod
    def print_event(msg, level="INFO"):
        """运行时：在心跳上方打印事件"""
        # 先清除当前行（因为当前行有 \r 的心跳）
        print("\r" + " "*100 + "\r", end="")
        
        if level == "WARN":
            print(f"{ConsoleUI.WARNING}⚠️ {msg}{ConsoleUI.ENDC}")
        elif level == "ERROR":
            print(f"{ConsoleUI.FAIL}❌ {msg}{ConsoleUI.ENDC}")
        elif level == "SUCCESS":
            print(f"{ConsoleUI.HEADER}⚡ {msg}{ConsoleUI.ENDC}")
        else:
            print(f"ℹ️ {msg}")

# ================= 1. 数据管理 (核心修复) =================
class DataManager:
    def __init__(self):
        self.cache_dir = os.path.join(os.getcwd(), "stock_data_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # 获取“市场锚点日期”：即目前市场上最新的交易日
        self.market_anchor_date = self._get_market_anchor_date()

    def _get_market_anchor_date(self):
        """获取上证指数的最新日期，作为数据是否过期的判断基准"""
        print("📅 正在校准市场交易日锚点 (连接上证指数)... ", end="")
        try:
            # 尝试获取上证指数日线
            df_index = ak.stock_zh_index_daily(symbol="sh000001")
            last_date = pd.to_datetime(df_index['date']).max()
            print(f"{ConsoleUI.OKGREEN}成功: {last_date.strftime('%Y-%m-%d')}{ConsoleUI.ENDC}")
            return last_date
        except Exception as e:
            print(f"{ConsoleUI.WARNING}校准失败 ({e})。默认将尝试强制更新数据。{ConsoleUI.ENDC}")
            return None

    def _normalize_columns(self, df):
        """标准化列名，兼容不同接口的返回格式"""
        df.columns = df.columns.str.strip()
        rename_map = {
            "日期": "date", "date": "date",
            "开盘": "open", "open": "open", "开盘价": "open",
            "收盘": "close", "close": "close", "收盘价": "close", "最新价": "close",
            "最高": "high", "high": "high", "最高价": "high",
            "最低": "low", "low": "low", "最低价": "low",
            "成交量": "volume", "volume": "volume",
            "成交额": "amount", "amount": "amount"
        }
        df.rename(columns=rename_map, inplace=True)
        return df

    def get_history_data(self, code):
        """
        智能获取历史数据：
        1. 检查本地是否有缓存
        2. 检查缓存日期是否 >= 市场锚点日期
        3. 如果过期或不存在，则联网下载（支持备用接口）
        """
        file_path = os.path.join(self.cache_dir, f"{code}.csv")
        need_update = True
        df = pd.DataFrame()
        source_mark = "未知"

        # --- 步骤1: 检查本地 ---
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df = self._normalize_columns(df)
                df['date'] = pd.to_datetime(df['date'])
                
                if not df.empty and self.market_anchor_date is not None:
                    local_last = df['date'].max()
                    # 只有当本地日期 >= 市场最新日期，才算命中缓存
                    if local_last >= self.market_anchor_date:
                        need_update = False
                        source_mark = "CACHE"
            except Exception as e:
                need_update = True

        # --- 步骤2: 联网更新 (如果需要) ---
        if need_update:
            end_str = datetime.datetime.now().strftime("%Y%m%d")
            start_str = (datetime.datetime.now() - datetime.timedelta(days=400)).strftime("%Y%m%d")
            
            # 随机延迟防封
            time.sleep(random.uniform(0.1, 0.3))
            
            df_new = pd.DataFrame()
            try:
                # 优先：东方财富接口
                df_new = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
                source_mark = "网络(东财)"
            except Exception as e1:
                # 备用：新浪财经接口 (通常不报 SSL 错)
                try:
                    symbol_sina = f"sh{code}" if code.startswith('6') else f"sz{code}"
                    df_new = ak.stock_zh_a_daily(symbol=symbol_sina, start_date=start_str, end_date=end_str, adjust="qfq")
                    source_mark = "网络(新浪)"
                except Exception as e2:
                    source_mark = "失败"

            if not df_new.empty:
                df_new = self._normalize_columns(df_new)
                df_new.to_csv(file_path, index=False)
                df = df_new
            elif not df.empty:
                # 如果联网失败但本地有旧数据，勉强使用旧数据
                source_mark = "旧数据(更新失败)"
        
        return df, source_mark

    @staticmethod
    def get_realtime_quotes_safe():
        """获取实时行情，带简单的重试机制"""
        for _ in range(3):
            try:
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    return df
            except:
                time.sleep(1)
        return pd.DataFrame()

# ================= 2. 特征工程 =================
class AlphaFactors:
    @staticmethod
    def process_data(df, code="未知"):
        # 已经在 DataManager 中做了列名标准化，这里再做一次防御
        if 'date' not in df.columns: return pd.DataFrame()
        
        cols = ['open', 'close', 'high', 'low', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df.dropna(subset=['close', 'open'], inplace=True)
        if df.empty: return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', inplace=True)

        try:
            df['pre_close'] = df['close'].shift(1)
            df.dropna(subset=['pre_close'], inplace=True)

            # MA & Bias
            df['MA20'] = df['close'].rolling(20).mean()
            df['Bias20'] = (df['close'] - df['MA20']) / (df['MA20'] + 1e-5) * 100
            
            # ATR
            tr = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['pre_close']))
            df['ATR'] = tr.rolling(14).mean()
            df['ATR_Pct'] = df['ATR'] / df['pre_close'] * 100 

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = 100 - (100 / (1 + gain/(loss+1e-5)))

            # MACD
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            df['DIF'] = exp12 - exp26
            df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
            df['MACD'] = 2 * (df['DIF'] - df['DEA']) 

            # BOLL
            df['BOLL_MID'] = df['close'].rolling(20).mean()
            df['BOLL_STD'] = df['close'].rolling(20).std()
            df['BOLL_UP'] = df['BOLL_MID'] + 2 * df['BOLL_STD']
            df['BOLL_LOW'] = df['BOLL_MID'] - 2 * df['BOLL_STD']
            df['BOLL_POS'] = (df['close'] - df['BOLL_LOW']) / (df['BOLL_UP'] - df['BOLL_LOW'] + 1e-9)

            # Vol Ratio
            df['Vol_MA5'] = df['volume'].rolling(5).mean()
            df['Vol_Ratio'] = df['volume'] / (df['Vol_MA5'] + 1e-9)

            df['Target_Low'] = (df['low'] - df['pre_close']) / df['pre_close'] * 100
            df['Target_High'] = (df['high'] - df['pre_close']) / df['pre_close'] * 100

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            if len(df) < 30: return pd.DataFrame()
            return df

        except Exception as e:
            return pd.DataFrame()

    @staticmethod
    def get_latest_summary(df):
        if df.empty: return "数据不足"
        row = df.iloc[-1]
        trend = "多头" if row['close'] > row['MA20'] else "空头"
        rsi_status = "超买" if row['RSI'] > 70 else ("超卖" if row['RSI'] < 30 else "中性")
        return (
            f"【趋势】{trend} (Bias:{row['Bias20']:.2f}%)\n"
            f"【MACD】DIF:{row['DIF']:.2f} DEA:{row['DEA']:.2f}\n"
            f"【RSI】{row['RSI']:.1f} ({rsi_status})\n"
        )

# ================= 3. 双核军师 =================
class DualAdvisor:
    def __init__(self):
        self.ds_client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
        dashscope.api_key = cfg.DASHSCOPE_API_KEY
        self.last_consult_time = {}

    def can_consult(self, code):
        last = self.last_consult_time.get(code, 0)
        return (time.time() - last) > cfg.AI_COOLDOWN_SECONDS

    def _safe_parse_json(self, content_str, source="AI"):
        try:
            content_str = re.sub(r'```json|```', '', content_str).strip()
            if content_str.endswith("}") and not content_str.endswith("}}"): pass 
            data = json.loads(content_str)
            return data
        except Exception as e:
            logger.log_system(f"[{source}] JSON解析失败: {e}")
            return {"action": "WAIT", "reason": f"解析异常: {str(e)[:20]}", "score": 0}

    def consult_joint_chiefs(self, code, name, realtime_data, tech_summary, market_data, trigger_reason, trigger_direction):
        self.last_consult_time[code] = time.time()
        
        # ... (参数解包保持不变)
        curr = realtime_data['current']
        pct = realtime_data['pct']
        vwap = realtime_data['vwap']
        bias_vwap = realtime_data['vwap_bias']
        vol_ratio = realtime_data.get('vol_ratio', 1.0)
        main_force_net = realtime_data.get('main_force_net', 0.0) 
        trend_slope = realtime_data.get('trend_slope', 0.0)
        display_threshold = realtime_data.get('dynamic_threshold', -2.0)
        
        slope_sign = "+" if trend_slope > 0 else ""
        slope_str = f"{slope_sign}{trend_slope:.4f}"
        
        if trend_slope > 0.1: slope_desc = "📈 强劲拉升"
        elif trend_slope > 0: slope_desc = "↗️ 震荡向上"
        elif trend_slope < -0.1: slope_desc = "📉 快速杀跌"
        else: slope_desc = "➡️ 弱势震荡"

        vol_status = "缩量"
        if vol_ratio > 1.2: vol_status = "温和放量"
        if vol_ratio > 2.0: vol_status = "显著放量"
        
        action_hint = "考虑【卖出止盈】" if trigger_direction == "SELL" else "机会！考虑【低吸博反弹】"
        
        flow_desc = f"流入 {main_force_net/10000:.2f} 亿" if abs(main_force_net) > 10000 else f"流入 {main_force_net:.2f} 万"
        if main_force_net < 0:
            flow_desc = f"流出 {abs(main_force_net)/10000:.2f} 亿" if abs(main_force_net) > 10000 else f"流出 {abs(main_force_net):.2f} 万"

        prompt = f"""
        # Role: 资深技术派交易员 (擅长动态均线与急跌博弈)
        
        # Task: 交易决策 ({action_hint})
        标的：{name} ({code})
        
        # Real-time Status
        - 现价：{curr} (今日涨幅: {pct:.2f}%)
        - 均价(VWAP)：{vwap:.2f}
        - **当前乖离率**：{bias_vwap:.2f}% (价格相对于均线的距离)
        - **实时趋势斜率(线性回归)**：{slope_str} [{slope_desc}]
        - **当前参考阈值**：{display_threshold:.2f}% 
        - 量比：{vol_ratio:.2f} ({vol_status})
        - 主力资金：{flow_desc} 
        
        # Trigger
        系统触发: {trigger_reason}
        方向: {trigger_direction}
        
        # Output Format (JSON ONLY)
        {{"action": "EXECUTE" | "WAIT", "reason": "基于斜率{slope_str}和乖离率的分析", "score": 0-100, "suggested_price": float}}
        """
        
        logger.log_llm(f" >>> [SEND {code}] Slope:{slope_str} Bias:{bias_vwap:.2f}%")

        def call_deepseek():
            try:
                res = self.ds_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={'type': 'json_object'}, temperature=0.2
                )
                return self._safe_parse_json(res.choices[0].message.content, "DeepSeek")
            except Exception as e:
                return {"action": "WAIT", "reason": f"DS Error: {e}", "score": 0}

        def call_qwen():
            try:
                res = dashscope.Generation.call(
                    model='qwen-turbo',
                    messages=[{'role': 'user', 'content': prompt}],
                    result_format='message'
                )
                if res.status_code == 200:
                    return self._safe_parse_json(res.output.choices[0].message.content, "Qwen")
                return {"action": "WAIT", "reason": "Qwen Error", "score": 0}
            except Exception as e:
                return {"action": "WAIT", "reason": f"Qwen Error: {e}", "score": 0}

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(call_deepseek)
            f2 = executor.submit(call_qwen)
            return f1.result(), f2.result()

# ================= 4. 模型融合 =================
class EnsembleBrain:
    def __init__(self, code, data_manager):
        self.code = code
        self.data_manager = data_manager # 注入数据管理器
        self.scaler = RobustScaler()
        self.latest_summary = ""
        self.vol_ma5 = 0.0

    def build_transformer(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(32, activation="gelu")(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        out_l = layers.Dense(1, name="l")(x)
        out_h = layers.Dense(1, name="h")(x)
        model = keras.Model(inputs, [out_l, out_h])
        model.compile(optimizer='adam', loss=['mse', 'mse'], loss_weights=[0.5, 0.5])
        return model

    def train(self):
        try:
            # ⚠️ 关键修改：使用 DataManager 获取数据（含缓存逻辑）
            df, source = self.data_manager.get_history_data(self.code)
            
            if df is None or df.empty: 
                return False, "无数据"

            df = AlphaFactors.process_data(df, self.code)
            if df.empty: return False, "指标计算失败"
            
            self.latest_summary = AlphaFactors.get_latest_summary(df)
            
            vol_hist = df['volume'].shift(1).rolling(5).mean()
            self.vol_ma5 = vol_hist.iloc[-1] if not pd.isna(vol_hist.iloc[-1]) else 0
            
            feat_cols = ['Bias20', 'ATR_Pct', 'Vol_Ratio', 'RSI', 'MACD', 'BOLL_POS']
            data_X = self.scaler.fit_transform(df[feat_cols].values)
            data_y_l = df['Target_Low'].values
            data_y_h = df['Target_High'].values

            X, yl, yh = [], [], []
            for i in range(cfg.SEQ_LEN, len(data_X)):
                X.append(data_X[i-cfg.SEQ_LEN:i])
                yl.append(data_y_l[i])
                yh.append(data_y_h[i])
            X, yl, yh = np.array(X), np.array(yl), np.array(yh)
            
            if len(X) < 10: return False, "样本不足"

            tf_model = self.build_transformer((cfg.SEQ_LEN, len(feat_cols)))
            tf_model.fit(X, [yl, yh], batch_size=32, epochs=5, verbose=0)
            
            return True, source
        except Exception as e:
            return False, str(e)

# ================= 5. 弹窗 UI (保持不变) =================
alert_lock = threading.Lock()
def popup_alert(data):
    def _show():
        with alert_lock:
            if winsound: winsound.Beep(1000, 400)
            root = tk.Tk()
            
            trigger_dir = data.get('direction', 'BUY')
            if trigger_dir == 'SELL':
                bg_col = '#660000'
                fg_title = '#FF5555'
                type_text = "卖出信号 (SELL)"
            else:
                bg_col = '#004d00'
                fg_title = '#55FF55'
                type_text = "💰 动态抄底 (BUY)"

            w, h = 850, 850
            x, y = (root.winfo_screenwidth()-w)//2, (root.winfo_screenheight()-h)//2
            root.geometry(f"{w}x{h}+{x}+{y}")
            root.configure(bg=bg_col)
            root.attributes('-topmost', True)
            
            title_txt = f"⚡ {type_text}: {data['name']} ({data['code']})"
            tk.Label(root, text=title_txt, font=("黑体", 20, "bold"), bg=bg_col, fg=fg_title).pack(pady=10)
            
            core_frame = tk.Frame(root, bg=bg_col)
            core_frame.pack(pady=10)
            
            tk.Label(core_frame, text=f"现价: {data['curr']}", 
                     font=("Arial", 36, "bold"), bg=bg_col, fg='white').pack(side='left', padx=20)
            
            pct_val = data['pct']
            pct_col = '#FF5555' if pct_val > 0 else '#55FF55' 
            tk.Label(core_frame, text=f"{pct_val:+.2f}%", 
                     font=("Arial", 36, "bold"), bg=bg_col, fg=pct_col).pack(side='left', padx=20)
            
            # === 趋势斜率显示 ===
            sub_frame = tk.Frame(root, bg=bg_col)
            sub_frame.pack(pady=5)
            
            vwap_val = data['vwap']
            slope_val = data.get('trend_slope', 0)
            
            slope_col = 'cyan' if slope_val < -0.05 else ('#FF5555' if slope_val > 0 else '#CCCCCC')
            slope_sign_display = "+" if slope_val > 0 else ""
            
            tk.Label(sub_frame, text=f"均价: {vwap_val:.2f}", font=("微软雅黑", 14), bg=bg_col, fg='#CCCCCC').pack(side='left', padx=10)
            tk.Label(sub_frame, text=f"趋势斜率: {slope_sign_display}{slope_val:.4f}", font=("微软雅黑", 14, "bold"), bg=bg_col, fg=slope_col).pack(side='left', padx=10)
            
            # === 阈值对比 ===
            threshold_frame = tk.Frame(root, bg=bg_col)
            threshold_frame.pack(pady=5)
            
            bias = data['vwap_bias']
            dyn_thresh = data['dynamic_threshold']
            
            thresh_label_text = "止盈阈值:" if trigger_dir == 'SELL' else "抄底阈值:"
            
            tk.Label(threshold_frame, text=f"当前乖离: {bias:.2f}%", font=("微软雅黑", 16, "bold"), bg=bg_col, fg='yellow').pack(side='left', padx=10)
            tk.Label(threshold_frame, text=f"vs", font=("微软雅黑", 12), bg=bg_col, fg='white').pack(side='left', padx=5)
            tk.Label(threshold_frame, text=f"{thresh_label_text} {dyn_thresh:.2f}%", font=("微软雅黑", 16, "bold"), bg=bg_col, fg='white').pack(side='left', padx=10)

            flow_val = data.get('main_force_net', 0)
            flow_str = f"{flow_val/10000:.1f}亿" if abs(flow_val) > 10000 else f"{flow_val:.0f}万"
            tk.Label(root, text=f"主力资金: {flow_str}", font=("微软雅黑", 12), bg=bg_col, fg='#DDDDDD').pack(pady=5)
            
            tk.Label(root, text=f"⚠️ {data['reason']}", font=("微软雅黑", 12, "bold"), bg=bg_col, fg='yellow').pack(pady=10)
            
            ai_frame = tk.LabelFrame(root, text="🧠 AI 决策 (基于线性回归斜率)", font=("微软雅黑", 12), bg=bg_col, fg='white')
            ai_frame.pack(fill='both', expand=True, padx=20, pady=10)
            
            ds, qw = data['ds'], data['qw']
            
            tk.Label(ai_frame, text=f"[DeepSeek] {ds.get('action')} (信心:{ds.get('score')}) -> 挂单:{ds.get('suggested_price')}\nReason: {ds.get('reason')}", 
                     font=("微软雅黑", 11), bg=bg_col, fg='cyan', wraplength=750, justify='left').pack(anchor='w', padx=10, pady=5)
            tk.Label(ai_frame, text="--------------------------------", bg=bg_col, fg='gray').pack()
            tk.Label(ai_frame, text=f"[Qwen] {qw.get('action')} (信心:{qw.get('score')}) -> 挂单:{qw.get('suggested_price')}\nReason: {qw.get('reason')}", 
                     font=("微软雅黑", 11), bg=bg_col, fg='orange', wraplength=750, justify='left').pack(anchor='w', padx=10, pady=5)
            
            tk.Button(root, text="关闭窗口", font=("微软雅黑", 12), command=root.destroy).pack(pady=10)
            root.mainloop()
            
    threading.Thread(target=_show, daemon=True).start()

# ================= 6. 监控系统 (线性回归 + 可视化) =================
class MonitorApp:
    def __init__(self):
        self.brains = {}
        self.advisor = DualAdvisor()
        self.data_manager = DataManager() # 实例化数据管理器
        self.market_data = {'sh':0.0, 'sz':0.0, 'cy':0.0, 'avg':0.0}
        self.price_history = {} 

    def init_models(self):
        print(f"\n🚀 启动交易系统...")
        print(f"📡 市场锚点日期 (最新交易日): {self.data_manager.market_anchor_date}")
        print("-" * 70)
        
        cnt = 0
        total = len(cfg.STOCK_LIST)
        success_list = []
        fail_list = []
        
        # 使用单线程顺序加载，确保控制台输出清晰
        for i, code in enumerate(cfg.STOCK_LIST):
            # 获取名字（这里简单用空字符串占位，因为历史数据接口一般不带名字，实盘获取时再补）
            name = "Loading" 
            
            code_res, brain, source, err = self._train_one(code)
            
            if brain:
                self.brains[code] = brain
                self.price_history[code] = deque(maxlen=cfg.PRICE_WINDOW_SIZE)
                cnt += 1
                status = "CACHE" if "CACHE" in source else "SUCCESS"
                ConsoleUI.print_status(i+1, total, code, name, status, source=source)
                success_list.append(code)
            else:
                ConsoleUI.print_status(i+1, total, code, name, "FAIL", error=err)
                fail_list.append(code)
            
            # 如果是网络请求，稍微停顿一下防封
            if "网络" in source:
                time.sleep(1.5)

        print("\n" + "="*50)
        print(f"🎉 初始化完成: 成功 {cnt} / 失败 {len(fail_list)}")
        print("="*50 + "\n")

    def _train_one(self, code):
        brain = EnsembleBrain(code, self.data_manager)
        success, msg = brain.train()
        if success:
            return code, brain, msg, None
        return code, None, "失败", msg

    def get_market_data(self):
        try:
            df = ak.stock_zh_index_spot_sina()
            sh = float(df[df['代码']=='sh000001']['涨跌幅'].values[0])
            return {'sh': sh, 'avg': sh}
        except:
            return {'sh':0, 'avg':0}

    def run(self):
        if not self.brains:
            print(ConsoleUI.FAIL + "❌ 没有可用的模型，程序退出。" + ConsoleUI.ENDC)
            return

        print(f"📡 [分时战法] 监控已启动 (间隔 {cfg.REALTIME_INTERVAL}s)...")
        
        while True:
            t_start = time.time()
            error_count = 0
            
            try:
                # 1. 获取大盘
                self.market_data = self.get_market_data()
                
                # 2. 获取实时数据 (带重试)
                df_real = DataManager.get_realtime_quotes_safe()
                
                if df_real.empty:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    ConsoleUI.print_event("实时行情获取失败，正在重试...", "WARN")
                    time.sleep(2)
                    continue
                
                # 3. 获取资金流向 (可选)
                try:
                    df_flow = ak.stock_individual_fund_flow_rank(indicator="今日")
                    flow_map = dict(zip(df_flow['代码'], df_flow['今日主力净流入-净额']))
                except:
                    flow_map = {}
                
                # 4. 遍历股票
                for code, brain in self.brains.items():
                    row = df_real[df_real['代码'] == code]
                    if row.empty: continue
                    
                    try:
                        name = row['名称'].values[0]
                        curr = float(row['最新价'].values[0])
                        pre_close = float(row['昨收'].values[0])
                        pct = (curr - pre_close) / pre_close * 100
                        
                        amount = float(row['成交额'].values[0])
                        volume_hand = float(row['成交量'].values[0]) 
                        
                        main_force_money = flow_map.get(code, 0) 
                        main_force_wan = main_force_money / 10000.0 
                        
                        real_vol_ratio = 1.0
                        if '量比' in row.columns:
                            try:
                                val = row['量比'].values[0]
                                real_vol_ratio = float(val) if str(val).replace('.', '', 1).isdigit() else 1.0
                            except: pass
                        
                        vwap = curr 
                        if volume_hand > 0:
                            vwap = amount / (volume_hand * 100)
                        
                        # === 记录历史价格计算斜率 ===
                        self.price_history[code].append(curr)
                        
                        trend_slope = 0.0
                        if len(self.price_history[code]) > 5:
                            y = np.array(self.price_history[code])
                            x = np.arange(len(y))
                            slope, intercept = np.polyfit(x, y, 1)
                            trend_slope = (slope / y.mean()) * 100

                        # === 动态阈值计算 ===
                        slope_penalty = 0.0
                        if trend_slope < 0:
                            slope_penalty = trend_slope * cfg.SLOPE_FACTOR
                        
                        buy_threshold = -cfg.BASE_THRESHOLD_PCT + slope_penalty
                        sell_threshold = cfg.SELL_THRESHOLD_PCT
                        
                        bias_vwap = (curr - vwap) / vwap * 100
                        
                        trigger_type = None
                        trigger_reason = ""
                        trigger_direction = "HOLD" 
                        display_threshold = buy_threshold 
                        
                        # === 触发逻辑 ===
                        if bias_vwap < buy_threshold:
                            trigger_type = "DYNAMIC_BUY"
                            trigger_direction = "BUY"
                            display_threshold = buy_threshold 
                            trigger_reason = f"趋势斜率{trend_slope:.3f}致门槛降至{buy_threshold:.2f}%"

                        elif bias_vwap > sell_threshold:
                            trigger_type = "PROFIT_TAKE"
                            trigger_direction = "SELL"
                            display_threshold = sell_threshold 
                            trigger_reason = f"股价超涨{bias_vwap:.2f}% (止盈线 {sell_threshold}%)"

                        if trigger_type and self.advisor.can_consult(code):
                            # 打印事件到控制台
                            ConsoleUI.print_event(f"[{name}] 触发 {trigger_direction} | 斜率:{trend_slope:.3f} | 乖离:{bias_vwap:.2f}%", "SUCCESS")
                            
                            realtime_data = {
                                'current': curr, 'pct': pct, 
                                'vwap': vwap, 'vwap_bias': bias_vwap,
                                'vol_ratio': real_vol_ratio,
                                'main_force_net': main_force_wan,
                                'trend_slope': trend_slope,  
                                'dynamic_threshold': display_threshold 
                            }
                            
                            res_ds, res_qw = self.advisor.consult_joint_chiefs(
                                code, name, realtime_data, brain.latest_summary, 
                                self.market_data, trigger_reason, trigger_direction
                            )
                            
                            should_popup = (
                                res_ds.get('action') == 'EXECUTE' or 
                                res_qw.get('action') == 'EXECUTE'
                            )
                            
                            if should_popup:
                                popup_alert({
                                    'code': code, 'name': name, 'direction': trigger_direction,
                                    'curr': curr, 'pct': pct,
                                    'vwap': vwap, 'vwap_bias': bias_vwap,
                                    'vol_ratio': real_vol_ratio,
                                    'main_force_net': main_force_wan,
                                    'reason': trigger_reason,
                                    'ds': res_ds, 'qw': res_qw,
                                    'trend_slope': trend_slope,
                                    'dynamic_threshold': display_threshold
                                })
                            else:
                                ConsoleUI.print_event(f"[{name}] AI 建议观望: {res_ds.get('reason')}", "INFO")

                    except Exception as inner_e:
                        error_count += 1
                        continue

                # 5. 打印心跳
                latency = time.time() - t_start
                ConsoleUI.print_heartbeat(len(self.brains), self.market_data['sh'], latency, error_count)
                
                time.sleep(cfg.REALTIME_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n👋 程序已停止")
                break
            except Exception as e:
                logger.log_system(f"Main Loop Error: {e}")
                ConsoleUI.print_event(f"循环异常: {e}", "ERROR")
                time.sleep(3)

if __name__ == "__main__":
    if "sk-" not in cfg.DEEPSEEK_API_KEY:
        print("❌ 请先在 Config 中填入 API Key")
    else:
        app = MonitorApp()
        app.init_models()
        app.run()