import akshare as ak
import pandas as pd
import numpy as np
import time
import random
import datetime
import os
import sys
import threading
import tkinter as tk
import warnings
import json
import logging
import re
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
# 确保 token_stock_list_config.py 在同级目录下
try:
    import token_stock_list_config as cfg
except ImportError:
    print("❌ 错误：未找到配置文件 'token_stock_list_config.py'。请先创建该文件并配置 API Key 和股票列表。")
    sys.exit(1)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import winsound
except ImportError:
    winsound = None

# ================= 0. 日志系统 =================
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

# ================= 1. 数据缓存管理器 (核心新增) =================
class DataManager:
    def __init__(self):
        self.cache_dir = os.path.join(os.getcwd(), "stock_data_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.latest_market_date = self._get_market_anchor_date()

    def _get_market_anchor_date(self):
        """
        获取市场基准日期（锚点策略）。
        请求上证指数的日线数据，取最新的一天作为“目前市场应有的最新日期”。
        如果本地个股日期 == 这个日期，则无需更新。
        """
        print("📅 正在校准市场交易日锚点...", end="")
        try:
            # 获取上证指数最近数据作为基准
            df_index = ak.stock_zh_index_daily(symbol="sh000001")
            last_date = pd.to_datetime(df_index['date']).max()
            print(f"基准日期: {last_date.strftime('%Y-%m-%d')}")
            return last_date
        except Exception as e:
            print(f"失败 ({e})。将默认强制更新所有数据。")
            return None

    def get_history_data(self, code):
        """
        智能获取历史数据：
        1. 检查本地是否有文件。
        2. 检查本地文件是否是最新的（对比锚点）。
        3. 如果是旧的或不存在，从 API 拉取并保存。
        """
        file_path = os.path.join(self.cache_dir, f"{code}.csv")
        need_update = True
        df = pd.DataFrame()

        # --- 步骤1: 尝试读取本地 ---
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                
                if not df.empty and self.latest_market_date is not None:
                    local_last_date = df['date'].max()
                    # 如果本地最新日期 >= 市场基准日期，说明数据已是最新，直接用
                    if local_last_date >= self.latest_market_date:
                        need_update = False
                        # print(f"[{code}] 本地缓存命中 ({local_last_date.date()})")
            except Exception as e:
                print(f"[{code}] 本地文件读取损坏，重新下载: {e}")
                need_update = True

        # --- 步骤2: 需要更新则请求接口 ---
        if need_update:
            try:
                # 设定下载范围：往前推400天到今天
                end_str = datetime.datetime.now().strftime("%Y%m%d")
                start_str = (datetime.datetime.now() - datetime.timedelta(days=400)).strftime("%Y%m%d")
                
                # 随机延迟防止封IP
                time.sleep(random.uniform(0.10, 0.15))
                
                df_new = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
                
                if df_new is not None and not df_new.empty:
                    # 标准化列名
                    df_new = AlphaFactors.process_columns_only(df_new) 
                    # 保存到本地
                    df_new.to_csv(file_path, index=False)
                    df = df_new
                    # print(f"[{code}] 数据已更新并缓存")
                else:
                    print(f"[{code}] 接口未返回数据，尝试使用旧数据")
            except Exception as e:
                print(f"❌ [{code}] 历史数据下载失败: {e}")
        
        return df

    @staticmethod
    def get_realtime_quotes():
        """
        获取实时行情，增加容错机制。
        此处主要依赖 akshare 的东财接口，这也是目前最稳定的免费源。
        """
        max_retries = 3
        for i in range(max_retries):
            try:
                # 备选接口1: 东方财富实时行情
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                if i == max_retries - 1:
                    logger.log_system(f"Realtime Data Error after {max_retries} retries: {e}")
                time.sleep(1)
        return pd.DataFrame()


# ================= 2. 特征工程 =================
class AlphaFactors:
    @staticmethod
    def process_columns_only(df):
        """仅处理列名，不进行技术指标计算，用于保存原始数据"""
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

    @staticmethod
    def process_data(df, code="未知"):
        """计算技术指标"""
        # 确保列名正确
        if 'date' not in df.columns:
            df = AlphaFactors.process_columns_only(df)
            
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

            # Vol Ratio (Simple history calc)
            df['Vol_MA5'] = df['volume'].rolling(5).mean()
            df['Vol_Ratio'] = df['volume'] / (df['Vol_MA5'] + 1e-9)

            df['Target_Low'] = (df['low'] - df['pre_close']) / df['pre_close'] * 100
            df['Target_High'] = (df['high'] - df['pre_close']) / df['pre_close'] * 100

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            if len(df) < 30: return pd.DataFrame()
            return df

        except Exception as e:
            print(f"❌ [{code}] 指标计算出错: {e}")
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
        # 使用 Config 中的参数
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
        
        curr = realtime_data['current']
        pct = realtime_data['pct']
        vwap = realtime_data['vwap']
        bias_vwap = realtime_data['vwap_bias']
        vol_ratio = realtime_data.get('vol_ratio', 1.0)
        
        vol_status = "缩量"
        if vol_ratio > 1.2: vol_status = "温和放量"
        if vol_ratio > 2.0: vol_status = "显著放量"
        
        action_hint = "考虑【卖出止盈】" if trigger_direction == "SELL" else "考虑【低吸买入】"
        
        prompt = f"""
        # Role: 资深A股日内操盘手
        
        # Task: 紧急交易判断 ({action_hint})
        标的：{name} ({code})
        
        # Real-time Status
        - 现价：{curr} (今日涨幅: {pct:.2f}%)
        - 均价(黄线)：{vwap:.2f}
        - **乖离率**：{bias_vwap:.2f}% (当前触发阈值)
        - **官方量比**：{vol_ratio:.2f} ({vol_status})
        
        # Trigger
        系统触发: {trigger_reason}
        方向倾向: {trigger_direction}
        
        # Context
        - 大盘情绪: {market_data['avg']:.2f}%
        - 技术面: {tech_summary}
        
        # Output Format (JSON ONLY)
        {{"action": "EXECUTE" | "WAIT", "reason": "简短理由", "score": 0-100, "suggested_price": float}}
        """
        
        logger.log_llm(f" >>> [SEND {code}] Type:{trigger_direction} Bias:{bias_vwap:.2f}%")

        def call_deepseek():
            try:
                res = self.ds_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={'type': 'json_object'}, temperature=0.1
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
            # 替换原有的直接 akshare 调用，改用 data_manager
            df = self.data_manager.get_history_data(self.code)
            
            if df is None or df.empty: return False

            df = AlphaFactors.process_data(df, self.code)
            if df.empty: return False
            
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
            
            if len(X) < 10: return False

            tf_model = self.build_transformer((cfg.SEQ_LEN, len(feat_cols)))
            tf_model.fit(X, [yl, yh], batch_size=32, epochs=5, verbose=0)
            return True
        except Exception as e:
            print(f"❌ [{self.code}] 训练报错: {e}")
            return False

# ================= UI. 控制台可视化增强 =================
class ConsoleUI:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def print_status(index, total, code, name, status, source="未知", error=None):
        """打印单只股票的初始化状态"""
        progress = f"[{index}/{total}]"
        
        if status == "SUCCESS":
            # 绿色显示成功
            color = ConsoleUI.OKGREEN
            icon = "✅"
            msg = f"{source}"
        elif status == "CACHE":
            # 青色显示缓存
            color = ConsoleUI.OKCYAN
            icon = "📂"
            msg = "本地缓存"
        else:
            # 红色显示失败
            color = ConsoleUI.FAIL
            icon = "❌"
            msg = f"失败: {str(error)[:30]}..."

        # 格式化输出
        print(f"{progress} {icon} {ConsoleUI.BOLD}{code}{ConsoleUI.ENDC} | {name:<6} | {color}{msg}{ConsoleUI.ENDC}")

    @staticmethod
    def print_heartbeat(count, market_sh, source_type, latency):
        """打印实时运行心跳"""
        now = datetime.datetime.now().strftime('%H:%M:%S')
        
        # 根据延迟变色
        lat_color = ConsoleUI.OKGREEN if latency < 1.0 else (ConsoleUI.WARNING if latency < 3.0 else ConsoleUI.FAIL)
        
        print(f"\r{ConsoleUI.OKBLUE}[{now}] 📡 运行中{ConsoleUI.ENDC} | "
              f"监控标的: {ConsoleUI.BOLD}{count}{ConsoleUI.ENDC}只 | "
              f"大盘: {market_sh:+.2f}% | "
              f"接口: {source_type} | "
              f"延迟: {lat_color}{latency:.2f}s{ConsoleUI.ENDC}", end="")

    @staticmethod
    def print_summary(success_list, fail_list):
        print("\n" + "="*50)
        print(f"🎉 初始化完成报告")
        print("="*50)
        print(f"🟢 成功加载: {len(success_list)} 只")
        print(f"🔴 加载失败: {len(fail_list)} 只")
        
        if fail_list:
            print("\n⚠️ 失败详情:")
            for item in fail_list:
                print(f"   - {item['code']}: {item['reason']}")
        print("="*50 + "\n")


# ================= 5. 弹窗 UI =================
alert_lock = threading.Lock()
def popup_alert(data):
    def _show():
        with alert_lock:
            if winsound: winsound.Beep(800, 300)
            root = tk.Tk()
            
            trigger_dir = data.get('direction', 'BUY')
            
            if trigger_dir == 'SELL':
                bg_col = '#660000'
                fg_title = '#FF5555'
                type_text = "卖出信号 (SELL)"
            else:
                bg_col = '#004d00'
                fg_title = '#55FF55'
                type_text = "买入信号 (BUY)"

            w, h = 800, 750
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
            
            sub_frame = tk.Frame(root, bg=bg_col)
            sub_frame.pack(pady=5)
            
            tk.Label(sub_frame, text=f"均价: {data['vwap']:.2f}",
                     font=("微软雅黑", 14), bg=bg_col, fg='#CCCCCC').pack(side='left', padx=15)
            
            tk.Label(sub_frame, text=f"乖离: {data['vwap_bias']:.2f}%",
                     font=("微软雅黑", 14, "bold"), bg=bg_col, fg='white').pack(side='left', padx=15)

            tk.Label(sub_frame, text=f"量比: {data.get('vol_ratio',0):.2f}",
                     font=("微软雅黑", 14), bg=bg_col, fg='cyan').pack(side='left', padx=15)
            
            tk.Label(root, text=f"触发原因: {data['reason']}", font=("微软雅黑", 12), bg=bg_col, fg='#DDDDDD').pack(pady=5)
            
            ai_frame = tk.LabelFrame(root, text="🧠 AI 军师团", font=("微软雅黑", 12), bg=bg_col, fg='white')
            ai_frame.pack(fill='both', expand=True, padx=20, pady=10)
            
            ds, qw = data['ds'], data['qw']
            
            tk.Label(ai_frame, text=f"[DeepSeek] {ds.get('action')} (信心:{ds.get('score')}) -> 挂单:{ds.get('suggested_price')}\nReason: {ds.get('reason')}",
                     font=("微软雅黑", 11), bg=bg_col, fg='cyan', wraplength=700, justify='left').pack(anchor='w', padx=10, pady=5)
            tk.Label(ai_frame, text="--------------------------------", bg=bg_col, fg='gray').pack()
            tk.Label(ai_frame, text=f"[Qwen] {qw.get('action')} (信心:{qw.get('score')}) -> 挂单:{qw.get('suggested_price')}\nReason: {qw.get('reason')}",
                     font=("微软雅黑", 11), bg=bg_col, fg='orange', wraplength=700, justify='left').pack(anchor='w', padx=10, pady=5)
            
            tk.Button(root, text="关闭窗口", font=("微软雅黑", 12), command=root.destroy).pack(pady=10)
            root.mainloop()
            
    threading.Thread(target=_show, daemon=True).start()


# ================= 6. 监控系统 (增强版) =================
class MonitorApp:
    def __init__(self):
        self.brains = {}
        self.advisor = DualAdvisor()
        self.data_manager = DataManager()
        self.market_data = {'sh':0.0, 'sz':0.0, 'cy':0.0, 'avg':0.0}
        self.fail_list = [] # 记录失败的股票
        
        # 启用Windows终端颜色支持
        os.system('color') 

    def init_models(self):
        print(f"\n🚀 启动交易系统...")
        print(f"📅 市场锚点日期: {self.data_manager.latest_market_date}")
        print("-" * 60)
        
        total = len(cfg.STOCK_LIST)
        success_list = []
        self.fail_list = []

        # 不使用多线程打印，避免控制台乱码，改用单线程顺序加载（虽然慢一点点，但看得清）
        # 如果追求速度，可以改回 ThreadPool，但控制台输出会乱
        for i, code in enumerate(cfg.STOCK_LIST, 1):
            try:
                # 1. 获取数据 (返回 df 和 来源标记)
                # 修改 DataManager.get_history_data 让它返回 source 标记
                # 这里我们假设 DataManager 还是原来的，我们通过逻辑判断来源
                
                df = self.data_manager.get_history_data(code)
                name = "未知" # 暂时没有名字，稍后获取实时数据时补全，或者这里调个接口
                
                if df is None or df.empty:
                    ConsoleUI.print_status(i, total, code, name, "FAIL", error="数据为空")
                    self.fail_list.append({'code': code, 'reason': '数据获取为空'})
                    continue

                # 2. 训练模型
                brain = EnsembleBrain(code, self.data_manager)
                # 这是一个hack，为了不重新下载，我们把刚才下的df传进去 (需要修改EnsembleBrain支持传入df，或者让它自己再读一遍缓存)
                # 这里简单起见，让brain自己去读缓存，肯定极快
                if brain.train():
                    self.brains[code] = brain
                    success_list.append(code)
                    
                    # 判断数据来源（根据文件修改时间）
                    file_path = os.path.join(self.data_manager.cache_dir, f"{code}.csv")
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    is_today = mtime.date() == datetime.datetime.now().date()
                    source_str = "⬇️ 网络下载" if is_today else "📂 本地历史"
                    
                    ConsoleUI.print_status(i, total, code, name, "SUCCESS", source=source_str)
                else:
                    ConsoleUI.print_status(i, total, code, name, "FAIL", error="模型训练未收敛")
                    self.fail_list.append({'code': code, 'reason': '模型训练失败'})
                    
            except Exception as e:
                ConsoleUI.print_status(i, total, code, "Error", "FAIL", error=e)
                self.fail_list.append({'code': code, 'reason': str(e)})

        ConsoleUI.print_summary(success_list, self.fail_list)
        time.sleep(3) # 让用户看一眼结果

    def get_market_data(self):
        try:
            df = ak.stock_zh_index_spot_sina()
            sh = float(df[df['代码']=='sh000001']['涨跌幅'].values[0])
            return {'sh': sh, 'avg': sh}
        except:
            return {'sh':0, 'avg':0}

    def run(self):
        if not self.brains:
            print(ConsoleUI.FAIL + "❌ 致命错误: 没有一只股票的模型加载成功，程序退出。" + ConsoleUI.ENDC)
            return

        print("\n📡 [实时监控模式启动] 按 Ctrl+C 停止")
        
        while True:
            t_start = time.time()
            try:
                # 1. 获取大盘
                self.market_data = self.get_market_data()
                
                # 2. 获取实时数据
                df_real = DataManager.get_realtime_quotes()
                
                if df_real.empty:
                    print(f"\r{ConsoleUI.WARNING}[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚠️ 实时接口无响应，重试中...{ConsoleUI.ENDC}", end="")
                    time.sleep(2)
                    continue

                # 3. 遍历策略
                valid_count = 0
                for code, brain in self.brains.items():
                    row = df_real[df_real['代码'] == code]
                    if row.empty: continue
                    valid_count += 1
                    
                    # --- 核心数据提取 ---
                    name = row['名称'].values[0]
                    curr = float(row['最新价'].values[0])
                    pre_close = float(row['昨收'].values[0])
                    pct = (curr - pre_close) / pre_close * 100
                    amount = float(row['成交额'].values[0])
                    volume_hand = float(row['成交量'].values[0])
                    
                    # 量比处理
                    real_vol_ratio = 1.0
                    if '量比' in row.columns:
                        try:
                            val = row['量比'].values[0]
                            if isinstance(val, (int, float)): real_vol_ratio = float(val)
                            elif str(val).replace('.', '', 1).isdigit(): real_vol_ratio = float(val)
                        except: pass
                    
                    # 均线与乖离
                    vwap = curr
                    if volume_hand > 0: vwap = amount / (volume_hand * 100)
                    bias_vwap = (curr - vwap) / vwap * 100
                    
                    # --- 触发判断 ---
                    trigger_direction = "HOLD"
                    trigger_reason = ""
                    
                    if bias_vwap < -cfg.VWAP_THRESHOLD_PCT:
                        trigger_direction = "BUY"
                        trigger_reason = f"超卖回归 (低于均线 {abs(bias_vwap):.2f}%)"
                    elif bias_vwap > cfg.VWAP_THRESHOLD_PCT:
                        trigger_direction = "SELL"
                        trigger_reason = f"超涨回调 (高于均线 {bias_vwap:.2f}%)"

                    # --- 触发处理 ---
                    if trigger_direction != "HOLD" and self.advisor.can_consult(code):
                        # 换行打印详细触发信息，以免被心跳覆盖
                        print(f"\n{ConsoleUI.HEADER}⚡ 触发信号: {name} ({code}) | {trigger_direction} | 乖离:{bias_vwap:.2f}%{ConsoleUI.ENDC}")
                        
                        realtime_data = {
                            'current': curr, 'pct': pct,
                            'vwap': vwap, 'vwap_bias': bias_vwap,
                            'vol_ratio': real_vol_ratio
                        }
                        
                        # 调用 AI
                        res_ds, res_qw = self.advisor.consult_joint_chiefs(
                            code, name, realtime_data, brain.latest_summary,
                            self.market_data, trigger_reason, trigger_direction
                        )
                        
                        should_popup = (
                            res_ds.get('action') == 'EXECUTE' or
                            res_qw.get('action') == 'EXECUTE' or
                            abs(bias_vwap) > (cfg.VWAP_THRESHOLD_PCT * 1.5)
                        )
                        
                        if should_popup:
                            popup_alert({
                                'code': code, 'name': name, 'direction': trigger_direction,
                                'curr': curr, 'pct': pct,
                                'vwap': vwap, 'vwap_bias': bias_vwap,
                                'vol_ratio': real_vol_ratio,
                                'reason': trigger_reason,
                                'ds': res_ds, 'qw': res_qw
                            })
                        else:
                            print(f"   -> AI建议观望: DS:{res_ds.get('reason')} ...")

                # 4. 打印心跳 (覆盖当前行)
                latency = time.time() - t_start
                ConsoleUI.print_heartbeat(valid_count, self.market_data['sh'], "东财/新浪", latency)
                
                time.sleep(cfg.REALTIME_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n👋 程序已停止")
                break
            except Exception as e:
                logger.log_system(f"Main Loop Error: {e}")
                print(f"\n{ConsoleUI.FAIL}❌ 运行时异常: {e}{ConsoleUI.ENDC}")
                time.sleep(3)


if __name__ == "__main__":
    app = MonitorApp()
    app.init_models()
    app.run()