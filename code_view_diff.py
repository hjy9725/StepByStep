为什么全是显示买入，有的明显是高于均线很多了，需要卖出了，因为代码不一定连接了我的持仓情况，所以有可能不知道我到底有没有底仓，但是这个不是代码需要考虑的问题，只需要考虑低于均线一定情况提示买入，高于均线一定情况提示卖出。改一下，返回完整代码。
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

# ================= 1. 配置中心 =================
class Config:
    # ⚠️⚠️⚠️ 请在此处填入你的 API Key ⚠️⚠️⚠️
    DEEPSEEK_API_KEY = "sk-" 
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DASHSCOPE_API_KEY = "sk-" 
    
    # --- 🎯 目标股票池 ---
    STOCK_LIST = [
#     {
#   "有色金属板块": 
    "001337", #四川黄金,
    "002716", #湖南白银,
    "603799", #华友钴业,
    "600362", #江西铜业,
    "002460", #赣锋锂业,
    "600961", #株冶集团,
    "000657", #中钨高新,
    "300618", #寒锐钴业,
    "600547", #山东黄金,
    "600988", #赤峰黄金,
    "601069", #西部黄金,
    "000630", #铜陵有色,
    "002240", #盛新锂能,
    "000831", #中国稀土,
    "601212", #白银有色,
    "600489", #中金黄金,
    "601899", #紫金矿业,
    "000426" #兴业银锡
#   ],
#   "科技板块": [
    "601208", #东材科技,
    "002759", #天际股份,
    "000681", #视觉中国,
    "002121", #科陆电子,
    "002837", #英维克,
    "002518", #科士达,
    "002407", #多氟多,
    "002466", #天齐锂业,
    "603090", #宏盛股份,
    "002409", #雅克科技,
    "002709", #天赐材料,
    "000409", #云鼎科技,
    "600183", #生益科技,
    "002050", #三花智控,
    "002463", #沪电股份,
    "600089", #特变电工,
    "601138", #工业富联,
    "603986", #兆易创新,
    "600895", #张江高科,
    "002851", #麦格米特,
    "000603", #盛达资源,
    "600730", #中国高科,
    "603119", #浙江荣泰,
    "605598", #上海港湾,
    "002027", #分众传媒,
    "002261", #拓维信息,
    "002792", #通宇通讯,
    "002202" #金风科技
#   ],
#   "航天军工板块": [
    "600501", #航天晨光,
    "600855", #航天长峰,
    "000901", #航天科技,
    "600343", #航天动力,
    "600877", #电科芯片,
    "600879", #航天电子,
    "000547", #航天发展,
    "002255" #海陆重工
  ]
    # 确保只保留数字代码
    
    # --- ⚙️ 策略参数 ---
    VWAP_THRESHOLD_PCT = 2.0  # 乖离率阈值
    REALTIME_INTERVAL = 3     # 刷新频率
    AI_COOLDOWN_SECONDS = 300 # AI冷却时间
    SEQ_LEN = 30              # 回看天数

# ================= 2. 特征工程 =================
class AlphaFactors:
    @staticmethod
    def process_data(df, code="未知"):
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
        self.ds_client = OpenAI(api_key=Config.DEEPSEEK_API_KEY, base_url=Config.DEEPSEEK_BASE_URL)
        dashscope.api_key = Config.DASHSCOPE_API_KEY
        self.last_consult_time = {}

    def can_consult(self, code):
        last = self.last_consult_time.get(code, 0)
        return (time.time() - last) > Config.AI_COOLDOWN_SECONDS

    def _safe_parse_json(self, content_str, source="AI"):
        try:
            content_str = re.sub(r'```json|```', '', content_str).strip()
            if content_str.endswith("}") and not content_str.endswith("}}"): pass 
            data = json.loads(content_str)
            return data
        except Exception as e:
            logger.log_system(f"[{source}] JSON解析失败: {e}")
            return {"action": "WAIT", "reason": f"解析异常: {str(e)[:20]}", "score": 0}

    def consult_joint_chiefs(self, code, name, realtime_data, tech_summary, market_data, trigger_reason):
        self.last_consult_time[code] = time.time()
        
        curr = realtime_data['current']
        pct = realtime_data['pct']
        vwap = realtime_data['vwap']
        bias_vwap = realtime_data['vwap_bias']
        vol_ratio = realtime_data.get('vol_ratio', 1.0)
        
        vol_status = "缩量"
        if vol_ratio > 1.2: vol_status = "温和放量"
        if vol_ratio > 2.0: vol_status = "显著放量"
        
        prompt = f"""
        # Role: 资深A股日内操盘手
        
        # Task: 紧急交易判断
        标的：{name} ({code})
        
        # Real-time Status
        - 现价：{curr} (今日涨幅: {pct:.2f}%)
        - 均价(黄线)：{vwap:.2f}
        - **乖离率**：{bias_vwap:.2f}%
        - **官方量比**：{vol_ratio:.2f} ({vol_status})
        
        # Trigger
        系统触发: {trigger_reason}
        
        # Context
        - 大盘情绪: {market_data['avg']:.2f}%
        - 技术面: {tech_summary}
        
        # Output Format (JSON ONLY)
        {{"action": "EXECUTE" | "WAIT", "reason": "简短理由", "score": 0-100, "suggested_price": float}}
        """
        
        logger.log_llm(f" >>> [SEND {code}] VolRatio:{vol_ratio:.2f} Bias:{bias_vwap:.2f}%")

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
    def __init__(self, code):
        self.code = code
        self.scaler = RobustScaler()
        self.latest_summary = ""
        self.vol_ma5 = 0.0 # 备用手动均量

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
            end = datetime.datetime.now().strftime("%Y%m%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=400)).strftime("%Y%m%d")
            
            df = ak.stock_zh_a_hist(symbol=self.code, period="daily", start_date=start, end_date=end, adjust="qfq")
            if df is None or df.empty: return False

            df = AlphaFactors.process_data(df, self.code)
            if df.empty: return False 
            
            self.latest_summary = AlphaFactors.get_latest_summary(df)
            
            # 计算备用均量 (以防万一API不返回量比)
            vol_hist = df['volume'].shift(1).rolling(5).mean()
            self.vol_ma5 = vol_hist.iloc[-1] if not pd.isna(vol_hist.iloc[-1]) else 0
            
            feat_cols = ['Bias20', 'ATR_Pct', 'Vol_Ratio', 'RSI', 'MACD', 'BOLL_POS']
            data_X = self.scaler.fit_transform(df[feat_cols].values)
            data_y_l = df['Target_Low'].values
            data_y_h = df['Target_High'].values

            X, yl, yh = [], [], []
            for i in range(Config.SEQ_LEN, len(data_X)):
                X.append(data_X[i-Config.SEQ_LEN:i])
                yl.append(data_y_l[i])
                yh.append(data_y_h[i])
            X, yl, yh = np.array(X), np.array(yl), np.array(yh)
            
            if len(X) < 10: return False

            tf_model = self.build_transformer((Config.SEQ_LEN, len(feat_cols)))
            tf_model.fit(X, [yl, yh], batch_size=32, epochs=5, verbose=0)
            return True
        except Exception as e:
            print(f"❌ [{self.code}] 训练报错: {e}")
            return False

# ================= 5. 弹窗 UI =================
alert_lock = threading.Lock()
def popup_alert(data):
    def _show():
        with alert_lock:
            if winsound: winsound.Beep(800, 300)
            root = tk.Tk()
            
            is_buy = 'BUY' in data['type']
            bg_col = '#004d00' if is_buy else '#660000'
            
            w, h = 800, 750
            x, y = (root.winfo_screenwidth()-w)//2, (root.winfo_screenheight()-h)//2
            root.geometry(f"{w}x{h}+{x}+{y}")
            root.configure(bg=bg_col)
            root.attributes('-topmost', True)
            
            title_txt = f"⚡ 信号触发: {data['name']} ({data['code']})"
            tk.Label(root, text=title_txt, font=("黑体", 20, "bold"), bg=bg_col, fg='yellow').pack(pady=10)
            
            # --- 核心数据 (现价+涨幅) ---
            core_frame = tk.Frame(root, bg=bg_col)
            core_frame.pack(pady=10)
            
            tk.Label(core_frame, text=f"现价: {data['curr']}", 
                     font=("Arial", 36, "bold"), bg=bg_col, fg='white').pack(side='left', padx=20)
            
            pct_val = data['pct']
            pct_col = '#FF5555' if pct_val > 0 else '#55FF55' 
            tk.Label(core_frame, text=f"{pct_val:+.2f}%", 
                     font=("Arial", 36, "bold"), bg=bg_col, fg=pct_col).pack(side='left', padx=20)
            
            # --- 辅助数据 ---
            sub_frame = tk.Frame(root, bg=bg_col)
            sub_frame.pack(pady=5)
            
            tk.Label(sub_frame, text=f"均价: {data['vwap']:.2f}", 
                     font=("微软雅黑", 14), bg=bg_col, fg='#CCCCCC').pack(side='left', padx=15)
            
            bias_col = '#FF9999' if data['vwap_bias'] > 0 else '#99FF99'
            tk.Label(sub_frame, text=f"乖离: {data['vwap_bias']:.2f}%", 
                     font=("微软雅黑", 14, "bold"), bg=bg_col, fg=bias_col).pack(side='left', padx=15)

            tk.Label(sub_frame, text=f"量比: {data.get('vol_ratio',0):.2f}", 
                     font=("微软雅黑", 14), bg=bg_col, fg='cyan').pack(side='left', padx=15)
            
            tk.Label(root, text=f"触发原因: {data['reason']}", font=("微软雅黑", 12), bg=bg_col, fg='#AAAAAA').pack(pady=5)
            
            # --- AI 建议 ---
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

# ================= 6. 监控系统 (优先使用官方量比) =================
class MonitorApp:
    def __init__(self):
        self.brains = {}
        self.advisor = DualAdvisor()
        self.market_data = {'sh':0.0, 'sz':0.0, 'cy':0.0, 'avg':0.0}
        
    def init_models(self):
        print(f"\n⚡ 初始化模型与数据...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._train_one, code) for code in Config.STOCK_LIST]
            cnt = 0
            for f in futures:
                code, brain = f.result()
                if brain:
                    self.brains[code] = brain
                    cnt += 1
                    sys.stdout.write(f"\r✅ {code} 就绪")
        print(f"\n🎉 监控列表已建立: {cnt} 只股票")

    def _train_one(self, code):
        brain = EnsembleBrain(code)
        if brain.train(): return code, brain
        return code, None

    def get_market_data(self):
        try:
            df = ak.stock_zh_index_spot_sina()
            sh = float(df[df['代码']=='sh000001']['涨跌幅'].values[0])
            return {'sh': sh, 'avg': sh}
        except:
            return {'sh':0, 'avg':0}

    def run(self):
        if not self.brains:
            print("❌ 没有可用的模型，请检查网络或股票代码。")
            return

        print("📡 [分时均线战法] 监控已启动...")
        while True:
            try:
                self.market_data = self.get_market_data()
                df_real = ak.stock_zh_a_spot_em()
                
                for code, brain in self.brains.items():
                    row = df_real[df_real['代码'] == code]
                    if row.empty: continue
                    
                    name = row['名称'].values[0]
                    curr = float(row['最新价'].values[0])
                    pre_close = float(row['昨收'].values[0])
                    pct = (curr - pre_close) / pre_close * 100
                    
                    amount = float(row['成交额'].values[0])
                    volume_hand = float(row['成交量'].values[0]) 
                    
                    # === 核心修改：优先读取 akshare 官方计算的量比 ===
                    real_vol_ratio = 1.0
                    
                    # 尝试直接读取 '量比' 字段 (最准确)
                    if '量比' in row.columns and row['量比'].values[0] is not None:
                        try:
                            val = row['量比'].values[0]
                            # 有时候返回 '-' 或 NaN
                            if str(val).replace('.', '', 1).isdigit():
                                real_vol_ratio = float(val)
                            else:
                                raise ValueError("Invalid VR")
                        except:
                            # 如果官方数据读取失败，启用备用手动计算
                            minutes_elapsed = (datetime.datetime.now() - datetime.datetime.now().replace(hour=9, minute=30)).seconds / 60
                            minutes_elapsed = max(1, minutes_elapsed)
                            pred_vol_day = volume_hand / minutes_elapsed * 240
                            real_vol_ratio = pred_vol_day / (brain.vol_ma5 + 1e-5)
                    
                    # === 计算 VWAP (黄线) ===
                    vwap = curr 
                    if volume_hand > 0:
                        vwap = amount / (volume_hand * 100)
                    
                    bias_vwap = (curr - vwap) / vwap * 100
                    
                    # === 信号触发 ===
                    trigger_type = None
                    trigger_reason = ""
                    
                    if bias_vwap < -Config.VWAP_THRESHOLD_PCT:
                        trigger_type = "BUY_VWAP"
                        trigger_reason = f"股价低于均线 {abs(bias_vwap):.2f}% (超卖回归)"
                    elif bias_vwap > Config.VWAP_THRESHOLD_PCT:
                        trigger_type = "SELL_VWAP"
                        trigger_reason = f"股价高于均线 {bias_vwap:.2f}% (超买回归)"
                        
                    if trigger_type and self.advisor.can_consult(code):
                        print(f"\n🔍 [{name}] 触发 {trigger_type} | 现价:{curr} ({pct:.2f}%) | 量比:{real_vol_ratio:.2f}")
                        
                        realtime_data = {
                            'current': curr, 'pct': pct, 
                            'vwap': vwap, 'vwap_bias': bias_vwap,
                            'vol_ratio': real_vol_ratio
                        }
                        
                        res_ds, res_qw = self.advisor.consult_joint_chiefs(
                            code, name, realtime_data, brain.latest_summary, 
                            self.market_data, trigger_reason
                        )
                        
                        if res_ds.get('action') == 'EXECUTE' or res_qw.get('action') == 'EXECUTE' or res_ds.get('score', 0) > 80:
                            popup_alert({
                                'code': code, 'name': name, 'type': trigger_type,
                                'curr': curr, 'pct': pct,
                                'vwap': vwap, 'vwap_bias': bias_vwap,
                                'vol_ratio': real_vol_ratio,
                                'reason': trigger_reason,
                                'ds': res_ds, 'qw': res_qw
                            })
                        else:
                            print(f"   -> AI建议观望: {res_ds.get('reason')}")

                sys.stdout.write(f"\r[{datetime.datetime.now().strftime('%H:%M:%S')}] 监控中... 大盘:{self.market_data['sh']:.2f}%")
                sys.stdout.flush()
                time.sleep(Config.REALTIME_INTERVAL)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.log_system(f"Main Loop Error: {e}")
                time.sleep(3)

if __name__ == "__main__":
    if "sk-" not in Config.DEEPSEEK_API_KEY:
        print("❌ 请先在 Config 中填入 API Key")
    else:
        app = MonitorApp()
        app.init_models()
        app.run()