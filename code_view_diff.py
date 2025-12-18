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
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

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
        self.detail_fmt = logging.Formatter('%(asctime)s %(message)s') 
        self.sys_logger = self._get_logger("system", "system.log", self.simple_fmt)
        self.mkt_logger = self._get_logger("market", "market_data.log", self.detail_fmt)
        self.pred_logger = self._get_logger("prediction", "model_pred.log", self.simple_fmt)
        self.llm_logger = self._get_logger("llm", "llm_dialog.log", self.simple_fmt)

    def _get_logger(self, name, filename, formatter):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_path = os.path.join(self.base_dir, filename)
            handler = RotatingFileHandler(file_path, maxBytes=20*1024*1024, backupCount=10, encoding='utf-8')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def log_system(self, msg): self.sys_logger.info(msg)
    def log_market(self, msg): self.mkt_logger.info(msg)
    def log_pred(self, msg): self.pred_logger.info(msg)
    def log_llm(self, msg): self.llm_logger.info(msg)

logger = LogSystem()

# ================= 1. 配置中心 =================
class Config:
    # ⚠️⚠️⚠️ 请填入 API Key ⚠️⚠️⚠️
    # DeepSeek Key
    DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" 
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    
    # 通义千问: https://dashscope.console.aliyun.com/
    DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" 
    
    
    # --- 🎯 目标股票池 ---
    STOCK_LIST = [
        "002240", # 盛新锂能
        "002709", # 天赐材料
        "002407", # 多氟多
        "002463", # 沪电股份
        "600489", # 中金黄金
        "600343", # 航天动力
        "603126", # 中材节能
        "603986", # 兆易创新
        "002121", # 科陆电子
        "600089", # 特变电工
        "605598", # 上海港湾
        "600183", # 生益科技
        "600118", # 中国卫星
        "300455", # 航天智装
        "000547", # 航天发展
        "000426", # 兴业银锡
        "000603", # 盛达资源
        "600988", # 赤峰黄金
        "002050", # 三花智控
        "002837", # 英维克
        "002080", # 中材科技
        "601138", # 工业富联
        "001267", # 汇绿生态
        "002466", # 天齐锂业
        "000630", # 铜陵有色
        "601069", # 西部黄金
        "603119", # 浙江荣泰
        "600879", # 航天电子
        "000901", # 航天科技
        "000547", # 航天发展
        "600855", # 航天长峰
        # "515880", # 通信ETF
    ]
    STOCK_LIST = list(set([x for x in STOCK_LIST if x.isdigit()]))

    WEIGHTS = {'transformer': 0.4, 'xgboost': 0.2, 'lightgbm': 0.2, 'catboost': 0.2}
    SEQ_LEN = 30          
    EPOCHS = 20           
    BATCH_SIZE = 32       
    ALERT_BUFFER_PCT = 1.5 
    MARKET_BETA = 1.2 
    REALTIME_INTERVAL = 5 
    AI_COOLDOWN_SECONDS = 300 

# ================= 2. 特征工程 =================
class AlphaFactors:
    @staticmethod
    def process_data(df, code="未知"):
        df = df.copy()
        
        # 【修复】增加对日期列名的兼容性检查
        date_col = 'date' if 'date' in df.columns else '日期'
        if date_col in df.columns:
            last_date = df.iloc[-1][date_col]
            logger.log_market(f"[{code}] 数据源最新日期: {last_date}")
        else:
            logger.log_market(f"[{code}] 警告: 未找到日期列")

        cols = ['open', 'close', 'high', 'low', 'volume']
        # 确保列存在，防止报错
        for c in cols:
            if c not in df.columns: return pd.DataFrame() # 缺列直接返回空
            
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df['pre_close'] = df['close'].shift(1)
        df.dropna(inplace=True)
        
        # --- 指标计算 ---
        df['MA20'] = df['close'].rolling(20).mean()
        df['Bias20'] = (df['close'] - df['MA20']) / df['MA20'] * 100
        
        high_low = df['high'] - df['low']
        tr = np.maximum(high_low, np.abs(df['high'] - df['pre_close']))
        tr = np.maximum(tr, np.abs(df['low'] - df['pre_close']))
        df['ATR'] = tr.rolling(14).mean()
        df['ATR_Pct'] = df['ATR'] / df['pre_close'] * 100 
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/(loss+1e-5)))
        
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp12 - exp26
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD'] = 2 * (df['DIF'] - df['DEA'])
        
        df['BOLL_MID'] = df['close'].rolling(20).mean()
        df['BOLL_STD'] = df['close'].rolling(20).std()
        df['BOLL_UP'] = df['BOLL_MID'] + 2 * df['BOLL_STD']
        df['BOLL_LOW'] = df['BOLL_MID'] - 2 * df['BOLL_STD']
        df['BOLL_POS'] = (df['close'] - df['BOLL_LOW']) / (df['BOLL_UP'] - df['BOLL_LOW'] + 1e-5)

        df['Vol_Ratio'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-5)

        # 记录计算结果
        last = df.iloc[-1]
        log_msg = (
            f"[{code}] 指标验算:\n"
            f"  > 收盘: {last['close']:.2f}\n"
            f"  > RSI: {last['RSI']:.2f}\n"
            f"  > MACD: DIF={last['DIF']:.3f}, DEA={last['DEA']:.3f}\n"
            f"  > BOLL: UP={last['BOLL_UP']:.2f}, LOW={last['BOLL_LOW']:.2f}\n"
        )
        logger.log_market(log_msg)

        df['Target_Low'] = (df['low'] - df['pre_close']) / df['pre_close'] * 100
        df['Target_High'] = (df['high'] - df['pre_close']) / df['pre_close'] * 100
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df

    @staticmethod
    def get_latest_summary(df):
        row = df.iloc[-1]
        rsi_hist = df['RSI'].tail(60) 
        rsi_rank = (rsi_hist < row['RSI']).mean() * 100 
        
        if row['DIF'] > row['DEA']: macd_status = "金叉(多)"
        else: macd_status = "死叉(空)"
        
        boll_status = f"{row['BOLL_POS']:.2f}"
        if row['BOLL_POS'] > 1: boll_status += "(超买)"
        elif row['BOLL_POS'] < 0: boll_status += "(超卖)"
        
        summary = (
            f"【RSI】{row['RSI']:.1f} (历史分位:{rsi_rank:.0f}%)\n"
            f"【MACD】{macd_status} | DIF:{row['DIF']:.2f}\n"
            f"【布林带】位置:{boll_status}\n"
            f"【ATR】{row['ATR_Pct']:.2f}%\n"
            f"【量比】{row['Vol_Ratio']:.2f}"
        )
        return summary

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
            data = json.loads(content_str)
            
            def find_val(d, key):
                if isinstance(d, dict):
                    if key in d: return d[key]
                    for v in d.values():
                        res = find_val(v, key)
                        if res: return res
                return None

            action = find_val(data, 'action')
            if not action: action = data.get("decision", "WAIT")
            action = str(action).upper()
            if action not in ["EXECUTE", "WAIT"]: action = "WAIT"
            
            reason = data.get("reason", "AI未给出理由")
            if isinstance(reason, dict): reason = str(reason)
            
            score = data.get("score", 0)
            
            suggested_price = data.get("suggested_price", 0.0)
            if isinstance(suggested_price, str):
                nums = re.findall(r"\d+\.?\d*", suggested_price)
                if nums: suggested_price = float(nums[0])
                else: suggested_price = 0.0
            
            return {
                "action": action, 
                "reason": str(reason)[:200], 
                "score": int(score),
                "suggested_price": float(suggested_price)
            }
            
        except Exception as e:
            logger.log_system(f"[{source}] 解析失败: {e} | 原文: {content_str}")
            return {"action": "WAIT", "reason": f"解析异常: {e}", "score": 0, "suggested_price": 0.0}

    def consult_joint_chiefs(self, code, name, curr_price, chg_pct, tech_summary, market_data, target_type, target_price):
        self.last_consult_time[code] = time.time()
        direction = "抄底买入" if target_type == 'BUY' else "止盈卖出"
        
        prompt = f"""
        # Role: 资深A股超短线游资操盘手
        
        # Task: 紧急交易决策
        标的：{name} ({code})
        现价：{curr_price} (涨幅 {chg_pct:.2f}%)
        量化信号：逼近{direction}位 {target_price:.2f}
        
        # Market Context (大盘全景)
        上证指数(sh000001)：{market_data['sh']:.2f}%
        深证成指(sz399001)：{market_data['sz']:.2f}%
        创业板指(sz399006)：{market_data['cy']:.2f}%
        综合情绪：{market_data['avg']:.2f}%
        
        # Technical Indicators (技术面)
        {tech_summary}
        
        # Decision Requirements
        请基于上述数据，判断该量化信号是否有效。如果不安全，请给出观望建议。
        
        # Output Format (JSON ONLY)
        1. 必须返回扁平JSON，严禁嵌套。
        2. 字段说明：
           - "action": "EXECUTE" (坚决执行) 或 "WAIT" (风险大，观望)
           - "reason": "30字以内中文理由，犀利直接"
           - "score": 0-100 (信心分数)
           - "suggested_price": float (你认为最合理的{direction}挂单价格，数字)
        
        # Example
        {{"action": "WAIT", "reason": "大盘情绪冰点，量能背离，建议更低位接回", "score": 40, "suggested_price": {curr_price * 0.98:.2f}}}
        """
        
        logger.log_llm(f" >>> [SEND to {code}] \n{prompt}")

        def call_deepseek():
            try:
                res = self.ds_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={'type': 'json_object'}, temperature=0.1
                )
                raw = res.choices[0].message.content
                logger.log_llm(f" <<< [DeepSeek Raw] {raw}") 
                return self._safe_parse_json(raw, "DeepSeek")
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
                    raw = res.output.choices[0].message.content
                    logger.log_llm(f" <<< [Qwen Raw] {raw}")
                    return self._safe_parse_json(raw, "Qwen")
                return {"action": "WAIT", "reason": "Qwen拒绝", "score": 0}
            except Exception as e:
                return {"action": "WAIT", "reason": f"Qwen Error: {e}", "score": 0}

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(call_deepseek)
            f2 = executor.submit(call_qwen)
            return f1.result(), f2.result()

# ================= 4. 模型融合 =================
# ================= 4. 模型融合 =================
class EnsembleBrain:
    def __init__(self, code):
        self.code = code
        self.scaler = RobustScaler()
        self.pred_low_pct = 0.0
        self.pred_high_pct = 0.0
        self.latest_summary = ""

    def build_transformer(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        att = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
        x = layers.Add()([inputs, att])
        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="gelu")(x)
        out_l = layers.Dense(1, name="l")(x)
        out_h = layers.Dense(1, name="h")(x)
        model = keras.Model(inputs, [out_l, out_h])
        model.compile(optimizer='adam', loss=['mse', 'mse'], loss_weights=[0.5, 0.5])
        return model

    def train(self):
        try:
            end = datetime.datetime.now().strftime("%Y%m%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=400)).strftime("%Y%m%d")
            
            # 获取数据
            df = ak.stock_zh_a_hist(symbol=self.code, period="daily", start_date=start, end_date=end, adjust="qfq")
            
            if df is None or df.empty:
                # logger.log_system(f"[{self.code}] 获取数据为空") # 可选日志
                return False

            # 【修复】统一列名重命名，增强鲁棒性
            # 1. 先打印原始列名，方便调试 (可选)
            # logger.log_market(f"[{self.code}] 原始列名: {df.columns.tolist()}")

            # 2. 定义映射关系，覆盖中文和英文情况
            rename_map = {
                "日期": "date", "date": "date",
                "开盘": "open", "open": "open",
                "收盘": "close", "close": "close",
                "最高": "high", "high": "high",
                "最低": "low", "low": "low",
                "成交量": "volume", "volume": "volume"
            }
            df.rename(columns=rename_map, inplace=True)

            # 3. 检查关键列是否存在
            required_cols = ['date', 'open', 'close', 'high', 'low', 'volume']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                logger.log_system(f"[{self.code}] 缺失关键列: {missing_cols}，跳过训练")
                return False
            
            if len(df) < 60: return False
            
            # 【日志记录】验证数据源是否包含今日数据
            logger.log_market(f"[{self.code}] 训练数据最后日期: {df.iloc[-1]['date']}")

            df = AlphaFactors.process_data(df, self.code)
            if df.empty: return False 
            
            self.latest_summary = AlphaFactors.get_latest_summary(df)
            
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

            # 模型训练
            tf_model = self.build_transformer((Config.SEQ_LEN, len(feat_cols)))
            tf_model.fit(X, [yl, yh], batch_size=32, epochs=15, verbose=0)
            last_seq = data_X[-Config.SEQ_LEN:].reshape(1, Config.SEQ_LEN, len(feat_cols))
            tf_pred = tf_model.predict(last_seq, verbose=0)
            tf_low, tf_high = tf_pred[0][0][0], tf_pred[1][0][0]

            X_flat, last_seq_flat = X.reshape(X.shape[0], -1), last_seq.reshape(1, -1)
            xgb_l = xgb.XGBRegressor(n_estimators=50, max_depth=3, verbosity=0).fit(X_flat, yl)
            xgb_h = xgb.XGBRegressor(n_estimators=50, max_depth=3, verbosity=0).fit(X_flat, yh)
            lgb_l = lgb.LGBMRegressor(n_estimators=50, max_depth=3, verbose=-1).fit(X_flat, yl)
            lgb_h = lgb.LGBMRegressor(n_estimators=50, max_depth=3, verbose=-1).fit(X_flat, yh)
            cat_l = cb.CatBoostRegressor(iterations=50, depth=3, verbose=0).fit(X_flat, yl)
            cat_h = cb.CatBoostRegressor(iterations=50, depth=3, verbose=0).fit(X_flat, yh)

            w = Config.WEIGHTS
            self.pred_low_pct = (tf_low*w['transformer'] + xgb_l.predict(last_seq_flat)[0]*w['xgboost'] + lgb_l.predict(last_seq_flat)[0]*w['lightgbm'] + cat_l.predict(last_seq_flat)[0]*w['catboost'])
            self.pred_high_pct = (tf_high*w['transformer'] + xgb_h.predict(last_seq_flat)[0]*w['xgboost'] + lgb_h.predict(last_seq_flat)[0]*w['lightgbm'] + cat_h.predict(last_seq_flat)[0]*w['catboost'])
            
            logger.log_pred(f"[{self.code}] 原始预测: Low={self.pred_low_pct:.2f}% High={self.pred_high_pct:.2f}%")

            if self.pred_low_pct > -0.5: self.pred_low_pct = -1.0
            if self.pred_high_pct < 0.5: self.pred_high_pct = 1.0
            return True
        except Exception as e:
            # 打印更详细的错误堆栈
            import traceback
            logger.log_system(f"训练异常 {self.code}: {e}\n{traceback.format_exc()}")
            return False

# ================= 5. 弹窗 UI =================
alert_lock = threading.Lock()
def popup_alert(data):
    def _show():
        with alert_lock:
            if winsound: winsound.Beep(600, 200)
            root = tk.Tk()
            is_buy = data['target_type'] == 'BUY'
            bg_col = '#004d00' if is_buy else '#800000'
            w, h = 750, 650
            x, y = (root.winfo_screenwidth()-w)//2, (root.winfo_screenheight()-h)//2
            root.geometry(f"{w}x{h}+{x}+{y}")
            root.configure(bg=bg_col)
            root.attributes('-topmost', True)
            
            title = f"🚀 抄底: {data['name']}" if is_buy else f"🛑 逃顶: {data['name']}"
            tk.Label(root, text=title, font=("黑体", 22, "bold"), bg=bg_col, fg='yellow').pack(pady=10)
            tk.Label(root, text=f"现价: {data['curr']:.2f} (涨幅:{data['pct']:.2f}%)", font=("Arial", 28, "bold"), bg=bg_col, fg='white').pack(pady=5)
            tk.Label(root, text=f"🎯 预测目标: ¥{data['target_price']:.2f}", font=("微软雅黑", 16), bg=bg_col, fg='#DDDDDD').pack(pady=5)
            
            tech_frame = tk.LabelFrame(root, text="📊 关键指标", font=("微软雅黑", 10), bg=bg_col, fg='white')
            tech_frame.pack(fill='x', padx=20, pady=5)
            tk.Label(tech_frame, text=data['tech_summary'], font=("Consolas", 10), bg=bg_col, fg='#AAFFAA', justify='left').pack(padx=10, pady=5)
            
            ai_frame = tk.LabelFrame(root, text="🧠 双核军师建议", font=("微软雅黑", 12), bg=bg_col, fg='yellow')
            ai_frame.pack(fill='both', expand=True, padx=20, pady=10)
            
            ds, qw = data['ds_advice'], data['qw_advice']
            
            ds_pr = ds.get('suggested_price', 0)
            qw_pr = qw.get('suggested_price', 0)
            ds_pr_str = f" | 挂单: ¥{ds_pr:.2f}" if ds_pr > 0 else ""
            qw_pr_str = f" | 挂单: ¥{qw_pr:.2f}" if qw_pr > 0 else ""
            
            tk.Label(ai_frame, text=f"[DeepSeek] {ds.get('action','WAIT')} (信心:{ds.get('score',0)}){ds_pr_str}\nReason: {ds.get('reason','无')}", 
                     font=("微软雅黑", 11), bg=bg_col, fg='cyan', wraplength=650, justify='left').pack(anchor='w', padx=10, pady=5)
            
            tk.Label(ai_frame, text=f"--------------------------------", bg=bg_col, fg='#555555').pack()
            
            tk.Label(ai_frame, text=f"[通义千问] {qw.get('action','WAIT')} (信心:{qw.get('score',0)}){qw_pr_str}\nReason: {qw.get('reason','无')}", 
                     font=("微软雅黑", 11), bg=bg_col, fg='orange', wraplength=650, justify='left').pack(anchor='w', padx=10, pady=5)
            
            tk.Button(root, text="关闭", font=("微软雅黑", 12), command=root.destroy).pack(pady=10)
            root.mainloop()
    threading.Thread(target=_show, daemon=True).start()

# ================= 6. 监控系统 =================
class MonitorApp:
    def __init__(self):
        self.brains = {}
        self.advisor = DualAdvisor()
        self.market_data = {'sh':0.0, 'sz':0.0, 'cy':0.0, 'avg':0.0}
        
    def init_models(self):
        print(f"\n⚡ 启动多模型训练 (日志已开启: ./logs/)...")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self._train_one, code) for code in Config.STOCK_LIST]
            cnt = 0
            for f in futures:
                code, brain = f.result()
                if brain:
                    self.brains[code] = brain
                    cnt += 1
                    sys.stdout.write(f"\r✅ {code} 就绪")
                    sys.stdout.flush()
        print(f"\n🎉 训练完成: {cnt}/{len(Config.STOCK_LIST)}")

    def _train_one(self, code):
        brain = EnsembleBrain(code)
        if brain.train(): return code, brain
        return code, None

    def get_market_data(self):
        try:
            df = ak.stock_zh_index_spot_sina()
            sh, sz, cy = 0.0, 0.0, 0.0
            
            row_sh = df[df['代码'] == 'sh000001']
            if not row_sh.empty: sh = float(row_sh.iloc[0]['涨跌幅'])
                
            row_sz = df[df['代码'] == 'sz399001']
            if not row_sz.empty: sz = float(row_sz.iloc[0]['涨跌幅'])
                
            row_cy = df[df['代码'] == 'sz399006']
            if not row_cy.empty: cy = float(row_cy.iloc[0]['涨跌幅'])
            
            avg = (sh + sz + cy) / 3.0
            logger.log_market(f"Market Sentiment: SH={sh}% SZ={sz}% CY={cy}% AVG={avg}%")
            return {'sh': sh, 'sz': sz, 'cy': cy, 'avg': avg}

        except Exception as e:
            logger.log_system(f"大盘数据异常: {e}")
            return {'sh':0.0, 'sz':0.0, 'cy':0.0, 'avg':0.0}

    def run(self):
        print("📡 实时监控启动...")
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
                    
                    beta_fix = self.market_data['avg'] * Config.MARKET_BETA
                    target_low_pct = brain.pred_low_pct + (beta_fix if beta_fix < 0 else 0)
                    target_high_pct = brain.pred_high_pct + (beta_fix if beta_fix > 0 else 0)
                    
                    price_buy = pre_close * (1 + target_low_pct / 100)
                    price_sell = pre_close * (1 + target_high_pct / 100)
                    
                    buffer = Config.ALERT_BUFFER_PCT / 100
                    is_buy = curr <= price_buy * (1 + buffer)
                    is_sell = curr >= price_sell * (1 - buffer)
                    
                    if (is_buy or is_sell) and self.advisor.can_consult(code):
                        target_type = 'BUY' if is_buy else 'SELL'
                        target_price = price_buy if is_buy else price_sell
                        print(f"\n🔍 [{name}] 触发 {target_type}, 咨询军师...")
                        
                        logger.log_system(f"触发咨询: {name}({code}) {target_type} Price:{curr} Target:{target_price}")

                        res_ds, res_qw = self.advisor.consult_joint_chiefs(
                            code, name, curr, pct, brain.latest_summary, 
                            self.market_data, target_type, target_price
                        )
                        
                        score = res_ds.get('score',0) + res_qw.get('score',0)
                        any_exec = (res_ds.get('action')=='EXECUTE') or (res_qw.get('action')=='EXECUTE')
                        hard_trig = (curr <= price_buy) if is_buy else (curr >= price_sell)
                        
                        if any_exec or score > 130 or hard_trig:
                            popup_alert({
                                'code': code, 'name': name, 'curr': curr, 'pct': pct,
                                'target_price': target_price, 'target_type': target_type,
                                'tech_summary': brain.latest_summary,
                                'ds_advice': res_ds, 'qw_advice': res_qw
                            })
                        else:
                            print(f"📉 观望: DS信心{res_ds.get('score')} / QW信心{res_qw.get('score')}")

                sys.stdout.write(f"\r[{datetime.datetime.now().strftime('%H:%M:%S')}] SH:{self.market_data['sh']:.2f}% | AVG:{self.market_data['avg']:.2f}% | Monitoring...")
                sys.stdout.flush()
                time.sleep(Config.REALTIME_INTERVAL)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.log_system(f"主循环: {e}")
                time.sleep(3)

if __name__ == "__main__":
    if "sk-" not in Config.DEEPSEEK_API_KEY:
        print("❌ 错误: 请填写 API Key")
    else:
        app = MonitorApp()
        app.init_models()
        app.run()