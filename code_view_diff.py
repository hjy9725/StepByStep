# coding=utf-8
from __future__ import print_function, absolute_import
import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
from gm.api import *

# === 1. 环境配置 ===
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class Ridge: 
        def fit(self, x, y): pass
        def predict(self, x): return [0.0]
    class StandardScaler:
        def fit(self, x): pass
        def transform(self, x): return x

# ================= 2. 策略池配置 =================
try:
    import token_stock_list_config as cfg
    RAW_STOCK_LIST = cfg.CURRENT_STOCK_LIST
except ImportError:
    RAW_STOCK_LIST = ['600460', '000625', '000063'] 

TARGET_SYMBOLS = []
for code in RAW_STOCK_LIST:
    if code.startswith('6'): TARGET_SYMBOLS.append("SHSE." + code)
    else: TARGET_SYMBOLS.append("SZSE." + code)

# --- 参数 ---
POSITION_RATIO = 0.99   
SWITCH_THRESHOLD = 1.3  # 提高换仓门槛，减少频繁操作
MAX_DRAWDOWN_SELL = 0.05 

# ================= 3. AI 核心 (保持不变) =================
class AlphaFactors:
    @staticmethod
    def process_data(df):
        if df.empty or len(df) < 30: return pd.DataFrame()
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs = gain / (loss + 1e-5)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['ROC_5'] = df['close'].pct_change(5) * 100
        df['MA20'] = df['close'].rolling(20).mean()
        df['Trend'] = (df['close'] - df['MA20']) / df['MA20']
        df['Target'] = df['close'].shift(-3) / df['close'] - 1.0
        df.dropna(inplace=True)
        return df

class MomentumBrain:
    def __init__(self, symbol):
        self.symbol = symbol
        self.model = Ridge(alpha=1.0) 
        self.scaler = StandardScaler()
        self.is_ready = False

    def train(self, data_df):
        try:
            df = AlphaFactors.process_data(data_df)
            if len(df) < 30: return 
            X = df[['RSI', 'ROC_5', 'Trend']].values
            y = df['Target'].values
            self.scaler.fit(X)
            self.model.fit(self.scaler.transform(X), y)
            self.is_ready = True
        except:
            self.is_ready = False

    def predict_score(self, rsi, roc_5, trend):
        if not self.is_ready: return 0.0
        try:
            X_new = np.array([[rsi, roc_5, trend]])
            pred = self.model.predict(self.scaler.transform(X_new))
            return float(pred[0]) * 100 
        except:
            return 0.0

# ================= 4. 优化后的主逻辑 =================

def init(context):
    print("🔥 [AI 主升浪追击 v3.1 极速版] 启动...")
    context.target_symbols = TARGET_SYMBOLS 
    context.brains = {}         
    context.high_water_mark = {} 
    
    # === 优化点1：减少训练数据量，只取最近1年 ===
    # 之前是3年，其实最近1年的数据对捕捉当前妖股特征更重要，且训练快3倍
    print(f"🧠 正在极速训练 AI (最近1年数据)...")
    end_t = context.backtest_start_time
    start_t = (pd.to_datetime(end_t) - timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')
    
    count = 0
    # 批量处理：如果本地有缓存，这里其实很快
    for symbol in context.target_symbols:
        try:
            data = history(symbol=symbol, frequency='1d', start_time=start_t, end_time=end_t, 
                           fields='close,high,low', adjust=ADJUST_PREV, df=True)
            brain = MomentumBrain(symbol)
            brain.train(data)
            if brain.is_ready:
                context.brains[symbol] = brain
                count += 1
        except: pass
            
    print(f"✅ 模型就绪: {count}/{len(TARGET_SYMBOLS)} 只. 开始监控...")
    # 依然订阅分钟线，用于精确止损，但选股逻辑我们会降频处理
    subscribe(symbols=context.target_symbols, frequency='60s')

def on_bar(context, bars):
    # --- 模块 A: 实时风控 (每一分钟都跑) ---
    # 必须每分钟检查，防止主升浪瞬间跳水
    check_positions_risk(context, bars)
    
    # --- 模块 B: AI 选股与换仓 (每 30 分钟跑一次) ---
    # 优化核心：不在每分钟都做全市场扫描
    # 逻辑：只有在 10:00, 10:30, 11:00, 13:30... 这种整点半点时刻才选股
    if context.now.minute % 30 != 0:
        return

    # 执行选股逻辑
    do_market_scan_and_switch(context, bars)

def check_positions_risk(context, bars):
    """
    负责持仓的止盈止损，反应要快
    """
    positions = context.account().positions()
    for pos in positions:
        symbol = pos.symbol
        if pos.volume <= 0: continue
        
        # 快速获取当前价格
        current_bar = [b for b in bars if b.symbol == symbol]
        if not current_bar: continue
        price = current_bar[0].close
        
        # 更新高水位
        if symbol not in context.high_water_mark: context.high_water_mark[symbol] = pos.vwap
        if price > context.high_water_mark[symbol]: context.high_water_mark[symbol] = price
        
        highest = context.high_water_mark[symbol]
        drawdown = (highest - price) / highest
        pnl_pct = (price - pos.vwap) / pos.vwap

        should_sell = False
        reason = ""

        # 止损止盈逻辑
        if pnl_pct < -0.08:
            should_sell = True; reason = "硬止损-8%"
        elif pnl_pct > 0.05 and drawdown > MAX_DRAWDOWN_SELL:
            should_sell = True; reason = f"回撤{drawdown:.1%}止盈"

        if should_sell:
            print(f"⚡ [风控卖出] {symbol} | 盈亏: {pnl_pct:.2%} | 原因: {reason}")
            order_close_all()

def do_market_scan_and_switch(context, bars):
    """
    负责全市场扫描，寻找更强的票
    """
    market_scores = scan_market_scores(context)
    if not market_scores: return

    best_candidate = market_scores[0]
    positions = context.account().positions()
    
    # 1. 空仓买入
    if not positions:
        if best_candidate['score'] > 0:
            print(f"🚀 [AI 买入] 龙头 {best_candidate['symbol']} | 评分: {best_candidate['score']:.2f}")
            order_target_percent(symbol=best_candidate['symbol'], percent=POSITION_RATIO, order_type=OrderType_Market, position_side=PositionSide_Long)
            context.high_water_mark[best_candidate['symbol']] = best_candidate['price']
        return

    # 2. 持仓换股 (Switch)
    # 只有当手里有票时才对比
    for pos in positions:
        if pos.symbol == best_candidate['symbol']: return # 已经持有第一名，不动
        
        current_score = -99
        for item in market_scores:
            if item['symbol'] == pos.symbol:
                current_score = item['score']
                break
        
        # 换仓阈值：新票比老票强 1.3 倍才换
        if best_candidate['score'] > current_score * SWITCH_THRESHOLD and best_candidate['score'] > 5.0:
            print(f"🔄 [强弱切换] 卖 {pos.symbol}({current_score:.1f}) -> 买 {best_candidate['symbol']}({best_candidate['score']:.1f})")
            order_close_all()
            order_target_percent(symbol=best_candidate['symbol'], percent=POSITION_RATIO, order_type=OrderType_Market, position_side=PositionSide_Long)
            context.high_water_mark[best_candidate['symbol']] = best_candidate['price']

def scan_market_scores(context):
    candidates = []
    # 优化：批量获取最后一行数据用于快速过滤，避免每只票都拉历史
    # 这里为了代码兼容性，我们依然遍历，但由于频率降低到了30分钟一次，速度是完全可以接受的
    
    for symbol in context.target_symbols:
        # 只取最近25个数据点，极速模式
        recent = history_n(symbol=symbol, frequency='1d', count=25, end_time=context.now, fields='close', df=True)
        if len(recent) < 22: continue
        
        close = recent['close'].iloc[-1]
        close_5 = recent['close'].iloc[-6]
        roc_5 = (close / close_5 - 1) * 100
        
        # 粗过滤：如果最近5天没涨，甚至在跌，直接 pass，不计算 RSI 和 AI，节省时间
        if roc_5 < 0: continue 

        ma20 = recent['close'].iloc[-20:].mean()
        trend_bias = (close - ma20) / ma20
        
        # 只有多头排列才算细账
        if trend_bias > 0:
            # RSI 计算
            delta = recent['close'].diff()
            u = delta.where(delta > 0, 0).iloc[-6:].mean()
            d = (-delta.where(delta < 0, 0)).iloc[-6:].mean()
            rsi = 100 if d == 0 else 100 - (100 / (1 + u/d))
            
            brain = context.brains.get(symbol)
            ai_pred = brain.predict_score(rsi, roc_5, trend_bias) if (brain and brain.is_ready) else 0
            
            score = roc_5 * 1.5 + (rsi - 50) * 0.2 + ai_pred * 5.0
            
            candidates.append({'symbol': symbol, 'score': score, 'price': close})
        
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates

def on_order_status(context, order):
    if order['status'] == 3:
        print(f"✅ 成交: {order['symbol']} {'买' if order['side']==1 else '卖'} @ {order['price']}")

# ================= 回测入口 =================
if __name__ == '__main__':
    now = pd.Timestamp.now()
    end_str = now.strftime("%Y-%m-%d %H:%M:%S")
    start_str = (now - timedelta(days=180)).strftime("%Y-%m-%d %H:%M:%S") # 回测4个月
    
    print("========================================")
    print(f"⏳ 回测区间: {start_str} ~ {end_str}")
    print("========================================")

    run(strategy_id=cfg.ID, 
        filename='longtou.py',                  
        mode=MODE_BACKTEST,
        token=cfg.TOKEN,           
        backtest_start_time=start_str,
        backtest_end_time=end_str,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000, 
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)