# trading_bot/config.py
from datetime import datetime, timezone, timedelta

# --- 环境选择 ---
# True 使用模拟盘API密钥, False 使用实盘API密钥
IS_DEMO_TRADING = True # 这个开关现在更重要了！

# --- OKX API 配置 ---
# 请在OKX官网申请API Key，并替换以下值
# 强烈建议使用环境变量或更安全的配置管理方式，而不是直接硬编码
API_KEY = ""
SECRET_KEY = ""
PASSPHRASE = ""
IS_DEMO_TRADING = True # True 使用模拟盘, False 使用实盘 (非常重要!)

# --- OKX 模拟盘 API 配置 ---
# 请在OKX模拟交易中申请模拟盘API Key，并替换以下值
DEMO_API_KEY = ""
DEMO_SECRET_KEY = ""
DEMO_PASSPHRASE = ""

# --- 交易参数 ---
SYMBOL = "BTC-USDT-SWAP"
TIMEFRAME = "1H"

DEFAULT_KLINE_LIMIT = 100
KLINE_LIMIT_FOR_LIVE = 300
KLINE_LIMIT_FOR_OPTIMIZATION = 5000
KLINE_LIMIT_FOR_OOS = 4000

# === 遗传算法 (GA) 配置 ===
POPULATION_SIZE = 30
NUM_GENERATIONS = 40
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.7
ELITISM_COUNT = 2

# --- 验证集相关配置 (用于 _is_validation_result_acceptable) ---
VALIDATION_SET_RATIO = 0.25
MIN_TRADES_FOR_VALIDATION = 10        # 从 15 降到 10
MAX_DRAWDOWN_FOR_VALIDATION = 0.25   # 从 0.18 (18%) 放宽到 0.25 (25%)
MIN_PNL_PCT_FOR_VALIDATION = 0.01   # 从 0.03 (3%) 降到 0.01 (1%)
MIN_PROFIT_FACTOR_FOR_VALIDATION = 1.2 # 从 1.5 降到 1.2
MIN_SHARPE_FOR_VALIDATION = 0.0       # 从 0.10 降到 0.0 (允许夏普为0或略负的进入候选池)

# --- 用于最终基因选择阶段的验证集交易次数门槛 (在main.py中使用) ---
MIN_VALIDATION_TRADES_FOR_FINAL_SELECTION = 15 # 从20降到15 (配合上面MIN_TRADES_FOR_VALIDATION的调整)

# --- 主要验证指标 ---
VALIDATION_PRIMARY_METRIC = 'Profit Factor' # 保持

# --- 训练集适应度进入验证的门槛 ---
MIN_TRAIN_FITNESS_FOR_VALIDATION = -500.0  # 保持一个较宽松的负值，允许更多尝试
                                          # （因为新适应度函数可能产生较大的负值）
                                          # 之前的 -200 可能还是有点高，我们再放宽些

# FITNESS_WEIGHTS 保持不变 (因为它们已经引导出了高PF的策略)
FITNESS_WEIGHTS = {
    'base_penalty': -1e10,                     
    'penalty_per_missing_trade': -1000,        
    'train_min_trades_threshold': 20,          
    'train_ideal_max_drawdown': 0.10,          
    'train_acceptable_max_drawdown': 0.25,     
    'train_drawdown_penalty_factor': -2000,    
    'train_drawdown_exponent': 1.5,            
    'train_min_profit_factor': 1.75,           
    'train_profit_factor_target': 3.0,         
    'train_profit_factor_weight': 300,         
    'train_sharpe_ratio_weight': 100,          
    'train_pnl_pct_weight': 5,                 
}

# === 策略基因定义 (保持不变，除非你想调整范围) ===
GENE_SCHEMA = {
    'short_ma_period':  {'type': 'int', 'range': (5, 80)},
    'long_ma_period':   {'type': 'int', 'range': (20, 250)}, # 确保 long_ma_period 范围能满足与 short 的差异
    'atr_period':       {'type': 'int', 'range': (5, 40)},
    'atr_stop_loss_multiplier': {'type': 'float', 'range': (1.0, 5.0)}, # TP/SL ratio 在 strategy.py 中约束
    'atr_take_profit_multiplier': {'type': 'float', 'range': (1.5, 10.0)},
}

INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.0005
RISK_PER_TRADE_PERCENT = 0.02
MIN_POSITION_SIZE = 0.001
