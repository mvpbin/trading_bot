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
KLINE_LIMIT_FOR_OPTIMIZATION = 5000 # 用于训练+验证的总数据量
KLINE_LIMIT_FOR_OOS = 2000          # 用于样本外测试的数据量

# === 遗传算法 (GA) 配置 ===
POPULATION_SIZE = 30 
NUM_GENERATIONS = 40 
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITISM_COUNT = 2

# --- 验证集相关配置 ---
VALIDATION_SET_RATIO = 0.25          # 验证集占总优化数据的比例
MIN_TRADES_FOR_VALIDATION = 5        # 验证集回测结果被认为有效的最低交易次数
MAX_DRAWDOWN_FOR_VALIDATION = 0.40   # 验证集回测结果可接受的最大回撤 (例如40%)
MIN_PNL_PCT_FOR_VALIDATION = 0.005   # 验证集至少盈利0.5%才被考虑 (相对于初始资本)
MIN_PROFIT_FACTOR_FOR_VALIDATION = 1.05 # 验证集盈利因子至少1.05
MIN_SHARPE_FOR_VALIDATION = -0.5      # 验证集夏普至少为-0.5 (允许轻微负夏普，主要靠其他指标过滤)
VALIDATION_PRIMARY_METRIC = 'Sharpe Ratio' # 用于比较验证集表现的主要指标 ('Sharpe Ratio', 'Sortino Ratio', 'Profit Factor', 'Total PnL', 'Calmar_like')
MIN_TRAIN_FITNESS_FOR_VALIDATION = 0.0 # 候选个体在训练集上的适应度至少要达到这个值才进行验证

# --- 适应度函数权重 (Fitness Function Weights) - 用于训练集评估 ---
FITNESS_WEIGHTS = {
    'hard_penalty_base': -1e9,
    'min_trades_threshold': 10, # 训练集上的最低交易次数         
    'max_allowed_drawdown': 0.40, # 训练集上最大允许回撤       
    'significant_loss_threshold_pct': -0.20 # 训练集上显著亏损阈值
}

# === 策略基因定义 (回归到纯ATR动态止损/止盈) ===
GENE_SCHEMA = {
    'short_ma_period':  {'type': 'int', 'range': (10, 150)},
    'long_ma_period':   {'type': 'int', 'range': (50, 400)},
    
    'atr_period':       {'type': 'int', 'range': (5, 50)},
    'atr_stop_loss_multiplier': {'type': 'float', 'range': (1.0, 7.0)}, 
    'atr_take_profit_multiplier': {'type': 'float', 'range': (1.5, 15.0)},
}

INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.0005
