# trading_bot/strategy.py
import pandas as pd
from indicators import Indicators
import config 
from typing import Optional
import logging
import random # 确保导入 random

logger = logging.getLogger(f"trading_bot.{__name__}")

class EvolvableStrategy:
    def __init__(self, genes: dict):
        self.genes = genes 
        self.indicators_calculator = Indicators()

        # MA Genes
        self.short_ma_period = int(genes['short_ma_period'])
        self.long_ma_period = int(genes['long_ma_period'])

        # ATR Genes for SL/TP
        self.atr_period = int(genes['atr_period'])
        self.atr_stop_loss_multiplier = float(genes['atr_stop_loss_multiplier'])
        self.atr_take_profit_multiplier = float(genes['atr_take_profit_multiplier'])

        # Position state
        self.position: Optional[str] = None
        self.entry_price: float = 0.0
        self.entry_price_time: Optional[pd.Timestamp] = None
        self.trades_count: int = 0
        
        self.stop_loss_price: float = 0.0
        self.take_profit_price: float = 0.0
        self.atr_at_entry: Optional[float] = None

        self._ensure_parameter_constraints()

    def _ensure_parameter_constraints(self):
        """确保策略参数符合逻辑约束。"""
        # MA周期约束
        if 'short_ma_period' in self.genes and 'long_ma_period' in self.genes:
            s_min, s_max = config.GENE_SCHEMA['short_ma_period']['range']
            l_min, l_max = config.GENE_SCHEMA['long_ma_period']['range']
            self.short_ma_period = max(s_min, min(self.short_ma_period, s_max))
            self.long_ma_period = max(l_min, min(self.long_ma_period, l_max))
            if self.short_ma_period >= self.long_ma_period:
                self.long_ma_period = min(self.short_ma_period + random.randint(max(1, int((l_max-s_min)*0.05) if l_max > s_min else 1), 
                                                                          max(5, int((l_max-s_min)*0.1)) if l_max > s_min else 5), 
                                          l_max)
                if self.short_ma_period >= self.long_ma_period:
                    self.short_ma_period = max(s_min, self.long_ma_period - random.randint(max(1, int((l_max-s_min)*0.05) if l_max > s_min else 1), 
                                                                                      max(5, int((l_max-s_min)*0.1)) if l_max > s_min else 5))
                    self.short_ma_period = max(s_min, self.short_ma_period) 
                    if self.short_ma_period >= self.long_ma_period and l_max > s_min +1 :
                         self.short_ma_period = (s_min + s_max) // 2
                         self.long_ma_period = min(self.short_ma_period + int((l_max-s_min)*0.2) if (l_max-s_min)*0.2 >=1 else 5 , l_max)
                         self.long_ma_period = max(self.long_ma_period, self.short_ma_period + 1)

        # ATR乘数约束
        if 'atr_take_profit_multiplier' in self.genes and 'atr_stop_loss_multiplier' in self.genes:
            sl_mult_min, sl_mult_max = config.GENE_SCHEMA['atr_stop_loss_multiplier']['range']
            tp_mult_min, tp_max_range = config.GENE_SCHEMA['atr_take_profit_multiplier']['range']
            self.atr_stop_loss_multiplier = max(sl_mult_min, min(self.atr_stop_loss_multiplier, sl_mult_max))
            self.atr_take_profit_multiplier = max(tp_mult_min, min(self.atr_take_profit_multiplier, tp_max_range))
            if self.atr_take_profit_multiplier <= self.atr_stop_loss_multiplier:
                self.atr_take_profit_multiplier = self.atr_stop_loss_multiplier + 0.5 
                self.atr_take_profit_multiplier = min(self.atr_take_profit_multiplier, tp_max_range) 
                if self.atr_take_profit_multiplier <= self.atr_stop_loss_multiplier and \
                   self.atr_stop_loss_multiplier > sl_mult_min + 0.1: 
                    self.atr_stop_loss_multiplier = self.atr_take_profit_multiplier - 0.1
                    self.atr_stop_loss_multiplier = max(self.atr_stop_loss_multiplier, sl_mult_min)

    def _calculate_indicators(self, df_klines: pd.DataFrame) -> pd.DataFrame:
        df = df_klines.copy()
        df = self.indicators_calculator.add_sma(df, period=self.short_ma_period, column_name='close')
        df = self.indicators_calculator.add_sma(df, period=self.long_ma_period, column_name='close')
        df = self.indicators_calculator.add_atr(df, period=self.atr_period)
        return df

    def generate_signals(self, df_klines_with_indicators: pd.DataFrame) -> Optional[str]:
        df = df_klines_with_indicators
        if df.empty or len(df) < 2: 
            return None

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        sma_short_col = f'SMA_{self.short_ma_period}'
        sma_long_col = f'SMA_{self.long_ma_period}'
        atr_col = f'ATR_{self.atr_period}'

        required_cols_latest = [sma_short_col, sma_long_col, atr_col]
        for col in required_cols_latest:
            if col not in latest or pd.isna(latest[col]):
                return None
        if pd.isna(previous[sma_short_col]) or pd.isna(previous[sma_long_col]):
            return None

        signal: Optional[str] = None
        current_price = float(latest['close'])
        current_atr = float(latest[atr_col])
        
        buy_condition_ma_cross = (previous[sma_short_col] < previous[sma_long_col]) and \
                                 (latest[sma_short_col] > latest[sma_long_col])
        
        if buy_condition_ma_cross and self.position is None: 
            if current_atr > 0: 
                signal = 'buy_long'
                self.atr_at_entry = current_atr 
            else: 
                logger.warning(f"信号生成：在 {latest.name} ATR值为 {current_atr}，无法入场。")

        if self.position == 'long':
            if self.stop_loss_price > 0 and current_price <= self.stop_loss_price:
                signal = 'close_long_sl'
            elif self.take_profit_price > 0 and current_price >= self.take_profit_price:
                signal = 'close_long_tp'
            else:
                sell_condition_ma_cross = (previous[sma_short_col] > previous[sma_long_col]) and \
                                          (latest[sma_short_col] < latest[sma_long_col])
                if sell_condition_ma_cross:
                    signal = 'close_long_signal'
        return signal

    def update_position_after_trade(self, signal_type: str,
                                    entry_or_exit_price: float, 
                                    timestamp: pd.Timestamp):
        if signal_type == 'buy_long':
            self.position = 'long'
            self.entry_price = entry_or_exit_price
            self.entry_price_time = timestamp
            self.trades_count += 1

            if self.atr_at_entry is not None and self.atr_at_entry > 0:
                actual_sl_distance = self.atr_at_entry * self.atr_stop_loss_multiplier
                tp_distance = self.atr_at_entry * self.atr_take_profit_multiplier
                self.stop_loss_price = self.entry_price - actual_sl_distance
                self.take_profit_price = self.entry_price + tp_distance
            else:
                logger.error(f"严重错误：为基因 {self.genes} 进入交易，但入场时ATR无效：{self.atr_at_entry}。此交易可能行为异常。")
                self.position = None 
                self.trades_count = max(0, self.trades_count - 1) 
                return

        elif signal_type.startswith('close_long'):
            self.position = None
            self.entry_price = 0.0
            self.entry_price_time = None
            self.atr_at_entry = None 
            self.stop_loss_price = 0.0 
            self.take_profit_price = 0.0