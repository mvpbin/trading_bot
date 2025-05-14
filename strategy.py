# trading_bot/strategy.py
import pandas as pd
from indicators import Indicators
import config
from typing import Optional, Dict, Any
import logging
import random
import copy

logger = logging.getLogger(f"trading_bot.{__name__}")

class EvolvableStrategy:
    def __init__(self, genes: dict):
        self.genes_input = copy.deepcopy(genes)
        self.genes = genes # This will be modified by _ensure_parameter_constraints
        self.indicators_calculator = Indicators()

        self.short_ma_period: Optional[int] = None
        self.long_ma_period: Optional[int] = None
        self.atr_period: Optional[int] = None
        self.atr_stop_loss_multiplier: Optional[float] = None
        self.atr_take_profit_multiplier: Optional[float] = None
        
        self.position: Optional[str] = None
        self.entry_price: float = 0.0
        self.entry_price_time: Optional[pd.Timestamp] = None
        self.trades_count: int = 0 # This is per backtest run for this strategy instance
        self.stop_loss_price: float = 0.0
        self.take_profit_price: float = 0.0
        self.atr_at_entry: Optional[float] = None
        self.initial_calculated_stop_loss_price_for_sizing: float = 0.0

        self._ensure_parameter_constraints() # Apply constraints
        if not self._validate_parameters_after_constraints(): # Validate after constraints
            logger.critical(f"策略参数在约束后仍然无效! 原始基因: {self.genes_input}, 约束后基因: {self.genes}")
            # Consider raising an error to prevent use of invalid strategy
            # raise ValueError("策略参数约束后仍然无效。")


    def _validate_parameters_after_constraints(self) -> bool:
        """在应用约束后再次验证参数的有效性"""
        if not (isinstance(self.short_ma_period, int) and isinstance(self.long_ma_period, int) and
                self.short_ma_period > 0 and self.long_ma_period > 0 and
                self.short_ma_period < self.long_ma_period): # Basic check
            logger.error(f"MA周期在约束后无效: short={self.short_ma_period}, long={self.long_ma_period}")
            return False
        
        # More specific MA period difference check (example: long at least 10% longer or 5 periods longer)
        # This is now part of _ensure_parameter_constraints, but a final check here is fine.
        min_ma_ratio = 1.1 # Example: long must be at least 10% longer
        min_ma_abs_diff = 3 # Example: long must be at least 3 periods longer
        if not (self.long_ma_period >= self.short_ma_period * min_ma_ratio or \
                self.long_ma_period >= self.short_ma_period + min_ma_abs_diff):
            # This might be too strict if _ensure_parameter_constraints already handles it well
            # logger.warning(f"MA周期差异在约束后可能仍不理想: short={self.short_ma_period}, long={self.long_ma_period}")
            pass # Relax this specific validation if _ensure handles it robustly


        if not (isinstance(self.atr_period, int) and self.atr_period > 0):
            logger.error(f"ATR周期在约束后无效: {self.atr_period}")
            return False
        
        if not (isinstance(self.atr_stop_loss_multiplier, float) and
                isinstance(self.atr_take_profit_multiplier, float) and
                self.atr_stop_loss_multiplier > 0.09 and # SL multiplier should be meaningfully positive
                self.atr_take_profit_multiplier > self.atr_stop_loss_multiplier):
            logger.error(f"ATR乘数基本验证失败: SL={self.atr_stop_loss_multiplier}, TP={self.atr_take_profit_multiplier}")
            return False

        # More specific TP/SL ratio check (example: TP mult at least 1.2x SL mult)
        # This is now part of _ensure_parameter_constraints.
        min_tp_sl_mult_ratio = 1.2 # Example: TP mult should be at least 1.2 times SL mult
        if not (self.atr_take_profit_multiplier >= self.atr_stop_loss_multiplier * min_tp_sl_mult_ratio):
            # logger.warning(f"ATR TP/SL乘数比例在约束后可能仍不理想: SL={self.atr_stop_loss_multiplier}, TP={self.atr_take_profit_multiplier}")
            pass # Relax this specific validation if _ensure handles it robustly

        return True


    def _ensure_parameter_constraints(self):
        schema = config.GENE_SCHEMA
        current_genes = self.genes # Operate on the genes dict directly

        # --- MA周期约束 ---
        s_name, l_name = 'short_ma_period', 'long_ma_period'
        s_min, s_max = schema[s_name]['range']
        l_min, l_max = schema[l_name]['range']

        try: self.short_ma_period = int(current_genes[s_name])
        except (ValueError, TypeError, KeyError): self.short_ma_period = random.randint(s_min, (s_min + s_max) // 2)
        self.short_ma_period = max(s_min, min(self.short_ma_period, s_max))

        try: self.long_ma_period = int(current_genes[l_name])
        except (ValueError, TypeError, KeyError): self.long_ma_period = random.randint((l_min + l_max) // 2, l_max)
        self.long_ma_period = max(l_min, min(self.long_ma_period, l_max))

        # **NEW CONSTRAINT: Enforce significant difference between short and long MAs**
        min_ma_ratio_diff = 1.2  # Long MA period must be at least 20% greater than short MA
        min_ma_abs_diff = 5      # Long MA period must be at least 5 periods greater than short MA

        required_long_ma_by_ratio = int(self.short_ma_period * min_ma_ratio_diff)
        required_long_ma_by_abs = self.short_ma_period + min_ma_abs_diff
        
        self.long_ma_period = max(self.long_ma_period, required_long_ma_by_ratio, required_long_ma_by_abs)
        self.long_ma_period = min(self.long_ma_period, l_max) # Ensure it doesn't exceed long_ma_max range

        # If short_ma is now too close or greater than the adjusted long_ma, adjust short_ma down
        if self.short_ma_period >= self.long_ma_period:
            self.short_ma_period = max(s_min, min(self.short_ma_period, self.long_ma_period - min_ma_abs_diff))
            self.short_ma_period = max(s_min, min(self.short_ma_period, int(self.long_ma_period / min_ma_ratio_diff)))
        # Final fallback if still problematic (should be rare with above logic)
        if self.short_ma_period >= self.long_ma_period:
            self.short_ma_period = s_min 
            self.long_ma_period = max(int(s_min * min_ma_ratio_diff), s_min + min_ma_abs_diff)
            self.long_ma_period = min(self.long_ma_period, l_max)
            if self.short_ma_period >= self.long_ma_period: # Ultimate failsafe
                 self.long_ma_period = self.short_ma_period + min_ma_abs_diff


        current_genes[s_name] = self.short_ma_period
        current_genes[l_name] = self.long_ma_period

        # --- ATR周期约束 ---
        atr_p_name = 'atr_period'
        atr_p_min, atr_p_max = schema[atr_p_name]['range']
        try: self.atr_period = int(current_genes[atr_p_name])
        except (ValueError, TypeError, KeyError): self.atr_period = random.randint(atr_p_min, atr_p_max)
        self.atr_period = max(atr_p_min, min(self.atr_period, atr_p_max))
        current_genes[atr_p_name] = self.atr_period

        # --- ATR乘数约束 ---
        sl_mult_name, tp_mult_name = 'atr_stop_loss_multiplier', 'atr_take_profit_multiplier'
        sl_mult_min, sl_mult_max = schema[sl_mult_name]['range']
        tp_mult_min, tp_mult_max = schema[tp_mult_name]['range']

        try: self.atr_stop_loss_multiplier = float(current_genes[sl_mult_name])
        except (ValueError, TypeError, KeyError): self.atr_stop_loss_multiplier = random.uniform(sl_mult_min, (sl_mult_min+sl_mult_max)/2)
        self.atr_stop_loss_multiplier = max(sl_mult_min, min(self.atr_stop_loss_multiplier, sl_mult_max))

        try: self.atr_take_profit_multiplier = float(current_genes[tp_mult_name])
        except (ValueError, TypeError, KeyError): self.atr_take_profit_multiplier = random.uniform((tp_mult_min+tp_mult_max)/2, tp_mult_max)
        self.atr_take_profit_multiplier = max(tp_mult_min, min(self.atr_take_profit_multiplier, tp_mult_max))

        # **NEW CONSTRAINT: Enforce a minimum TP/SL ratio for multipliers**
        min_tp_sl_ratio_mult = 1.5  # TP multiplier must be at least 1.5x SL multiplier

        required_tp_mult = self.atr_stop_loss_multiplier * min_tp_sl_ratio_mult
        self.atr_take_profit_multiplier = max(self.atr_take_profit_multiplier, required_tp_mult)
        self.atr_take_profit_multiplier = min(self.atr_take_profit_multiplier, tp_mult_max) # Ensure within range

        # If SL is now too high relative to adjusted TP, adjust SL down
        if self.atr_stop_loss_multiplier >= self.atr_take_profit_multiplier :
             self.atr_stop_loss_multiplier = max(sl_mult_min, min(self.atr_stop_loss_multiplier, self.atr_take_profit_multiplier / min_tp_sl_ratio_mult))
        # Final fallback
        if self.atr_take_profit_multiplier <= self.atr_stop_loss_multiplier:
            self.atr_stop_loss_multiplier = sl_mult_min
            self.atr_take_profit_multiplier = max(tp_mult_min, sl_mult_min * min_tp_sl_ratio_mult)
            self.atr_take_profit_multiplier = min(self.atr_take_profit_multiplier, tp_mult_max)
            if self.atr_take_profit_multiplier <= self.atr_stop_loss_multiplier: # Ultimate failsafe
                 self.atr_take_profit_multiplier = self.atr_stop_loss_multiplier * min_tp_sl_ratio_mult


        current_genes[sl_mult_name] = self.atr_stop_loss_multiplier
        current_genes[tp_mult_name] = self.atr_take_profit_multiplier
        
        self.genes = current_genes # Assign back the modified genes dict

    def reset_state(self):
        self.position = None
        self.entry_price = 0.0
        self.entry_price_time = None
        # self.trades_count = 0 # Resetting this here might be incorrect if GA optimizer expects it to be cumulative per instance for some reason
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.atr_at_entry = None
        self.initial_calculated_stop_loss_price_for_sizing = 0.0
        # logger.debug(f"策略状态已重置 (Genes: short_ma={self.short_ma_period}, long_ma={self.long_ma_period})")

    def _calculate_indicators(self, df_klines: pd.DataFrame) -> pd.DataFrame:
        df = df_klines.copy()
        try:
            if self.short_ma_period is None or self.long_ma_period is None or self.atr_period is None:
                logger.error("策略参数未在指标计算前初始化。将尝试再次约束。")
                self._ensure_parameter_constraints()
                if not self._validate_parameters_after_constraints():
                     logger.critical("无法使用无效参数计算指标。原始基因: %s, 约束后: %s", self.genes_input, self.genes)
                     raise ValueError("无法使用无效参数计算指标。")

            df = self.indicators_calculator.add_sma(df, period=self.short_ma_period, column_name='close')
            df = self.indicators_calculator.add_sma(df, period=self.long_ma_period, column_name='close')
            df = self.indicators_calculator.add_atr(df, period=self.atr_period)
        except KeyError as e:
            logger.error(f"指标计算失败: DataFrame 缺少列 {e}. Genes: {self.genes}")
            return df_klines
        except ValueError as ve:
            logger.error(f"指标计算因参数无效失败: {ve}. Genes: {self.genes}")
            return df_klines
        except Exception as e_calc:
            logger.error(f"计算指标时发生意外错误: {e_calc}. Genes: {self.genes}", exc_info=True)
            return df_klines
        return df

    def generate_signals(self, df_klines_with_indicators: pd.DataFrame) -> Optional[str]:
        df = df_klines_with_indicators
        # Ensure enough data, considering the longest MA period and ATR period
        # The +2 is a common buffer for comparisons (e.g. previous vs current)
        min_len_for_signal = max(self.long_ma_period if self.long_ma_period else 0, 
                                 self.atr_period if self.atr_period else 0) + 2
        
        if df.empty or len(df) < min_len_for_signal:
            return None

        latest = df.iloc[-1]
        previous = df.iloc[-2] # df must have at least 2 rows for this

        sma_short_col = f'SMA_{self.short_ma_period}'
        sma_long_col = f'SMA_{self.long_ma_period}'
        atr_col = f'ATR_{self.atr_period}'

        # Check for NaN in required indicator values on latest and previous candles
        required_cols_latest = [sma_short_col, sma_long_col, atr_col, 'close', 'low', 'high', 'open']
        required_cols_previous = [sma_short_col, sma_long_col, 'close']

        for col in required_cols_latest:
            if col not in latest.index or pd.isna(latest[col]): # Check if col exists before accessing
                return None
        for col in required_cols_previous:
             if col not in previous.index or pd.isna(previous[col]):
                return None
        
        signal: Optional[str] = None
        current_price = float(latest['close'])
        current_open = float(latest['open'])
        current_low = float(latest['low'])
        current_high = float(latest['high'])
        current_atr = float(latest[atr_col])
        
        buy_condition_ma_cross = (previous[sma_short_col] < previous[sma_long_col]) and \
                                 (latest[sma_short_col] > latest[sma_long_col])
        
        if buy_condition_ma_cross and self.position is None:
            if current_atr > 0.000001:
                signal = 'buy_long'
                self.initial_calculated_stop_loss_price_for_sizing = current_price - (current_atr * self.atr_stop_loss_multiplier)
            else: 
                logger.warning(f"信号生成：在 {latest.name if hasattr(latest, 'name') else 'current time'} 金叉，但ATR无效({current_atr})。")
        
        if self.position == 'long':
            if self.stop_loss_price > 0 and (current_low <= self.stop_loss_price or current_open <= self.stop_loss_price):
                signal = 'close_long_sl'
            elif self.take_profit_price > 0 and (current_high >= self.take_profit_price or current_open >= self.take_profit_price):
                signal = 'close_long_tp'
            else:
                sell_condition_ma_cross = (previous[sma_short_col] > previous[sma_long_col]) and \
                                          (latest[sma_short_col] < latest[sma_long_col])
                if sell_condition_ma_cross:
                    signal = 'close_long_signal'
        return signal

    def update_position_after_trade(self, signal_type: str,
                                    entry_or_exit_price: float, 
                                    timestamp: pd.Timestamp,
                                    atr_value_at_entry: Optional[float] = None):
        if signal_type == 'buy_long':
            self.position = 'long'
            self.entry_price = entry_or_exit_price
            self.entry_price_time = timestamp
            self.trades_count += 1

            if atr_value_at_entry is not None and atr_value_at_entry > 0.000001:
                self.atr_at_entry = atr_value_at_entry 
                actual_sl_distance = self.atr_at_entry * self.atr_stop_loss_multiplier
                tp_distance = self.atr_at_entry * self.atr_take_profit_multiplier
                self.stop_loss_price = self.entry_price - actual_sl_distance
                self.take_profit_price = self.entry_price + tp_distance
            else:
                logger.error(f"策略买入更新错误: ATR无效 ({atr_value_at_entry})，无法设置SL/TP。Genes: {self.genes}")
                self.position = None 
                self.entry_price = 0.0
                self.entry_price_time = None
                self.trades_count = max(0, self.trades_count - 1) # Decrement if trade failed to set up
                self.stop_loss_price = 0.0
                self.take_profit_price = 0.0
                self.atr_at_entry = None
                return 

        elif signal_type.startswith('close_long'):
            self.position = None
            self.entry_price = 0.0
            self.entry_price_time = None
            self.atr_at_entry = None 
            self.stop_loss_price = 0.0 
            self.take_profit_price = 0.0
            self.initial_calculated_stop_loss_price_for_sizing = 0.0
