# trading_bot/backtester.py
import random
import pandas as pd
import numpy as np
import logging
import config
from typing import Optional, List, Tuple, Dict, Any
import copy # 确保导入 copy

logger = logging.getLogger(f"trading_bot.{__name__}")

class Backtester:
    def __init__(self, data_df: pd.DataFrame, strategy_instance,
                 initial_capital: float = config.INITIAL_CAPITAL,
                 fee_rate: float = config.FEE_RATE,
                 risk_per_trade_percent: float = config.RISK_PER_TRADE_PERCENT,
                 min_position_size: float = config.MIN_POSITION_SIZE):

        if data_df.empty:
             logger.critical("回测器初始化错误：传入的数据 DataFrame 为空。")
             raise ValueError("传入回测器的数据 DataFrame 不能为空。")

        self.raw_data_df = data_df.copy()
        self.strategy = strategy_instance # 策略实例现在应该已经约束过参数了

        try:
            logger.debug("回测器：开始计算指标...")
            self.data_df_with_indicators = self.strategy._calculate_indicators(self.raw_data_df)
            if self.data_df_with_indicators.empty:
                 logger.error("回测器：指标计算返回了空的 DataFrame。")
            elif len(self.data_df_with_indicators) != len(self.raw_data_df):
                 logger.warning(f"回测器：指标计算后 DataFrame 长度从 {len(self.raw_data_df)} 变为 {len(self.data_df_with_indicators)}。")
            logger.debug(f"回测器：指标计算完成。DataFrame 包含列: {self.data_df_with_indicators.columns.tolist()}")
        except Exception as e_indic:
            logger.critical(f"回测器：在初始化时计算指标失败: {e_indic}", exc_info=True)
            raise RuntimeError(f"回测器指标计算失败: {e_indic}") from e_indic

        self.initial_capital = float(initial_capital)
        self.fee_rate = float(fee_rate)
        self.risk_per_trade_percent = float(risk_per_trade_percent)
        self.min_position_size = float(min_position_size)

        self.position_active = False
        self.entry_price = 0.0
        self.position_size = 0.0
        self.entry_time = None
        self.atr_at_entry_for_trade: Optional[float] = None

        self.trades = []
        self.equity_curve = []
        self.current_equity = self.initial_capital

        logger.info(f"回测器初始化完成。初始资金: {self.initial_capital:.2f}, 手续费率: {self.fee_rate:.4f}, "
                    f"每笔交易风险: {self.risk_per_trade_percent:.2%}, 最小仓位: {self.min_position_size}")


    def _reset_backtest_state(self):
        self.position_active = False
        self.entry_price = 0.0
        self.position_size = 0.0
        self.entry_time = None
        self.atr_at_entry_for_trade = None
        self.trades = []
        self.current_equity = self.initial_capital
        self.equity_curve = [self.initial_capital] # 初始化权益曲线，第一个点是初始资金
        logger.debug("回测器状态已重置。")

        if hasattr(self.strategy, 'reset_state') and callable(self.strategy.reset_state):
            try:
                self.strategy.reset_state()
                logger.debug("策略实例状态已通过 reset_state() 重置。")
            except Exception as e_reset:
                 logger.error(f"调用 strategy.reset_state() 时出错: {e_reset}", exc_info=True)
        # else: # 之前这里有warning，但如果策略就是简单的不需要重置内部状态，这也不是问题
             # logger.debug("策略实例没有 'reset_state' 方法。")

    def _record_trade(self, exit_time: pd.Timestamp, exit_price: float, signal_exit: str):
        if not self.entry_time or self.entry_price <= 0 or self.position_size <= 0:
            logger.error(f"尝试记录交易错误：入场信息无效。Entry Time: {self.entry_time}, Entry Price: {self.entry_price}, Size: {self.position_size}")
            return

        pnl_per_unit = exit_price - self.entry_price
        gross_pnl = pnl_per_unit * self.position_size

        entry_value = self.entry_price * self.position_size
        exit_value = exit_price * self.position_size
        total_fees = abs(entry_value * self.fee_rate) + abs(exit_value * self.fee_rate)
        net_pnl = gross_pnl - total_fees

        trade_log_entry = {
            'entry_time': self.entry_time,
            'exit_time': exit_time,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'size': self.position_size,
            'pnl': net_pnl, # Store net_pnl, as this is what affects equity
            'gross_pnl': gross_pnl, # Store gross_pnl for reference
            'fee': total_fees,
            'type': 'long', # Assuming long only for now
            'signal_entry': 'buy_long',
            'signal_exit': signal_exit
        }
        self.trades.append(trade_log_entry)

        # prev_equity = self.current_equity # Not used currently
        self.current_equity += net_pnl

        # logger.debug(f"交易记录: Net PnL={net_pnl:.2f}, Gross PnL={gross_pnl:.2f}, Size={self.position_size:.4f}, Fees={total_fees:.2f}")


    def run_backtest(self) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
        self._reset_backtest_state()

        min_data_for_indicators = 0
        max_period_needed = 0
        if hasattr(self.strategy, 'long_ma_period') and self.strategy.long_ma_period is not None:
             max_period_needed = max(max_period_needed, self.strategy.long_ma_period)
        if hasattr(self.strategy, 'atr_period') and self.strategy.atr_period is not None:
             max_period_needed = max(max_period_needed, self.strategy.atr_period)
        min_data_for_indicators = max(max_period_needed, 20) + 5 # 需要最大周期 + 少量缓冲 (至少20)

        if self.data_df_with_indicators.empty or len(self.data_df_with_indicators) < min_data_for_indicators:
            logger.warning(f"回测中止：数据不足 ({len(self.data_df_with_indicators)} 条，需要约 {min_data_for_indicators} 条含有效指标数据)。")
            summary = self._generate_empty_summary()
            trades_df = pd.DataFrame(self.trades) # Might be empty
            if not trades_df.empty:
                 trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                 trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            return summary, trades_df

        logger.info(f"开始回测... 数据点: {len(self.data_df_with_indicators)}, "
                    f"起始时间: {self.data_df_with_indicators.index.min()}, "
                    f"结束时间: {self.data_df_with_indicators.index.max()}")

        start_index = min_data_for_indicators
        if start_index >= len(self.data_df_with_indicators):
             logger.warning("回测中止：起始索引大于或等于数据长度。")
             summary = self._generate_empty_summary()
             return summary, pd.DataFrame(self.trades) # Might be empty

        # self.equity_curve is already initialized with initial_capital in _reset_backtest_state

        for i in range(start_index, len(self.data_df_with_indicators)):
            data_slice_for_strategy = self.data_df_with_indicators.iloc[:i+1]
            current_kline = self.data_df_with_indicators.iloc[i]
            current_price_close = float(current_kline['close'])
            current_price_open = float(current_kline['open']) # 使用开盘价作为潜在成交价
            current_time = current_kline.name

            if len(data_slice_for_strategy) < 2:
                self.equity_curve.append(self.current_equity)
                continue

            try:
                signal = self.strategy.generate_signals(data_slice_for_strategy)
            except Exception as e_signal:
                 logger.error(f"在时间 {current_time} 生成信号时出错: {e_signal}", exc_info=True)
                 signal = None

            if signal == 'buy_long' and not self.position_active:
                entry_price_candidate = current_price_open

                initial_sl_price_estimate = self.strategy.initial_calculated_stop_loss_price_for_sizing

                if initial_sl_price_estimate <= 0 or initial_sl_price_estimate >= entry_price_candidate:
                    logger.debug(f"无法买入 @{current_time}: 策略未提供有效初始止损价或止损价高于入场价 ({initial_sl_price_estimate} vs {entry_price_candidate}).")
                elif self.current_equity <=0:
                    logger.warning(f"无法买入 @{current_time}: 当前净值为负或零 ({self.current_equity}).")
                else:
                    risk_per_unit = entry_price_candidate - initial_sl_price_estimate
                    if risk_per_unit <= 0.000001:
                        logger.debug(f"无法买入 @{current_time}: 每单位风险过小或为零 ({risk_per_unit}).")
                    else:
                        capital_at_risk = self.current_equity * self.risk_per_trade_percent
                        calculated_size = capital_at_risk / risk_per_unit
                        self.position_size = max(self.min_position_size, calculated_size)
                        
                        required_capital_for_trade = self.position_size * entry_price_candidate * (1 + self.fee_rate)
                        if self.current_equity < required_capital_for_trade * 0.1 :
                            logger.warning(f"无法买入 @{current_time}: 资金不足 ({self.current_equity:.2f}) "
                                           f"开仓 {self.position_size:.4f} 单位 (需约 {required_capital_for_trade:.2f})。")
                            self.position_size = 0.0
                        
                        if self.position_size >= self.min_position_size:
                            self.position_active = True
                            self.entry_price = entry_price_candidate
                            self.entry_time = current_time
                            
                            atr_col_name = f'ATR_{self.strategy.atr_period}'
                            atr_val_for_strategy_update = np.nan
                            if atr_col_name in current_kline and not pd.isna(current_kline[atr_col_name]) and current_kline[atr_col_name] > 0:
                                atr_val_for_strategy_update = float(current_kline[atr_col_name])
                            elif atr_col_name in data_slice_for_strategy.columns:
                                valid_atr_series = data_slice_for_strategy[atr_col_name].iloc[:-1].dropna()
                                if not valid_atr_series.empty and valid_atr_series.iloc[-1] > 0:
                                    atr_val_for_strategy_update = float(valid_atr_series.iloc[-1])
                            
                            if pd.isna(atr_val_for_strategy_update) or atr_val_for_strategy_update <= 0.000001:
                                logger.error(f"买入 @{current_time}: 无法获取有效的ATR值进行策略更新! (Val={atr_val_for_strategy_update}) 仓位可能无法正确设置SL/TP。")
                                self.atr_at_entry_for_trade = None
                            else:
                                 self.atr_at_entry_for_trade = atr_val_for_strategy_update

                            try:
                                self.strategy.update_position_after_trade(
                                    signal_type=signal,
                                    entry_or_exit_price=self.entry_price,
                                    timestamp=current_time,
                                    atr_value_at_entry=self.atr_at_entry_for_trade
                                )
                                if self.strategy.position is None:
                                     logger.debug(f"策略在 update_position_after_trade 中拒绝了买入。回滚回测器状态。")
                                     self.position_active = False; self.entry_price = 0.0; self.entry_time = None
                                     self.position_size = 0.0; self.atr_at_entry_for_trade = None
                                else:
                                     logger.debug(f"执行买入: Time={current_time}, Price={self.entry_price:.2f}, "
                                                  f"Size={self.position_size:.4f}, "
                                                  f"ATR@Entry={self.atr_at_entry_for_trade if self.atr_at_entry_for_trade else 'N/A'}")
                            except Exception as e_update:
                                 logger.error(f"调用 strategy.update_position_after_trade (买入) 时出错: {e_update}", exc_info=True)
                                 self.position_active = False; self.entry_price = 0.0; self.entry_time = None
                                 self.position_size = 0.0; self.atr_at_entry_for_trade = None
                        else:
                            if self.position_size < self.min_position_size and self.position_size > 0 :
                                logger.debug(f"想买入但计算出的仓位 {self.position_size:.6f} 小于最小允许仓位 {self.min_position_size:.6f}。不执行交易。")
                            self.position_size = 0.0

            elif signal in ['close_long_signal', 'close_long_sl', 'close_long_tp'] and self.position_active:
                exit_price_candidate = current_price_open
                
                if signal == 'close_long_sl' and self.strategy.stop_loss_price > 0:
                    if current_price_open <= self.strategy.stop_loss_price:
                        exit_price_candidate = current_price_open
                    elif current_kline['low'] <= self.strategy.stop_loss_price:
                        exit_price_candidate = self.strategy.stop_loss_price
                elif signal == 'close_long_tp' and self.strategy.take_profit_price > 0:
                    if current_price_open >= self.strategy.take_profit_price:
                        exit_price_candidate = current_price_open
                    elif current_kline['high'] >= self.strategy.take_profit_price:
                        exit_price_candidate = self.strategy.take_profit_price

                self._record_trade(exit_time=current_time, exit_price=exit_price_candidate, signal_exit=signal)
                
                self.position_active = False
                self.entry_price = 0.0
                self.position_size = 0.0
                self.entry_time = None
                self.atr_at_entry_for_trade = None

                try:
                    self.strategy.update_position_after_trade(
                        signal_type=signal,
                        entry_or_exit_price=exit_price_candidate,
                        timestamp=current_time
                    )
                    # logger.debug(f"执行平仓: Time={current_time}, Signal={signal}, Price={exit_price_candidate:.2f}")
                except Exception as e_update_close:
                     logger.error(f"调用 strategy.update_position_after_trade (平仓) 时出错: {e_update_close}", exc_info=True)

            equity_at_this_kline_close = self.current_equity
            if self.position_active:
                unrealized_pnl_per_unit = current_price_close - self.entry_price
                unrealized_gross_pnl = unrealized_pnl_per_unit * self.position_size
                # For equity curve, we consider net PnL if position were closed now
                # This requires estimating exit fee, which is an approximation
                # A simpler way is to just use gross PnL for unrealized equity for consistency
                # Or, more accurately, current_equity (which is after last closed trade) + unrealized_gross_pnl
                # Let's use current_equity + unrealized_gross_pnl - estimated_fee_if_closed_now
                entry_fee_paid_for_this_trade = abs(self.entry_price * self.position_size * self.fee_rate)
                potential_exit_fee = abs(current_price_close * self.position_size * self.fee_rate)
                # current_equity already accounts for the entry fee if it was deducted from capital immediately
                # For simplicity and common practice: equity = initial_capital + sum_of_closed_trade_net_pnl + current_open_trade_gross_pnl - open_trade_entry_fee - potential_exit_fee
                # However, our self.current_equity already reflects closed trades' net pnl.
                # So, unrealized part is: gross_pnl_open_trade - fee_for_open_trade_entry - fee_for_potential_exit
                
                # Revised equity calculation:
                # self.current_equity is the equity *after* the last closed trade.
                # If a position is active, its mark-to-market value contributes to the total equity.
                # Mark-to-market value = current_value_of_position - entry_value_of_position
                #                        = (current_price_close * size) - (entry_price * size)
                #                        = unrealized_gross_pnl
                # The cost of opening this position (fees) has already been "paid" from current_equity
                # if fees are deducted from available capital upon trade.
                # Our _record_trade updates current_equity by net_pnl.
                # So, equity_at_this_kline_close = current_equity_before_this_open_trade_result + unrealized_net_pnl_of_open_trade
                # This is complex. Let's simplify:
                # self.equity_curve should reflect the value if we liquidated everything at current_kline['close']
                
                # Current self.current_equity is based on the last *closed* trade.
                # The value of the open position is (current_price_close - self.entry_price) * self.position_size (gross)
                # Minus fees for this open position (entry fee already paid from equity, potential exit fee)
                unrealized_net_pnl = (current_price_close - self.entry_price) * self.position_size \
                                   - (abs(self.entry_price * self.position_size * self.fee_rate) + \
                                      abs(current_price_close * self.position_size * self.fee_rate))
                
                # Let's use a simpler approach for equity curve for now:
                # Equity = Initial Capital + Sum of Net PnL of all closed trades + Net PnL of current open trade if closed at current_price_close
                sum_closed_net_pnl = sum(t['pnl'] for t in self.trades)
                
                open_trade_net_pnl_if_closed = 0
                if self.position_active:
                    open_trade_gross_pnl = (current_price_close - self.entry_price) * self.position_size
                    open_trade_total_fees = abs(self.entry_price * self.position_size * self.fee_rate) + \
                                           abs(current_price_close * self.position_size * self.fee_rate)
                    open_trade_net_pnl_if_closed = open_trade_gross_pnl - open_trade_total_fees
                
                equity_at_this_kline_close = self.initial_capital + sum_closed_net_pnl + open_trade_net_pnl_if_closed

            self.equity_curve.append(equity_at_this_kline_close)


        logger.info(f"回测循环结束。总交易次数: {len(self.trades)}")
        summary = self._calculate_summary_metrics()
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        return summary, trades_df


    def _generate_empty_summary(self) -> dict:
        """生成一个空的/默认的回测摘要字典。"""
        return {
            "Initial Capital": self.initial_capital,
            "Final Capital": self.initial_capital,
            "Total PnL": 0.0,
            "Total PnL Pct": 0.0,
            "Total Trades": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Win Rate": 0.0,
            "Max Drawdown": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Profit Factor": 0.0,
            "Avg PnL per Trade": 0.0,
            "Avg Holding Period": str(pd.Timedelta(0)),
            "Total Fees Paid": 0.0,
            "Avg Win/Loss Ratio": 0.0, # New
            "Max Consecutive Wins": 0,  # New
            "Max Consecutive Losses": 0, # New
            "Calmar Ratio": 0.0, # New
            "SQN": 0.0 # New
        }

    def _calculate_periodic_returns(self, kline_bar_for_resample: Optional[str] = 'D') -> pd.Series:
        if not self.equity_curve or len(self.equity_curve) < 2:
            return pd.Series(dtype=float)

        # Equity curve starts with initial capital, then one point per kline *after* the warmup period.
        # The timestamps for these equity points correspond to the klines from self.data_df_with_indicators
        # starting from the point where backtest loop begins.
        
        # The first point in self.equity_curve is initial capital (before any trading decisions on data)
        # The subsequent points correspond to equity at the close of each kline in the trading loop
        
        # Determine the number of klines used in the actual trading loop
        # This should match len(self.equity_curve) - 1 (because first element is initial capital)
        num_trading_klines = len(self.equity_curve) -1
        if num_trading_klines < 1:
            return pd.Series(dtype=float)

        # Get the timestamps for these trading klines
        # The trading loop starts from `start_index` in `self.data_df_with_indicators`
        # `start_index` is `min_data_for_indicators`
        
        max_period_needed = 0
        if hasattr(self.strategy, 'long_ma_period') and self.strategy.long_ma_period is not None:
             max_period_needed = max(max_period_needed, self.strategy.long_ma_period)
        if hasattr(self.strategy, 'atr_period') and self.strategy.atr_period is not None:
             max_period_needed = max(max_period_needed, self.strategy.atr_period)
        min_data_for_indicators_calc = max(max_period_needed, 20) + 5
        
        # Timestamps for equity curve points (excluding the initial capital point)
        # should align with the klines from `min_data_for_indicators_calc` onwards.
        if len(self.data_df_with_indicators) < min_data_for_indicators_calc + num_trading_klines :
            logger.warning(f"周期收益率计算：数据长度不足以对齐所有净值点。DF len={len(self.data_df_with_indicators)}, "
                           f"Warmup={min_data_for_indicators_calc}, TradingKlines={num_trading_klines}")
            # Attempt to align what we can
            actual_num_trading_klines_can_align = len(self.data_df_with_indicators) - min_data_for_indicators_calc
            if actual_num_trading_klines_can_align < 1 : return pd.Series(dtype=float)
            
            equity_timestamps = self.data_df_with_indicators.index[min_data_for_indicators_calc : min_data_for_indicators_calc + actual_num_trading_klines_can_align]
            valid_equity_points = self.equity_curve[1 : 1 + actual_num_trading_klines_can_align] # Corresponding equity points
        else:
            equity_timestamps = self.data_df_with_indicators.index[min_data_for_indicators_calc : min_data_for_indicators_calc + num_trading_klines]
            valid_equity_points = self.equity_curve[1:] # All equity points after initial capital

        if len(valid_equity_points) != len(equity_timestamps):
            logger.warning(f"周期收益率计算：净值点数 ({len(valid_equity_points)}) 与时间戳数量 ({len(equity_timestamps)}) 不匹配。将使用最小长度。")
            min_len = min(len(valid_equity_points), len(equity_timestamps))
            if min_len < 1: return pd.Series(dtype=float) # Need at least one point to potentially get returns later
            valid_equity_points = valid_equity_points[:min_len]
            equity_timestamps = equity_timestamps[:min_len]

        if not valid_equity_points or len(valid_equity_points) < 1: # Need at least 1 point for Series creation, 2 for pct_change
            return pd.Series(dtype=float)

        equity_series_for_returns = pd.Series(valid_equity_points, index=equity_timestamps)
        equity_series_for_returns = equity_series_for_returns.dropna() # Drop if any NaN equity points somehow occurred

        if len(equity_series_for_returns) < 2: # Need at least 2 points for pct_change
             return pd.Series(dtype=float)

        resampled_equity = equity_series_for_returns
        if kline_bar_for_resample:
            try:
                # For daily returns, we want the equity at the end of each day.
                resampled_equity = equity_series_for_returns.resample(kline_bar_for_resample).last().ffill()
            except Exception as e_resample:
                logger.warning(f"周期收益率计算：重采样到 {kline_bar_for_resample} 失败: {e_resample}。将使用原始K线频率。")
                # Fallback: use original frequency if resampling fails
                # resampled_equity remains equity_series_for_returns
        
        resampled_equity = resampled_equity.dropna()
        if len(resampled_equity) < 2:
            return pd.Series(dtype=float)

        periodic_returns = resampled_equity.pct_change().dropna()
        if periodic_returns.empty:
            return pd.Series(dtype=float)
            
        return periodic_returns


    def _calculate_summary_metrics(self) -> dict:
        if not self.equity_curve or len(self.equity_curve) <= 1:
            logger.warning("无法计算摘要指标：净值曲线数据不足。")
            return self._generate_empty_summary()

        final_capital = float(self.equity_curve[-1])
        total_pnl = float(final_capital - self.initial_capital)
        total_pnl_pct = (total_pnl / self.initial_capital) if self.initial_capital > 1e-9 else 0.0
        num_total_trades = len(self.trades)

        pnl_values = [float(trade['pnl']) for trade in self.trades if 'pnl' in trade and trade['pnl'] is not None]
        total_fees = sum(float(trade['fee']) for trade in self.trades if 'fee' in trade and trade['fee'] is not None)

        winning_trades_pnl = [pnl for pnl in pnl_values if pnl > 1e-9] # Consider practically > 0
        losing_trades_pnl = [pnl for pnl in pnl_values if pnl < -1e-9] # Consider practically < 0

        num_winning_trades = len(winning_trades_pnl)
        num_losing_trades = len(losing_trades_pnl)
        win_rate = (num_winning_trades / num_total_trades) * 100.0 if num_total_trades > 0 else 0.0
        avg_pnl_per_trade = total_pnl / num_total_trades if num_total_trades > 0 else 0.0

        gross_profit = sum(winning_trades_pnl)
        gross_loss = abs(sum(losing_trades_pnl))
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float('inf') if gross_profit > 1e-9 else 0.0
        
        avg_winning_pnl = np.mean(winning_trades_pnl) if num_winning_trades > 0 else 0.0
        avg_losing_pnl = np.mean(losing_trades_pnl) if num_losing_trades > 0 else 0.0 # Will be negative or zero
        avg_win_loss_ratio = abs(avg_winning_pnl / avg_losing_pnl) if avg_losing_pnl < -1e-9 else float('inf') if avg_winning_pnl > 1e-9 else 0.0


        equity_series = pd.Series(self.equity_curve)
        # Ensure no NaNs or Infs in equity_series for cummax and drawdown calculations
        equity_series = equity_series.replace([np.inf, -np.inf], np.nan).ffill().bfill() # Fill Infs then NaNs
        if equity_series.isna().any() or equity_series.empty: # If still problematic
            max_dd_pct = 1.0 # Max possible drawdown if equity is problematic
        else:
            running_max = equity_series.cummax()
            absolute_drawdown = running_max - equity_series
            # Ensure running_max is not zero or negative before division
            # Replace 0s in running_max with a small positive number or handle division by zero
            safe_running_max = running_max.replace(0, 1e-9) # Avoid division by zero
            percentage_drawdowns = absolute_drawdown / safe_running_max
            max_dd_pct = float(percentage_drawdowns.max()) if not percentage_drawdowns.empty else 0.0
            
        # Max Consecutive Wins/Losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_consecutive_wins = 0
        current_consecutive_losses = 0
        if pnl_values:
            for pnl_val in pnl_values:
                if pnl_val > 1e-9:
                    current_consecutive_wins += 1
                    current_consecutive_losses = 0
                elif pnl_val < -1e-9:
                    current_consecutive_losses += 1
                    current_consecutive_wins = 0
                else: # Break-even
                    current_consecutive_wins = 0
                    current_consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)

        holding_periods = []
        if self.trades:
            for trade in self.trades:
                entry_t = trade.get('entry_time')
                exit_t = trade.get('exit_time')
                if isinstance(entry_t, pd.Timestamp) and isinstance(exit_t, pd.Timestamp):
                    holding_periods.append(exit_t - entry_t)
        avg_holding_period = sum(holding_periods, pd.Timedelta(0)) / len(holding_periods) if holding_periods else pd.Timedelta(0)

        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        calmar_ratio = 0.0
        annualized_return_pct = 0.0

        # Use daily returns for Sharpe, Sortino, Calmar
        daily_returns = self._calculate_periodic_returns(kline_bar_for_resample='D')
        if daily_returns.empty and len(self.equity_curve) > 25: # Fallback to hourly if daily is empty but enough data exists
            logger.debug("Daily returns empty, trying with original (hourly) frequency for Sharpe/Sortino.")
            original_freq_returns = self._calculate_periodic_returns(kline_bar_for_resample=None)
            # If using hourly, need to annualize differently (e.g., sqrt(24*365) for Sharpe if returns are for each hour)
            # For simplicity, if daily fails, we might just report 0 or try to estimate from overall PnL
            # Let's stick to daily for now or report 0 if daily is problematic.
            # If we use original_freq_returns, need to scale mean and std.
            # For now, if daily_returns is empty, Sharpe/Sortino will remain 0.
            pass


        if not daily_returns.empty and len(daily_returns) >= 2: # Need at least 2 periods for std dev
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()

            # Annualized Return
            if len(self.data_df_with_indicators.index) > 0:
                days_in_backtest = (self.data_df_with_indicators.index[-1] - self.data_df_with_indicators.index[0]).days
                if days_in_backtest > 0:
                    annualized_factor = 365.0 / days_in_backtest
                    annualized_return_pct = ((1 + total_pnl_pct)**annualized_factor - 1) * 100.0 if total_pnl_pct is not None else 0.0
                else: # Less than a day
                    annualized_return_pct = total_pnl_pct * 100.0 # Not really annualized
            
            if std_daily_return > 1e-9 and not (np.isnan(std_daily_return) or np.isinf(std_daily_return)):
                sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252) # Assuming 252 trading days for annualization

                negative_daily_returns = daily_returns[daily_returns < 0]
                if not negative_daily_returns.empty:
                    downside_deviation_daily = negative_daily_returns.std()
                    if downside_deviation_daily > 1e-9 and not (np.isnan(downside_deviation_daily) or np.isinf(downside_deviation_daily)):
                        sortino_ratio = (mean_daily_return / downside_deviation_daily) * np.sqrt(252)
            
            if max_dd_pct > 1e-9: # Max DD is already in percentage, convert to decimal
                calmar_ratio = (annualized_return_pct / 100.0) / max_dd_pct # Use decimal MaxDD for Calmar
            
        else:
            logger.debug(f"无法计算夏普/索提诺/卡玛比率：日收益率数据不足 ({len(daily_returns)} < 2) 或标准差无效。")

        # SQN
        sqn_value = 0.0
        if num_total_trades > 1 and pnl_values:
            std_dev_pnl = np.std(pnl_values)
            if std_dev_pnl > 1e-9:
                 sqn_value = (np.sqrt(num_total_trades) * avg_pnl_per_trade) / std_dev_pnl
        
        summary = {
            "Initial Capital": self.initial_capital,
            "Final Capital": float(final_capital),
            "Total PnL": float(total_pnl),
            "Total PnL Pct": float(total_pnl_pct * 100.0),
            "Total Trades": num_total_trades,
            "Winning Trades": num_winning_trades,
            "Losing Trades": num_losing_trades,
            "Win Rate": float(win_rate),
            "Max Drawdown": float(max_dd_pct * 100.0), # Store as percentage
            "Sharpe Ratio": float(sharpe_ratio) if not (np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio)) else 0.0,
            "Sortino Ratio": float(sortino_ratio) if not (np.isnan(sortino_ratio) or np.isinf(sortino_ratio)) else 0.0,
            "Profit Factor": float(profit_factor) if not (np.isnan(profit_factor) or np.isinf(profit_factor)) else 0.0,
            "Avg PnL per Trade": float(avg_pnl_per_trade),
            "Avg Holding Period": str(avg_holding_period).split('.')[0], # Remove microseconds
            "Total Fees Paid": float(total_fees),
            "Avg Win/Loss Ratio": float(avg_win_loss_ratio) if not (np.isnan(avg_win_loss_ratio) or np.isinf(avg_win_loss_ratio)) else 0.0,
            "Max Consecutive Wins": max_consecutive_wins,
            "Max Consecutive Losses": max_consecutive_losses,
            "Annualized Return Pct": float(annualized_return_pct) if not (np.isnan(annualized_return_pct) or np.isinf(annualized_return_pct)) else 0.0,
            "Calmar Ratio": float(calmar_ratio) if not (np.isnan(calmar_ratio) or np.isinf(calmar_ratio)) else 0.0,
            "SQN": float(sqn_value) if not (np.isnan(sqn_value) or np.isinf(sqn_value)) else 0.0
        }
        return summary

    def print_summary(self, summary_dict: Optional[Dict[str, Any]]):
        if summary_dict is None:
            logger.info("没有提供回测摘要信息用于打印。")
            return

        print("\n--- Backtest Summary ---")
        print(f"{'Metric':<28}: {'Value'}") # Increased width for metric name
        print("-" * 40) # Adjusted separator
        for key, value in summary_dict.items():
            if isinstance(value, float):
                if "Pct" in key or "Win Rate" in key or "Max Drawdown" in key:
                    print(f"{key:<28}: {value:>10.2f}%")
                elif "Ratio" in key or "Factor" in key or "SQN" in key: # Added SQN here
                    print(f"{key:<28}: {value:>10.3f}") # Using 3 decimal places for ratios
                else:
                    print(f"{key:<28}: {value:>10.2f}")
            elif isinstance(value, int):
                 print(f"{key:<28}: {value:>10d}")
            else:
                 print(f"{key:<28}: {str(value):>10}")
        print("-" * 40)


if __name__ == '__main__':
    print("Testing Backtester independently (with new metrics and stability improvements)...")
    if not logger.hasHandlers():
        ch_test_bt = logging.StreamHandler()
        formatter_bt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch_test_bt.setFormatter(formatter_bt)
        logger.addHandler(ch_test_bt)
        logger.setLevel(logging.DEBUG) # Keep DEBUG for testing
        logging.getLogger('trading_bot.strategy').setLevel(logging.INFO) # Reduce strategy noise for this test

    try:
        from okx_connector import OKXConnector
        from data_handler import DataHandler
        from strategy import EvolvableStrategy
    except ImportError as e_imp:
        print(f"Backtester Test Error: Failed to import required modules: {e_imp}")
        exit()

    try:
        okx_conn_test = OKXConnector()
        data_hdl_test = DataHandler(okx_conn_test)

        test_data_limit = 700 # Increased slightly for more robust testing of new metrics
        print(f"Fetching {test_data_limit} klines for backtester test...")
        test_df = data_hdl_test.fetch_klines_to_df(
            instId=config.SYMBOL,
            bar=config.TIMEFRAME,
            total_limit_needed=test_data_limit,
            kline_type="history_market"
        )

        if test_df.empty or len(test_df) < 200: # Increased min data for test
            print(f"Not enough data for backtester standalone test (fetched {len(test_df)}). Exiting.")
            exit()
        else:
            print(f"Fetched {len(test_df)} klines for backtester test.")

        # Use a fixed, reasonable gene set for consistent testing
        sample_genes_fixed = {
            'short_ma_period': 10,
            'long_ma_period': 30,  # Ensure difference from short_ma
            'atr_period': 14,
            'atr_stop_loss_multiplier': 2.0,
            'atr_take_profit_multiplier': 3.5 # Ensures TP > SL ratio
        }
        print(f"Using fixed genes for test (before strategy constraint): {sample_genes_fixed}")

        try:
            test_strategy_instance = EvolvableStrategy(copy.deepcopy(sample_genes_fixed)) # Apply constraints
            print(f"Created strategy instance with constrained genes: {test_strategy_instance.genes}")
        except Exception as e_strat_init:
             print(f"Failed to create strategy instance: {e_strat_init}")
             exit()

        backtester_instance = None # Define outside try
        summary_results, trades_log = None, None # Define outside try

        try:
            backtester_instance = Backtester(
                data_df=test_df,
                strategy_instance=test_strategy_instance,
                initial_capital=config.INITIAL_CAPITAL,
                fee_rate=config.FEE_RATE,
                risk_per_trade_percent=config.RISK_PER_TRADE_PERCENT,
                min_position_size=config.MIN_POSITION_SIZE
            )
            summary_results, trades_log = backtester_instance.run_backtest()
        except RuntimeError as re_bt_run:
             print(f"Backtester run failed with RuntimeError: {re_bt_run}")
        except Exception as e_bt_run:
             print(f"Backtester run failed with unexpected error: {e_bt_run}")
             import traceback; traceback.print_exc()

        if summary_results:
            backtester_instance.print_summary(summary_results)
        else:
             print("\nBacktest did not produce a summary.")

        if trades_log is not None and not trades_log.empty:
            print(f"\nTrades Log ({len(trades_log)} trades, first 5):")
            print(trades_log.head())
        else:
            print("\nNo trades were executed in this backtest run.")

        # --- Test for stability: Run again with same data and strategy ---
        if backtester_instance and test_strategy_instance: # If first run was successful enough to create instances
            print("\n--- Running Backtest Again for Stability Check ---")
            # Re-initialize strategy or ensure its state is reset if it's stateful beyond what backtester resets
            # EvolvableStrategy.reset_state() is called by backtester._reset_backtest_state()
            # So, just re-running backtest on the same backtester instance should be fine.
            # However, to be absolutely sure for testing, let's create a new strategy instance too.
            try:
                test_strategy_instance_run2 = EvolvableStrategy(copy.deepcopy(sample_genes_fixed))
                # Create a new backtester instance with a fresh copy of data to ensure no state leakage from prev run
                backtester_instance_run2 = Backtester(
                    data_df=test_df.copy(), # Use a fresh copy of data
                    strategy_instance=test_strategy_instance_run2,
                    initial_capital=config.INITIAL_CAPITAL,
                    fee_rate=config.FEE_RATE,
                    risk_per_trade_percent=config.RISK_PER_TRADE_PERCENT,
                    min_position_size=config.MIN_POSITION_SIZE
                )
                summary_results_run2, _ = backtester_instance_run2.run_backtest()
                if summary_results_run2:
                    print("Second run summary:")
                    backtester_instance_run2.print_summary(summary_results_run2)
                    
                    # Compare summaries (simple check for key metrics)
                    if summary_results and summary_results_run2:
                        mismatched_metrics = []
                        for key in summary_results:
                            val1 = summary_results[key]
                            val2 = summary_results_run2.get(key)
                            if isinstance(val1, float) and isinstance(val2, float):
                                if not np.isclose(val1, val2, rtol=1e-5, atol=1e-8): # Allow tiny float differences
                                    mismatched_metrics.append(f"{key} (Run1: {val1}, Run2: {val2})")
                            elif val1 != val2:
                                mismatched_metrics.append(f"{key} (Run1: {val1}, Run2: {val2})")
                        
                        if not mismatched_metrics:
                            print("\nSUCCESS: Key metrics in summary are consistent across two identical runs.")
                        else:
                            print("\nWARNING: Metrics mismatch between two identical runs:")
                            for m in mismatched_metrics: print(f"  - {m}")
                else:
                    print("Second backtest run did not produce a summary.")
            except Exception as e_run2:
                print(f"Error during second backtest run for stability check: {e_run2}")
                import traceback; traceback.print_exc()

    except ValueError as ve:
        print(f"Backtester Test Error (ValueError): {ve}")
    except Exception as e:
        print(f"Backtester Test Error (Exception): {e}")
        import traceback
        traceback.print_exc()

    print("\nBacktester standalone test finished.")
        import traceback
        traceback.print_exc()
    print("\nBacktester standalone test finished.")
