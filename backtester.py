# trading_bot/backtester.py
import random
import pandas as pd
import numpy as np
import config

class Backtester:
    def __init__(self, data_df: pd.DataFrame, strategy_instance, 
                 initial_capital: float = config.INITIAL_CAPITAL, 
                 fee_rate: float = config.FEE_RATE):
        self.raw_data_df = data_df.copy() # 保存传入的原始数据帧的副本
        self.strategy = strategy_instance
        # 策略计算指标时应该使用原始数据的副本，避免修改传入的df
        self.data_df_with_indicators = self.strategy._calculate_indicators(self.raw_data_df.copy())

        self.initial_capital = float(initial_capital) # 确保初始资金是浮点数
        self.fee_rate = float(fee_rate) # 确保费率是浮点数

        # --- 回测过程中的状态变量 ---
        self.position_active = False # 当前是否有持仓
        self.entry_price = 0.0       # 开仓价格
        self.position_size = 0.0     # 持仓数量 (例如合约张数或币的数量)
        self.entry_time = None       # 开仓时间 (pd.Timestamp)

        self.trades = []             # 记录每一笔完成的交易 (开仓+平仓)
        self.equity_curve = []       # 记录每个K线周期结束时的账户净值
        self.current_equity = self.initial_capital # 动态跟踪当前净值，主要通过已实现盈亏更新

    def _record_trade(self, exit_time: pd.Timestamp, exit_price: float, signal_exit: str):
        """记录一笔完成的交易，并更新当前净值"""
        pnl_per_unit = exit_price - self.entry_price # 每单位的毛利润/亏损
        gross_pnl = pnl_per_unit * self.position_size
        
        entry_value = self.entry_price * self.position_size
        exit_value = exit_price * self.position_size
        # 双边手续费
        total_fees = (entry_value * self.fee_rate) + (exit_value * self.fee_rate)
        net_pnl = gross_pnl - total_fees # 净利润/亏损

        self.trades.append({
            'entry_time': self.entry_time,          # 开仓时间
            'exit_time': exit_time,            # 平仓时间
            'entry_price': self.entry_price,      # 开仓价格
            'exit_price': exit_price,          # 平仓价格
            'size': self.position_size,        # 交易数量
            'pnl': net_pnl,                    # 净盈亏
            'type': 'long',                    # 交易类型 (目前假设只做多)
            'signal_entry': 'buy_long',        # 开仓信号 (可以从策略获取更具体的)
            'signal_exit': signal_exit         # 平仓信号
        })
        self.current_equity += net_pnl # CRITICAL: 用已实现的净盈亏更新当前净值

    def run_backtest(self):
        """执行回测"""
        # 确定计算指标所需的最少数据量
        min_data_for_indicators = 0
        # 基于策略可能使用的最长周期来估计
        if hasattr(self.strategy, 'long_ma_period'):
             min_data_for_indicators = max(min_data_for_indicators, self.strategy.long_ma_period)
        if hasattr(self.strategy, 'atr_period'): # 新增ATR周期考虑
             min_data_for_indicators = max(min_data_for_indicators, self.strategy.atr_period)
        elif hasattr(self.strategy, 'rsi_period'): # 如果没有均线或ATR，看RSI周期 (当前版本未使用RSI)
             min_data_for_indicators = max(min_data_for_indicators, self.strategy.rsi_period)
        
        min_data_for_indicators = max(min_data_for_indicators, 20) + 10 # 至少需要一些数据，并加一些buffer (例如10)

        if self.data_df_with_indicators.empty or len(self.data_df_with_indicators) < min_data_for_indicators:
            # print(f"Warning: Not enough data ({len(self.data_df_with_indicators)} vs {min_data_for_indicators} needed) for backtesting.")
            summary = self._generate_empty_summary()
            # 返回空的 trades DataFrame
            return summary, pd.DataFrame(columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'size', 'pnl', 'type', 'signal_entry', 'signal_exit'])

        # --- 为每次回测重置状态 ---
        self.position_active = False
        self.entry_price = 0.0
        self.position_size = 0.0
        self.entry_time = None
        self.trades = []
        self.current_equity = self.initial_capital # 重置当前净值
        self.equity_curve = [self.initial_capital] # 净值曲线以初始资金开始
        
        # 重置策略实例的内部状态 (非常重要)
        self.strategy.position = None
        self.strategy.entry_price = 0.0
        self.strategy.entry_price_time = None
        self.strategy.trades_count = 0
        # 如果策略中有其他需要重置的状态 (如ATR相关的 stop_loss_price, take_profit_price)
        if hasattr(self.strategy, 'stop_loss_price'): self.strategy.stop_loss_price = 0.0
        if hasattr(self.strategy, 'take_profit_price'): self.strategy.take_profit_price = 0.0
        if hasattr(self.strategy, 'atr_at_entry'): self.strategy.atr_at_entry = None
        # --- 状态重置结束 ---

        # 从第二根K线开始迭代 (索引1)，因为很多策略需要比较前一根K线
        # 或者从 min_data_for_indicators 开始，确保所有指标都已充分计算
        start_index = min_data_for_indicators -1 # 确保切片时有足够的前置数据供指标计算
        if start_index < 1: start_index = 1 # 至少从索引1开始，因为要比较 current 和 previous

        for i in range(start_index, len(self.data_df_with_indicators)):
            # 提供给策略的数据是到当前K线(包含)为止的所有数据
            # 注意：iloc的尾部是不包含的，所以 i+1 会取到索引 i 的行
            current_kline_data_slice = self.data_df_with_indicators.iloc[:i+1] 
            current_kline = self.data_df_with_indicators.iloc[i] # 当前正在处理的K线
            current_price = float(current_kline['close'])      # 以当前K线的收盘价作为执行价格（简化）
            current_time = current_kline.name                  # K线的时间戳 (DatetimeIndex)

            # 确保传递给策略的切片至少有2行数据，用于比较 latest 和 previous
            if len(current_kline_data_slice) < 2: 
                self.equity_curve.append(self.current_equity) 
                continue

            signal = self.strategy.generate_signals(current_kline_data_slice)
            
            # 处理买入信号
            if signal == 'buy_long' and not self.position_active: # 如果是买入信号且当前无持仓
                self.position_active = True
                self.entry_price = current_price # 实际入场价
                self.entry_time = current_time
                # 假设固定交易1单位。在更复杂的系统中，这里会有仓位管理逻辑。
                self.position_size = 1.0 
                # 调用策略的 update_position_after_trade，传递实际入场价
                self.strategy.update_position_after_trade(signal_type=signal, entry_or_exit_price=self.entry_price, timestamp=current_time)
                # print(f"DEBUG Backtester: {current_time} BUY LONG at {self.entry_price}, size {self.position_size}")

            # 处理平仓信号
            # 确保 self.strategy.position 也被检查，以防策略内部状态与回测器状态不一致 (虽然理论上应该一致)
            elif signal in ['close_long_signal', 'close_long_sl', 'close_long_tp'] and self.position_active: # and self.strategy.position == 'long'
                exit_price = current_price # 实际平仓价
                self._record_trade(exit_time=current_time, exit_price=exit_price, signal_exit=signal)
                
                self.position_active = False # 平仓后重置持仓状态
                # self.entry_price, self.position_size, self.entry_time 会在下次开仓时重新设置
                # 调用策略的 update_position_after_trade，传递实际平仓价
                self.strategy.update_position_after_trade(signal_type=signal, entry_or_exit_price=exit_price, timestamp=current_time)
                # print(f"DEBUG Backtester: {current_time} CLOSE LONG ({signal}) at {exit_price}, PnL for trade recorded.")
            
            # 更新每个K线结束时的净值曲线
            if self.position_active: # 如果仍然持有多仓 (此时 self.strategy.position 应该是 'long')
                # 注意：这里的 entry_price 是回测器记录的该笔交易的入场价
                unrealized_pnl = (current_price - self.entry_price) * self.position_size 
                # 这里的 current_equity 是基于已实现交易的净值
                equity_at_this_kline_close = self.current_equity + unrealized_pnl 
                self.equity_curve.append(equity_at_this_kline_close)
            else: # 如果空仓
                self.equity_curve.append(self.current_equity) # 净值等于已实现的总盈亏 + 初始资金
        
        summary = self._calculate_summary_metrics()
        trades_df = pd.DataFrame(self.trades)
        return summary, trades_df

    def _generate_empty_summary(self) -> dict:
        """生成一个空的/默认的回测摘要字典"""
        return {
            "Initial Capital": self.initial_capital, 
            "Final Capital": self.initial_capital,
            "Total PnL": 0.0, 
            "Total Trades": 0, 
            "Winning Trades": 0, 
            "Losing Trades": 0,
            "Win Rate": 0.0, 
            "Max Drawdown": 0.0, 
            "Sharpe Ratio": 0.0, 
            "Sortino Ratio": 0.0,
            "Avg PnL per Trade": 0.0, 
            "Profit Factor": 0.0, 
            "Avg Holding Period": str(pd.Timedelta(0)) # 使用字符串格式
        }

    def _calculate_periodic_returns(self, kline_bar_for_resample:str = '1D') -> pd.Series:
        """根据净值曲线计算周期性收益率"""
        if len(self.equity_curve) < 2: 
            return pd.Series(dtype=float)
        
        equity_timestamps = self.data_df_with_indicators.index
        
        # equity_curve[0] 是初始资金，对应于第一根K线开始之前
        # equity_curve[1:] 对应于每根K线结束时的净值
        # 我们需要确保 equity_series 的长度与有效的时间戳数量匹配
        
        # 有效的净值点 (从第一根K线收盘后开始)
        valid_equity_points = self.equity_curve[1:]
        # 对应的有效时间戳 (假设 equity_curve 的更新与K线一一对应)
        # 如果回测提前结束，可能时间戳数量会多于净值点
        if len(valid_equity_points) > len(equity_timestamps):
             # print(f"Warning: More equity points ({len(valid_equity_points)}) than timestamps ({len(equity_timestamps)}). Trimming equity points.")
             valid_equity_points = valid_equity_points[:len(equity_timestamps)]
        elif len(valid_equity_points) < len(equity_timestamps):
             # print(f"Warning: Fewer equity points ({len(valid_equity_points)}) than timestamps ({len(equity_timestamps)}). Trimming timestamps.")
             equity_timestamps = equity_timestamps[:len(valid_equity_points)]


        if not valid_equity_points or len(valid_equity_points) != len(equity_timestamps):
            # print(f"Warning: Length mismatch or no valid equity points for returns calculation. EP: {len(valid_equity_points)}, TS: {len(equity_timestamps)}")
            return pd.Series(dtype=float)

        equity_series_for_returns = pd.Series(valid_equity_points, index=equity_timestamps)

        if equity_series_for_returns.empty or len(equity_series_for_returns) < 2: 
            return pd.Series(dtype=float)

        # 根据config.TIMEFRAME决定是否需要以及如何重采样到日级别
        resample_freq = kline_bar_for_resample
        
        # 检查K线时间周期是否已经是日级别或更粗
        tf_upper = config.TIMEFRAME.upper()
        is_daily_or_coarser = tf_upper.endswith('D') or tf_upper.endswith('W') or \
                              tf_upper.endswith('M') or tf_upper.endswith('Y')

        if kline_bar_for_resample is None or is_daily_or_coarser: # 如果不指定重采样或K线已是日级别
            resampled_equity = equity_series_for_returns
        else: # 对于分钟、小时级别，尝试重采样到每日的收盘净值
            try:
                resampled_equity = equity_series_for_returns.resample(resample_freq).last()
            except Exception as e_resample:
                # print(f"Warning: Resampling equity to {resample_freq} failed: {e_resample}. Using kline-frequency returns.")
                resampled_equity = equity_series_for_returns # 回退

        resampled_equity = resampled_equity.dropna() 
        if len(resampled_equity) < 2: 
            return pd.Series(dtype=float)
            
        periodic_returns = resampled_equity.pct_change().dropna()
        return periodic_returns


    def _calculate_summary_metrics(self) -> dict:
        """计算详细的回测绩效指标"""
        if not self.equity_curve or len(self.equity_curve) <= 1:
            return self._generate_empty_summary()

        final_capital = float(self.equity_curve[-1])
        total_pnl = float(final_capital - self.initial_capital)
        num_total_trades = len(self.trades)
        pnl_values = [float(trade['pnl']) for trade in self.trades] if self.trades else []
        
        winning_trades = sum(1 for pnl in pnl_values if pnl > 0)
        losing_trades = sum(1 for pnl in pnl_values if pnl < 0)
        win_rate = (winning_trades / num_total_trades) * 100.0 if num_total_trades > 0 else 0.0
        avg_pnl_per_trade = total_pnl / num_total_trades if num_total_trades > 0 else 0.0
        
        gross_profit = sum(pnl for pnl in pnl_values if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnl_values if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.cummax()
        drawdowns = running_max - equity_series
        
        # 计算百分比回撤 (相对于当时的峰值)
        # 确保 running_max 不为0，以避免除以0的错误
        # 同时，如果 running_max 为负 (虽然在从正初始资金开始的策略中不常见)，处理方式可能需要调整
        percentage_drawdowns = pd.Series(index=equity_series.index, dtype=float)
        for k_idx in range(len(equity_series)):
            if running_max[k_idx] > 0.000001: # 使用一个小的epsilon来避免接近0的情况
                percentage_drawdowns[k_idx] = drawdowns[k_idx] / running_max[k_idx]
            elif running_max[k_idx] <= 0 and equity_series[k_idx] < running_max[k_idx]:
                percentage_drawdowns[k_idx] = 1.0 # 完全亏损或更多
            else:
                percentage_drawdowns[k_idx] = 0.0 # 无回撤 (例如，从未盈利或净值恒定)

        max_dd = float(percentage_drawdowns.max()) if not percentage_drawdowns.empty else 0.0
        
        holding_periods = []
        for trade in self.trades:
            if isinstance(trade.get('entry_time'), pd.Timestamp) and isinstance(trade.get('exit_time'), pd.Timestamp):
                holding_periods.append(trade['exit_time'] - trade['entry_time'])
        avg_holding_period = sum(holding_periods, pd.Timedelta(0)) / len(holding_periods) if holding_periods else pd.Timedelta(0)

        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        
        periodic_returns_daily = self._calculate_periodic_returns(kline_bar_for_resample='D') # 标准使用日收益率
        if periodic_returns_daily.empty and len(self.equity_curve) > 1 : 
             periodic_returns_daily = self._calculate_periodic_returns(kline_bar_for_resample=None) # 回退到K线频率

        if not periodic_returns_daily.empty and len(periodic_returns_daily) >= 2: # 至少需要2个周期才能计算std
            mean_return = periodic_returns_daily.mean()
            std_return = periodic_returns_daily.std()
            
            if std_return != 0 and not np.isnan(std_return):
                # 假设无风险利率为0，计算非年化夏普
                sharpe_ratio = mean_return / std_return
                
                negative_returns = periodic_returns_daily[periodic_returns_daily < 0]
                if not negative_returns.empty:
                    downside_deviation = negative_returns.std()
                    if downside_deviation != 0 and not np.isnan(downside_deviation):
                        sortino_ratio = mean_return / downside_deviation
            
        summary = {
            "Initial Capital": self.initial_capital, 
            "Final Capital": final_capital,
            "Total PnL": total_pnl, 
            "Total Trades": num_total_trades,
            "Winning Trades": winning_trades, 
            "Losing Trades": losing_trades,
            "Win Rate": win_rate, 
            "Max Drawdown": max_dd,
            "Sharpe Ratio": float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0,
            "Sortino Ratio": float(sortino_ratio) if not np.isnan(sortino_ratio) else 0.0,
            "Avg PnL per Trade": avg_pnl_per_trade, 
            "Profit Factor": profit_factor,
            "Avg Holding Period": str(avg_holding_period).split('.')[0]
        }
        return summary

    def print_summary(self, summary_dict=None):
        if summary_dict is None:
            print("No summary data provided to print.")
            return

        print("\n--- Backtest Summary ---")
        for key, value in summary_dict.items():
            if isinstance(value, float) and key not in ["Initial Capital", "Final Capital", "Total PnL", "Avg PnL per Trade"]:
                 print(f"{key:<20}: {value:>.4f}")
            elif isinstance(value, float):
                 print(f"{key:<20}: {value:>.2f}")
            else:
                 print(f"{key:<20}: {value}")

if __name__ == '__main__':
    print("Testing Backtester independently...")
    from okx_connector import OKXConnector
    from data_handler import DataHandler
    from strategy import EvolvableStrategy 

    try:
        okx_conn_test = OKXConnector()
        data_hdl_test = DataHandler(okx_conn_test)
        
        test_df = data_hdl_test.fetch_klines_to_df(
            instId=config.SYMBOL, 
            bar=config.TIMEFRAME, 
            total_limit_needed=500,
            kline_type="history_market" 
        )

        if test_df.empty or len(test_df) < 50:
            print(f"Not enough data for backtester standalone test (fetched {len(test_df)}).")
        else:
            print(f"Fetched {len(test_df)} klines for backtester test.")
            
            sample_genes = {}
            # 使用 config.py 中定义的 GENE_SCHEMA 来生成随机但有效的基因
            for param_name, details in config.GENE_SCHEMA.items():
                if details['type'] == 'int':
                    sample_genes[param_name] = random.randint(details['range'][0], details['range'][1])
                elif details['type'] == 'float':
                    sample_genes[param_name] = random.uniform(details['range'][0], details['range'][1])
            
            # 手动确保一些逻辑约束，因为 EvolvableStrategy 的 _ensure_parameter_constraints 可能更复杂
            if 'short_ma_period' in sample_genes and 'long_ma_period' in sample_genes:
                if sample_genes['short_ma_period'] >= sample_genes['long_ma_period']:
                    sample_genes['long_ma_period'] = sample_genes['short_ma_period'] + \
                        random.randint(5, config.GENE_SCHEMA['long_ma_period']['range'][1] - sample_genes['short_ma_period'] -1 \
                                       if config.GENE_SCHEMA['long_ma_period']['range'][1] > sample_genes['short_ma_period'] + 5 else 5)
                    sample_genes['long_ma_period'] = min(sample_genes['long_ma_period'], config.GENE_SCHEMA['long_ma_period']['range'][1])
                    if sample_genes['short_ma_period'] >= sample_genes['long_ma_period']: # 再次确保
                         sample_genes['short_ma_period'] = max(config.GENE_SCHEMA['short_ma_period']['range'][0], sample_genes['long_ma_period']-5)


            if 'atr_take_profit_multiplier' in sample_genes and 'atr_stop_loss_multiplier' in sample_genes:
                 if sample_genes['atr_take_profit_multiplier'] <= sample_genes['atr_stop_loss_multiplier']:
                      sample_genes['atr_take_profit_multiplier'] = sample_genes['atr_stop_loss_multiplier'] + 0.5
                      sample_genes['atr_take_profit_multiplier'] = min(sample_genes['atr_take_profit_multiplier'], config.GENE_SCHEMA['atr_take_profit_multiplier']['range'][1])


            print(f"Using sample genes for EvolvableStrategy: {sample_genes}")
            test_strategy_instance = EvolvableStrategy(sample_genes)
            
            backtester_instance = Backtester(
                data_df=test_df, 
                strategy_instance=test_strategy_instance
            )
            summary_results, trades_log = backtester_instance.run_backtest()

            if summary_results:
                backtester_instance.print_summary(summary_results)
            if trades_log is not None and not trades_log.empty:
                print("\nTrades Log (first 5):")
                print(trades_log.head())
            else:
                print("\nNo trades were executed in this backtest run.")
                
    except ValueError as ve:
        print(f"Backtester Test Error (ValueError): {ve}")
    except Exception as e:
        print(f"Backtester Test Error (Exception): {e}")
        import traceback
        traceback.print_exc()
    print("\nBacktester standalone test finished.")