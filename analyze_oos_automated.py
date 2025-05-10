# trading_bot/analyze_oos_automated.py
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import logging
import os
from datetime import timedelta

# (从你的主项目中)导入指标计算器和配置
try:
    from indicators import Indicators
    import config # 需要config来获取基因参数对应的指标周期等
except ImportError:
    print("错误：无法导入 indicators.py 或 config.py。")
    exit()

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# 定义输出目录
OUTPUT_DIR = "oos_analysis_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    """加载OOS回测所需的数据。"""
    try:
        oos_kline_df = pd.read_csv("oos_kline_data.csv", index_col=0, parse_dates=True)
        oos_trades_df = pd.read_csv("oos_trades_log.csv", parse_dates=['entry_time', 'exit_time'])
        with open("best_genes.json", 'r') as f:
            best_genes = json.load(f)
        logger.info("数据加载成功。")
        return oos_kline_df, oos_trades_df, best_genes
    except FileNotFoundError as e:
        logger.critical(f"错误: 必需的数据文件未找到: {e}。请先运行 'main.py backtest_oos' 并确保文件已生成。")
        return None, None, None
    except Exception as e_load:
        logger.critical(f"加载数据时发生错误: {e_load}", exc_info=True)
        return None, None, None

def add_indicators_to_kline_data(kline_df, genes):
    """为K线数据添加策略所需的指标。"""
    calculator = Indicators()
    try:
        kline_with_indicators = calculator.add_sma(kline_df, period=genes['short_ma_period'])
        kline_with_indicators = calculator.add_sma(kline_with_indicators, period=genes['long_ma_period'])
        kline_with_indicators = calculator.add_atr(kline_with_indicators, period=genes['atr_period'])
        logger.info("指标已添加到K线数据。")
        return kline_with_indicators
    except KeyError as e_gene:
        logger.critical(f"基因文件中缺少必要的参数: {e_gene}。无法计算指标。")
        return None
    except Exception as e_indic:
        logger.critical(f"计算指标时发生错误: {e_indic}", exc_info=True)
        return None

def calculate_equity_curve_and_drawdown(initial_capital, trades_df, kline_df_index):
    """根据交易日志和初始资金计算权益曲线和回撤序列。"""
    if trades_df.empty:
        logger.info("没有交易，无法计算权益曲线。")
        # 创建一个平坦的权益曲线，对应于K线数据的时间范围
        flat_equity = pd.Series(initial_capital, index=kline_df_index)
        return flat_equity, pd.Series(0.0, index=kline_df_index), pd.Series(0.0, index=kline_df_index)

    # 确保交易按时间排序
    trades_df = trades_df.sort_values(by='exit_time')
    
    equity = initial_capital
    equity_over_time = {kline_df_index[0] - timedelta(seconds=1): initial_capital} # 交易前初始资金

    # 将已实现盈亏分配到退出时间点
    realized_pnl_at_exit = trades_df.groupby('exit_time')['pnl'].sum()

    current_pnl_sum = 0
    for timestamp in kline_df_index:
        if timestamp in realized_pnl_at_exit.index:
            current_pnl_sum += realized_pnl_at_exit[timestamp]
        equity_over_time[timestamp] = initial_capital + current_pnl_sum
        
    equity_series = pd.Series(equity_over_time).sort_index()
    # 重新索引以匹配完整的K线时间范围，并向前填充缺失值
    equity_series = equity_series.reindex(kline_df_index, method='ffill').fillna(initial_capital)


    running_max = equity_series.cummax()
    absolute_drawdown = running_max - equity_series
    percentage_drawdown = pd.Series(index=equity_series.index, dtype=float)
    for idx in equity_series.index:
        if running_max[idx] > 0.000001:
            percentage_drawdown[idx] = absolute_drawdown[idx] / running_max[idx]
        else:
            percentage_drawdown[idx] = 0.0
            
    logger.info("权益曲线和回撤计算完成。")
    return equity_series, absolute_drawdown, percentage_drawdown


def plot_trade_context_automated(kline_df_with_indicators: pd.DataFrame, 
                                 trade_info: pd.Series, 
                                 genes: dict,
                                 trade_idx: int,
                                 output_dir: str,
                                 window_before: int = 70, 
                                 window_after: int = 50):
    """自动绘制单笔交易图表并保存到文件。"""
    # (与之前 plot_trade_context 函数逻辑类似，但最后是保存文件而不是显示)
    entry_time = trade_info['entry_time']
    exit_time = trade_info['exit_time']
    entry_price = trade_info['entry_price']
    exit_price = trade_info['exit_price']

    short_ma = genes['short_ma_period']
    long_ma = genes['long_ma_period']
    atr_period = genes['atr_period']
    atr_sl_mult = genes['atr_stop_loss_multiplier']
    atr_tp_mult = genes['atr_take_profit_multiplier']

    sma_short_col_name = f'SMA_{short_ma}'
    sma_long_col_name = f'SMA_{long_ma}'
    atr_col_name = f'ATR_{atr_period}'

    required_indicator_cols = [sma_short_col_name, sma_long_col_name, atr_col_name]
    for col in required_indicator_cols:
        if col not in kline_df_with_indicators.columns:
            logger.error(f"错误: 指标列 '{col}' 不在K线数据中 (Trade {trade_idx})。")
            return

    try:
        entry_time_loc = kline_df_with_indicators.index.asof(entry_time)
        exit_time_loc = kline_df_with_indicators.index.asof(exit_time)
        if pd.isna(entry_time_loc) or pd.isna(exit_time_loc):
            logger.error(f"无法在K线数据中定位有效交易时间 (Trade {trade_idx}): Entry={entry_time}, Exit={exit_time}")
            return

        start_idx_loc_num = kline_df_with_indicators.index.get_loc(entry_time_loc)
        end_idx_loc_num = kline_df_with_indicators.index.get_loc(exit_time_loc)
        
        plot_start_idx = max(0, start_idx_loc_num - window_before)
        plot_end_idx = min(len(kline_df_with_indicators), end_idx_loc_num + window_after + 1)
        plot_df = kline_df_with_indicators.iloc[plot_start_idx:plot_end_idx].copy()
    except Exception as e_loc:
        logger.error(f"定位交易时间或切片时发生错误 (Trade {trade_idx}): {e_loc}")
        return

    if plot_df.empty:
        logger.warning(f"无法获取用于绘图的数据范围 (Trade {trade_idx})。")
        return

    atr_at_entry_for_plot = np.nan
    if entry_time_loc in plot_df.index and atr_col_name in plot_df.columns:
        atr_at_entry_for_plot = plot_df.loc[entry_time_loc, atr_col_name]
    if pd.isna(atr_at_entry_for_plot) or atr_at_entry_for_plot <=0: # 更稳健的回退
        valid_atr_slice = plot_df.loc[:entry_time_loc, atr_col_name].ffill()
        if not valid_atr_slice.empty and not pd.isna(valid_atr_slice.iloc[-1]) and valid_atr_slice.iloc[-1] > 0:
            atr_at_entry_for_plot = valid_atr_slice.iloc[-1]
        else:
            atr_at_entry_for_plot = plot_df[atr_col_name].dropna().mean()
            if pd.isna(atr_at_entry_for_plot) or atr_at_entry_for_plot <= 0:
                atr_at_entry_for_plot = 0.001 # 最后的保险，防止计算错误

    stop_loss_level = entry_price - (atr_at_entry_for_plot * atr_sl_mult)
    take_profit_level = entry_price + (atr_at_entry_for_plot * atr_tp_mult)
    plot_df['stop_loss'] = stop_loss_level
    plot_df['take_profit'] = take_profit_level

    addplots = [
        mpf.make_addplot(plot_df[sma_short_col_name], color='blue', width=0.7, panel=0),
        mpf.make_addplot(plot_df[sma_long_col_name], color='orange', width=0.7, panel=0),
        mpf.make_addplot(plot_df['stop_loss'], color='red', linestyle='dashed', width=0.7, panel=0),
        mpf.make_addplot(plot_df['take_profit'], color='green', linestyle='dashed', width=0.7, panel=0),
    ]
    if atr_col_name in plot_df.columns:
         addplots.append(mpf.make_addplot(plot_df[atr_col_name].fillna(method='bfill').fillna(method='ffill'), 
                                          panel=2, color='purple', ylabel=f'ATR({atr_period})'))

    filename = os.path.join(output_dir, f"trade_{trade_idx}_pnl_{trade_info['pnl']:.0f}_{trade_info['signal_exit']}.png")
    
    # 构建标题
    title = (f"Trade {trade_idx} (PnL: {trade_info['pnl']:.2f}) - {trade_info['signal_exit']}\n"
             f"Entry: {entry_time_loc.strftime('%y-%m-%d %H:%M')} @ {entry_price:.1f} | "
             f"Exit: {exit_time_loc.strftime('%y-%m-%d %H:%M')} @ {exit_price:.1f}\n"
             f"SL: {stop_loss_level:.1f}, TP: {take_profit_level:.1f} (ATR@Entry: {atr_at_entry_for_plot:.2f})")


    try:
        mpf.plot(plot_df, type='candle', style='yahoo',
                 title=title,
                 addplot=addplots,
                 volume=True, panel_ratios=(6,1,2),
                 figscale=1.5, figsize=(16,9), 
                 savefig=dict(fname=filename, dpi=100, pad_inches=0.25) # 保存文件
                )
        plt.close('all') # 关闭所有 matplotlib 图形，防止内存泄漏
        logger.info(f"已保存交易图表: {filename}")
    except Exception as e_plot:
        logger.error(f"绘图并保存交易 {trade_idx} 时发生错误: {e_plot}", exc_info=False)


def analyze_oos_performance(kline_df, trades_df, genes, initial_capital):
    """执行OOS性能的主要分析流程。"""
    if kline_df is None or trades_df is None or genes is None:
        logger.error("数据不完整，无法开始分析。")
        return

    kline_with_indicators = add_indicators_to_kline_data(kline_df, genes)
    if kline_with_indicators is None:
        return

    # 1. 整体统计
    logger.info("\n--- OOS 整体统计 ---")
    total_pnl = trades_df['pnl'].sum()
    num_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0.0
    profit_factor = trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) \
                    if abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) > 0 else float('inf') if trades_df[trades_df['pnl'] > 0]['pnl'].sum() > 0 else 0.0
    
    logger.info(f"总盈亏: {total_pnl:.2f}")
    logger.info(f"总交易数: {num_trades}")
    logger.info(f"胜率: {win_rate:.2f}%")
    logger.info(f"盈利因子: {profit_factor:.2f}")

    # 2. 权益曲线和回撤分析
    equity_series, abs_dd, pct_dd = calculate_equity_curve_and_drawdown(initial_capital, trades_df, kline_with_indicators.index)
    
    if not equity_series.empty:
        max_pct_dd = pct_dd.max()
        max_abs_dd = abs_dd.max()
        logger.info(f"最大百分比回撤: {max_pct_dd:.2%}")
        logger.info(f"最大绝对回撤: {max_abs_dd:.2f}")

        # 绘制权益曲线图并保存
        plt.figure(figsize=(12, 6))
        equity_series.plot(title="OOS Equity Curve", grid=True)
        plt.xlabel("Time")
        plt.ylabel("Equity")
        equity_fig_path = os.path.join(OUTPUT_DIR, "oos_equity_curve.png")
        plt.savefig(equity_fig_path)
        plt.close()
        logger.info(f"权益曲线图已保存到: {equity_fig_path}")

        # 找到最大回撤发生的点
        if not pct_dd.empty:
            max_dd_time = pct_dd.idxmax()
            logger.info(f"最大回撤发生在: {max_dd_time}")
            # 可以在这里触发对最大回撤期间的交易进行可视化

    # 3. 分析特定类型的交易
    # 3.1 亏损最大的 N 笔交易
    N_largest_losses = 5
    largest_losing_trades = trades_df[trades_df['pnl'] < 0].nsmallest(N_largest_losses, 'pnl')
    logger.info(f"\n--- {N_largest_losses} 笔最大亏损交易 ---")
    for idx, trade in largest_losing_trades.iterrows():
        logger.info(f"  Trade Index (in oos_trades_log.csv): {idx}, PnL: {trade['pnl']:.2f}, Exit: {trade['signal_exit']}")
        plot_trade_context_automated(kline_with_indicators, trade, genes, idx, OUTPUT_DIR)

    # 3.2 盈利最大的 N 笔交易 (可选，用于对比)
    N_largest_wins = 3
    largest_winning_trades = trades_df[trades_df['pnl'] > 0].nlargest(N_largest_wins, 'pnl')
    logger.info(f"\n--- {N_largest_wins} 笔最大盈利交易 ---")
    for idx, trade in largest_winning_trades.iterrows():
        logger.info(f"  Trade Index: {idx}, PnL: {trade['pnl']:.2f}, Exit: {trade['signal_exit']}")
        # plot_trade_context_automated(kline_with_indicators, trade, genes, idx, OUTPUT_DIR) # 可选绘制

    # 3.3 分析导致最大回撤的交易 (更复杂，需要找到回撤区间内的交易)
    if not pct_dd.empty and max_pct_dd > 0:
        peak_before_max_dd_time = equity_series.loc[:max_dd_time].idxmax() # 最大回撤前的峰值点
        logger.info(f"\n--- 分析最大回撤期间 ({peak_before_max_dd_time} to {max_dd_time}) 的交易 ---")
        trades_during_max_dd = trades_df[
            (trades_df['entry_time'] >= peak_before_max_dd_time) & (trades_df['exit_time'] <= max_dd_time) |
            (trades_df['entry_time'] <= peak_before_max_dd_time) & (trades_df['exit_time'] >= peak_before_max_dd_time) & (trades_df['exit_time'] <= max_dd_time) |
            (trades_df['entry_time'] >= peak_before_max_dd_time) & (trades_df['entry_time'] <= max_dd_time) & (trades_df['exit_time'] >= max_dd_time)
        ]
        if not trades_during_max_dd.empty:
            logger.info(f"在最大回撤期间发生了 {len(trades_during_max_dd)} 笔交易:")
            for idx, trade in trades_during_max_dd.iterrows():
                logger.info(f"  Trade Index: {idx}, PnL: {trade['pnl']:.2f}, Entry: {trade['entry_time']}, Exit: {trade['exit_time']}")
                # plot_trade_context_automated(kline_with_indicators, trade, genes, idx, OUTPUT_DIR) # 可选绘制
        else:
            logger.info("最大回撤期间没有完整发生的交易，可能是单边下跌或持仓浮亏导致。")
            # 可以绘制峰值点和谷值点附近的K线图
            plot_window_dd = kline_with_indicators[ (kline_with_indicators.index >= peak_before_max_dd_time - pd.Timedelta(days=3)) & \
                                                    (kline_with_indicators.index <= max_dd_time + pd.Timedelta(days=3)) ]
            if not plot_window_dd.empty:
                dd_filename = os.path.join(OUTPUT_DIR, f"max_drawdown_period_around_{max_dd_time.strftime('%Y%m%d')}.png")
                try:
                    ap0 = [ # 在图上标记峰值和谷值
                        mpf.make_addplot(equity_series.loc[plot_window_dd.index], panel=1, color='blue', ylabel='Equity'),
                        mpf.make_addplot(pd.Series(equity_series[peak_before_max_dd_time], index=[peak_before_max_dd_time]), type='scatter', marker='^', color='green', panel=1, markersize=100),
                        mpf.make_addplot(pd.Series(equity_series[max_dd_time], index=[max_dd_time]), type='scatter', marker='v', color='red', panel=1, markersize=100),
                    ]
                    mpf.plot(plot_window_dd, type='candle', style='yahoo', addplot=ap0,
                             title=f"Max Drawdown Period ({max_pct_dd:.2%})",
                             volume=True, panel_ratios=(3,1,1), figscale=1.2, figsize=(15,8),
                             savefig=dict(fname=dd_filename, dpi=100, pad_inches=0.25)
                            )
                    plt.close('all')
                    logger.info(f"最大回撤期间图表已保存: {dd_filename}")
                except Exception as e_dd_plot:
                     logger.error(f"绘制最大回撤图表时出错: {e_dd_plot}")


    logger.info("\n自动化分析完成。请查看输出目录 " + OUTPUT_DIR + " 中的图表。")


if __name__ == '__main__':
    oos_kline_data, oos_trades_data, best_genes_data = load_data()
    
    if oos_kline_data is not None and oos_trades_data is not None and best_genes_data is not None:
        # 从config获取初始资金，或者直接在这里定义
        initial_cap = config.INITIAL_CAPITAL 
        analyze_oos_performance(oos_kline_data, oos_trades_data, best_genes_data, initial_cap)
    else:
        logger.error("无法加载所有必需数据，分析中止。")