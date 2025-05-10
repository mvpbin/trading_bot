# trading_bot/analyze_oos_results.py
import pandas as pd
import json
import matplotlib.pyplot as plt
import mplfinance as mpf # 用于绘制K线图，如果未安装，请执行 pip install mplfinance
import logging

# (从你的主项目中)导入指标计算器和配置 (确保路径正确)
# 假设 analyze_oos_results.py 与 indicators.py 和 config.py 在同一目录下或Python路径可达
try:
    from indicators import Indicators
    import config # 需要config来获取基因参数对应的指标周期等
except ImportError:
    print("错误：无法导入 indicators.py 或 config.py。请确保它们与此脚本在同一目录或Python路径可达。")
    print("如果此脚本在 trading_bot 文件夹的子目录，你可能需要调整导入路径，例如：")
    print("from ..indicators import Indicators")
    print("import ..config as config")
    exit()

# 配置日志 (可选，但有助于调试)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def plot_trade_context(kline_df_with_indicators: pd.DataFrame, 
                       trade_info: pd.Series, 
                       genes: dict,
                       window_before: int = 50, 
                       window_after: int = 30):
    """
    绘制单笔交易的上下文K线图及相关指标。
    kline_df_with_indicators: 包含指标的K线DataFrame
    trade_info: 一行交易记录 (来自 trades_df)
    genes: 包含策略参数的基因字典
    window_before, window_after: 交易发生前后显示的K线数量
    """
    entry_time = trade_info['entry_time']
    exit_time = trade_info['exit_time']
    entry_price = trade_info['entry_price']
    exit_price = trade_info['exit_price']

    # 从基因中获取参数
    short_ma = genes['short_ma_period']
    long_ma = genes['long_ma_period']
    atr_period = genes['atr_period']
    atr_sl_mult = genes['atr_stop_loss_multiplier']
    atr_tp_mult = genes['atr_take_profit_multiplier']

    sma_short_col_name = f'SMA_{short_ma}'
    sma_long_col_name = f'SMA_{long_ma}'
    atr_col_name = f'ATR_{atr_period}'

    # 确保指标列存在
    required_indicator_cols = [sma_short_col_name, sma_long_col_name, atr_col_name]
    for col in required_indicator_cols:
        if col not in kline_df_with_indicators.columns:
            logger.error(f"错误: 指标列 '{col}' 不在提供的K线数据中。请确保已正确计算指标。")
            return

    # 找到交易期间的数据范围
    try:
        # 确保entry_time和exit_time在索引中
        if entry_time not in kline_df_with_indicators.index:
            entry_time = kline_df_with_indicators.index.asof(entry_time) # 尝试获取最近的时间戳
        if exit_time not in kline_df_with_indicators.index:
            exit_time = kline_df_with_indicators.index.asof(exit_time)

        start_idx_loc = kline_df_with_indicators.index.get_loc(entry_time)
        end_idx_loc = kline_df_with_indicators.index.get_loc(exit_time)

        plot_start_idx = max(0, start_idx_loc - window_before)
        plot_end_idx = min(len(kline_df_with_indicators), end_idx_loc + window_after + 1) # +1 因为iloc尾部不包含
        
        plot_df = kline_df_with_indicators.iloc[plot_start_idx:plot_end_idx].copy()

    except KeyError as e:
        logger.error(f"无法在K线数据中定位交易时间: {e}. Entry={entry_time}, Exit={exit_time}")
        return
    except Exception as e_loc:
        logger.error(f"定位交易时间时发生其他错误: {e_loc}")
        return

    if plot_df.empty:
        logger.warning("无法获取用于绘图的数据范围。")
        return

    # 获取入场时的ATR值以计算固定的止损止盈线 (对于该笔交易)
    # 注意：理想情况下，这个 atr_at_entry 应该由策略在入场时记录并保存在交易日志中
    # 这里我们从K线数据中回溯获取，可能与策略内部的精确值有微小差异
    if entry_time in plot_df.index and atr_col_name in plot_df.columns:
        atr_at_entry_for_plot = plot_df.loc[entry_time, atr_col_name]
        if pd.isna(atr_at_entry_for_plot): # 如果入场时ATR是NaN，尝试前一个有效值
            atr_series_at_entry = plot_df.loc[:entry_time, atr_col_name].ffill()
            if not atr_series_at_entry.empty:
                atr_at_entry_for_plot = atr_series_at_entry.iloc[-1]
            else: # 实在找不到，给个默认值或警告
                logger.warning(f"无法在入场时 {entry_time} 获取有效的 ATR 值，止损止盈线可能不准确。")
                atr_at_entry_for_plot = plot_df[atr_col_name].mean() # 使用平均值作为粗略估计
                if pd.isna(atr_at_entry_for_plot) : atr_at_entry_for_plot = 0.001 # 最后的保险
    else:
        logger.warning(f"无法在入场时 {entry_time} 或在数据中找到 {atr_col_name}，止损止盈线可能不准确。")
        atr_at_entry_for_plot = (plot_df['high'] - plot_df['low']).mean() if not plot_df.empty else 0.001 # 极简回退

    if pd.isna(atr_at_entry_for_plot) or atr_at_entry_for_plot <= 0:
        logger.warning(f"入场时ATR值无效 ({atr_at_entry_for_plot})，无法准确绘制止损止盈。")
        # 可以选择不绘制止损止盈，或使用固定值
        stop_loss_level = entry_price * (1-0.05) # 示例回退
        take_profit_level = entry_price * (1+0.10) # 示例回退
    else:
        stop_loss_level = entry_price - (atr_at_entry_for_plot * atr_sl_mult)
        take_profit_level = entry_price + (atr_at_entry_for_plot * atr_tp_mult)
    
    plot_df['stop_loss'] = stop_loss_level
    plot_df['take_profit'] = take_profit_level

    # 准备附加图表 (addplot)
    addplots = [
        mpf.make_addplot(plot_df[sma_short_col_name], color='blue', width=0.7, panel=0), # panel 0 是主K线图
        mpf.make_addplot(plot_df[sma_long_col_name], color='orange', width=0.7, panel=0),
        mpf.make_addplot(plot_df['stop_loss'], color='red', linestyle='dashed', width=0.7, panel=0),
        mpf.make_addplot(plot_df['take_profit'], color='green', linestyle='dashed', width=0.7, panel=0),
    ]
    if atr_col_name in plot_df.columns: # 如果ATR数据存在，则作为副图绘制
         addplots.append(mpf.make_addplot(plot_df[atr_col_name].fillna(method='bfill').fillna(method='ffill'), 
                                          panel=2, color='purple', ylabel=f'ATR({atr_period})')) # panel 2 (0是主图, 1是成交量)

    # 绘制K线图
    try:
        fig, axes = mpf.plot(plot_df, type='candle', style='yahoo',
                             title=f"Trade Analysis (PnL: {trade_info['pnl']:.2f}) {trade_info['signal_exit']}\n"
                                   f"Entry: {entry_time.strftime('%Y-%m-%d %H:%M')} @ {entry_price:.2f} | "
                                   f"Exit: {exit_time.strftime('%Y-%m-%d %H:%M')} @ {exit_price:.2f}",
                             addplot=addplots,
                             volume=True, panel_ratios=(6,1,2), # 主图(6份), 成交量(1份), ATR副图(2份)
                             figscale=1.8, figsize=(15,8), # 调整图形大小
                             returnfig=True)

        # 在主图 (axes[0]) 上标记开仓和平仓点
        ax_main = axes[0]
        # mplfinance 使用 matplotlib 的日期格式作为x轴，可以直接用datetime对象
        
        ax_main.plot(entry_time, entry_price, '^', color='cyan', markersize=12, markeredgecolor='black', label='Entry Point')
        ax_main.plot(exit_time, exit_price, 'v' if trade_info['pnl'] > 0 else 'v', 
                     color='magenta' if trade_info['pnl'] > 0 else 'gold', 
                     markersize=12, markeredgecolor='black', label='Exit Point')
        
        ax_main.legend()
        mpf.show() # 使用mpf.show()以便在某些环境下正确显示
    except Exception as e_plot:
        logger.error(f"绘图时发生错误: {e_plot}", exc_info=True)


def main_analysis():
    logger.info("开始OOS结果分析...")

    # 1. 加载数据
    try:
        oos_kline_df = pd.read_csv("oos_kline_data.csv", index_col=0, parse_dates=True)
        oos_trades_df = pd.read_csv("oos_trades_log.csv", parse_dates=['entry_time', 'exit_time'])
        with open("best_genes.json", 'r') as f:
            best_genes = json.load(f)
    except FileNotFoundError as e:
        logger.critical(f"错误: 必需的数据文件未找到: {e}. 请先运行 'main.py backtest_oos'。")
        return
    except Exception as e_load:
        logger.critical(f"加载数据时发生错误: {e_load}", exc_info=True)
        return

    logger.info(f"加载的OOS K线数据: {oos_kline_df.shape[0]} 条")
    logger.info(f"加载的OOS交易日志: {oos_trades_df.shape[0]} 条")
    logger.info(f"加载的最佳基因: {best_genes}")

    # 2. 为K线数据添加指标 (使用与策略相同的逻辑)
    calculator = Indicators()
    try:
        oos_kline_with_indicators = calculator.add_sma(oos_kline_df, period=best_genes['short_ma_period'])
        oos_kline_with_indicators = calculator.add_sma(oos_kline_with_indicators, period=best_genes['long_ma_period'])
        oos_kline_with_indicators = calculator.add_atr(oos_kline_with_indicators, period=best_genes['atr_period'])
    except KeyError as e_gene:
        logger.critical(f"基因文件中缺少必要的参数: {e_gene}. 无法计算指标。")
        return
    except Exception as e_indic:
        logger.critical(f"计算指标时发生错误: {e_indic}", exc_info=True)
        return

    # 3. 分析交易
    logger.info("\n--- OOS交易统计分析 ---")
    if oos_trades_df.empty:
        logger.info("OOS期间没有交易。")
    else:
        total_pnl = oos_trades_df['pnl'].sum()
        num_trades = len(oos_trades_df)
        winning_trades = len(oos_trades_df[oos_trades_df['pnl'] > 0])
        losing_trades = len(oos_trades_df[oos_trades_df['pnl'] < 0])
        win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0
        
        logger.info(f"总盈亏 (OOS): {total_pnl:.2f}")
        logger.info(f"总交易数 (OOS): {num_trades}")
        logger.info(f"盈利交易数: {winning_trades}, 亏损交易数: {losing_trades}")
        logger.info(f"胜率 (OOS): {win_rate:.2f}%")

        logger.info("\n亏损交易详情:")
        losing_trades_details = oos_trades_df[oos_trades_df['pnl'] < 0].sort_values(by='pnl', ascending=True)
        for idx, trade in losing_trades_details.iterrows():
            logger.info(f"  交易 {idx}: PnL={trade['pnl']:.2f}, 入场={trade['entry_time']}, 出场={trade['exit_time']}, 出场信号={trade['signal_exit']}")

        # 4. 可视化每一笔交易 (或选择性可视化，例如只看亏损的)
        if not oos_trades_df.empty:
            logger.info("\n--- 开始可视化OOS交易 (按回车键查看下一笔) ---")
            for i, trade_row in oos_trades_df.iterrows():
                logger.info(f"\n正在分析交易 {i+1}/{len(oos_trades_df)}: PnL = {trade_row['pnl']:.2f}")
                plot_trade_context(
                    kline_df_with_indicators=oos_kline_with_indicators,
                    trade_info=trade_row,
                    genes=best_genes,
                    window_before=60, # 交易前显示的K线数
                    window_after=40   # 交易后显示的K线数
                )
                # input("按回车键继续分析下一笔交易 (或Ctrl+C退出)... ") # 如果图形窗口阻塞，这个可能不好用
                # mplfinance.show() 会阻塞，直到图形关闭。

    logger.info("\nOOS结果分析完成。")

if __name__ == '__main__':
    # 确保 matplotlib 使用一个非GUI的后端，如果是在没有图形界面的环境运行（例如服务器）
    # 但对于本地分析，通常不需要这个。如果 `mpf.show()` 出问题，可以尝试。
    # import matplotlib
    # matplotlib.use('Agg') # 例如，保存到文件而不是显示
    main_analysis()