# trading_bot/main.py
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
import json
import os
import sys
import logging
import copy
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

import cProfile 
import pstats
import io

import config 
from okx_connector import OKXConnector
from data_handler import DataHandler
from strategy import EvolvableStrategy 
from backtester import Backtester
from genetic_optimizer_phase1 import GeneticOptimizerPhase1 

BEST_GENES_FILE = "best_genes.json"

logger = logging.getLogger('trading_bot')
logger.setLevel(logging.DEBUG) 
ch = logging.StreamHandler()
ch.setLevel(logging.INFO) 
logging.getLogger('trading_bot.genetic_optimizer_phase1').setLevel(logging.INFO)
logging.getLogger('trading_bot.backtester').setLevel(logging.INFO) 
logging.getLogger('trading_bot.strategy').setLevel(logging.INFO) 
logging.getLogger('trading_bot.data_handler').setLevel(logging.INFO) 
logging.getLogger('trading_bot.okx_connector').setLevel(logging.INFO) 
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
if not logger.hasHandlers(): logger.addHandler(ch)

def run_live_trading_loop(connector: OKXConnector, data_handler: DataHandler, best_genes_from_ga: Dict[str, Any]):
    logger.info(f"\n--- Starting Live Trading Loop for {config.SYMBOL} ({config.TIMEFRAME}) ---")
    if not best_genes_from_ga: logger.error("Error: No best genes for live trading."); return
    try:
        strategy = EvolvableStrategy(copy.deepcopy(best_genes_from_ga))
        logger.info(f"Using Strategy for live trading: {strategy.genes}")
    except Exception as e: logger.error(f"ERROR: Init EvolvableStrategy for live trading failed: {e}", exc_info=True); return
    if not config.IS_DEMO_TRADING: logger.warning("!!! REAL TRADING MODE !!!")
    else: logger.info("DEMO TRADING MODE for live loop.")
    logger.warning("\nLive trading loop is conceptual.")

def select_final_best_gene_from_candidates(
    candidates: List[Dict[str, Any]],
    primary_metric: str, 
    optimization_dataset_for_final_check: Optional[pd.DataFrame] = None
) -> Optional[Dict[str, Any]]:
    if not candidates:
        logger.warning("候选基因列表为空，无法选择最终基因。")
        return None

    def get_metric_value(candidate_summary, metric_name, default_value=-float('inf')):
        if not isinstance(candidate_summary, dict): return default_value
        val = candidate_summary.get(metric_name, default_value)
        if metric_name == 'Profit Factor' and val == float('inf'): return float('inf')
        if isinstance(val, (int, float)) and not pd.isna(val):
            if metric_name == 'Profit Factor' or not np.isinf(val): return val
        return default_value

    sorted_candidates = sorted(
        candidates,
        key=lambda c: get_metric_value(c.get('validation_summary'), primary_metric, -float('inf')),
        reverse=True 
    )
    logger.info(f"\n--- 从 {len(sorted_candidates)} 个通过验证的候选中选择最终基因 (主要指标: {primary_metric} on Validation Set) ---")
    if not sorted_candidates: return None

    logger.info(f"按主要指标 ({primary_metric}) 排序后的前几名候选者 (验证集表现):")
    for i, cand_data in enumerate(sorted_candidates[:min(5, len(sorted_candidates))]):
        genes_d = cand_data.get('genes', {}); val_summary_d = cand_data.get('validation_summary', {}); train_fit_d = cand_data.get('train_fitness', 'N/A')
        train_fit_str_d = f"{train_fit_d:.2f}" if isinstance(train_fit_d, float) else str(train_fit_d)
        logger.info(f"   cand {i+1}: Genes={ {k: (f'{v:.3f}' if isinstance(v,float) else v) for k,v in genes_d.items()} }, "+
                    f"Val-{primary_metric}={get_metric_value(val_summary_d, primary_metric):.3f}, " +
                    f"Val-Trades={get_metric_value(val_summary_d, 'Total Trades', 0):.0f}, " +
                    f"Val-MaxDD={get_metric_value(val_summary_d, 'Max Drawdown', 100.0):.2f}%, " +
                    f"Val-Sharpe={get_metric_value(val_summary_d, 'Sharpe Ratio', -1.0):.3f}, " +
                    f"TrainFit={train_fit_str_d}")

    min_trades_final_val = config.MIN_VALIDATION_TRADES_FOR_FINAL_SELECTION
    qualified_by_val_trades = [
        c for c in sorted_candidates
        if get_metric_value(c.get('validation_summary'), 'Total Trades', 0) >= min_trades_final_val
    ]
    logger.info(f"筛选后，{len(qualified_by_val_trades)} 个候选者在验证集交易次数 >= {min_trades_final_val}。")
    if not qualified_by_val_trades:
        logger.warning(f"没有候选者达到最终选择的验证集交易次数门槛。将从之前排序的候选中选择（如果有）。")
        return sorted_candidates[0]['genes'] if sorted_candidates else None

    final_candidates_after_dd_check = []
    if optimization_dataset_for_final_check is not None and not optimization_dataset_for_final_check.empty:
        logger.info("对合格候选者在整个优化数据集上进行最终回撤检查...")
        # *** MODIFIED THRESHOLD FOR FULL OPTIMIZATION DATASET DD CHECK ***
        MAX_DD_ABSOLUTE_LIMIT_ON_FULL_OPT = config.FITNESS_WEIGHTS.get('train_acceptable_max_drawdown', 0.25) 
        logger.info(f"  完整优化集最大允许回撤 (基于train_acceptable_max_drawdown): {MAX_DD_ABSOLUTE_LIMIT_ON_FULL_OPT:.2%}")

        for cand_data in qualified_by_val_trades:
            genes = cand_data['genes']
            logger.debug(f"  检查基因 { {k: (f'{v:.3f}' if isinstance(v,float) else v) for k,v in genes.items()} } 在完整优化集上的回撤...")
            try:
                temp_strat_full = EvolvableStrategy(copy.deepcopy(genes))
                backtester_full = Backtester(
                    data_df=optimization_dataset_for_final_check.copy(),
                    strategy_instance=temp_strat_full, initial_capital=config.INITIAL_CAPITAL, fee_rate=config.FEE_RATE,
                    risk_per_trade_percent=config.RISK_PER_TRADE_PERCENT, min_position_size=config.MIN_POSITION_SIZE
                )
                summary_full_opt, _ = backtester_full.run_backtest()
                if summary_full_opt:
                    max_dd_full_opt_pct = get_metric_value(summary_full_opt, 'Max Drawdown', 100.0)
                    max_dd_full_opt_decimal = max_dd_full_opt_pct / 100.0
                    if max_dd_full_opt_decimal <= MAX_DD_ABSOLUTE_LIMIT_ON_FULL_OPT:
                        final_candidates_after_dd_check.append({**cand_data, "full_opt_summary": summary_full_opt}) 
                        logger.debug(f"    通过回撤检查: 完整优化集 MaxDD={max_dd_full_opt_decimal:.2%}")
                    else:
                        logger.info(f"  候选基因 { {k: (f'{v:.3f}' if isinstance(v,float) else v) for k,v in genes.items()} } 未通过完整优化集回撤检查 (MaxDD={max_dd_full_opt_decimal:.2%} > {MAX_DD_ABSOLUTE_LIMIT_ON_FULL_OPT:.2%}).")
                else: logger.warning(f"  无法获取基因 {genes} 在完整优化集上的回测摘要。")
            except Exception as e_full_bt: logger.error(f"  对基因 {genes} 进行完整优化集回测时出错: {e_full_bt}")
        
        if not final_candidates_after_dd_check:
            logger.warning("在完整优化集回撤检查后，没有候选者符合条件。将从之前的合格候选中选择。")
            final_candidates_after_dd_check = [{**c, "full_opt_summary": c.get('validation_summary')} for c in qualified_by_val_trades]
    else:
        logger.warning("未提供完整优化数据集进行最终回撤检查，将跳过。")
        final_candidates_after_dd_check = [{**c, "full_opt_summary": c.get('validation_summary')} for c in qualified_by_val_trades]

    if not final_candidates_after_dd_check:
        logger.error("所有筛选后，没有候选基因。回退到验证集交易次数筛选后的最佳（按主指标）。")
        return qualified_by_val_trades[0]['genes'] if qualified_by_val_trades else (sorted_candidates[0]['genes'] if sorted_candidates else None)

    logger.info(f"对 {len(final_candidates_after_dd_check)} 个最终候选者应用多级次要排序 (基于完整优化集表现)...")
    final_candidates_after_dd_check.sort(key=lambda c: get_metric_value(c.get('full_opt_summary'), 'Max Drawdown', float('inf')), reverse=False)
    final_candidates_after_dd_check.sort(key=lambda c: get_metric_value(c.get('full_opt_summary'), primary_metric, -float('inf')), reverse=True) 
    final_candidates_after_dd_check.sort(key=lambda c: get_metric_value(c.get('full_opt_summary'), 'Sharpe Ratio', -float('inf')), reverse=True)
    final_candidates_after_dd_check.sort(key=lambda c: get_metric_value(c.get('full_opt_summary'), 'Total PnL Pct', -float('inf')), reverse=True)
    final_candidates_after_dd_check.sort(key=lambda c: get_metric_value(c.get('validation_summary'), 'Total Trades', 0), reverse=True)

    best_candidate_overall = final_candidates_after_dd_check[0]
    logger.info(f"通过多级次要标准细化选择后的最佳候选者基因: {best_candidate_overall['genes']}")
    val_summary_best = best_candidate_overall.get('validation_summary', {}); full_opt_summary_best = best_candidate_overall.get('full_opt_summary', {})
    logger.info(f"  其验证集表现: {primary_metric}={get_metric_value(val_summary_best, primary_metric):.3f}, Trades={get_metric_value(val_summary_best, 'Total Trades', 0):.0f}, MaxDD={get_metric_value(val_summary_best, 'Max Drawdown', 100.0):.2f}%, Sharpe={get_metric_value(val_summary_best, 'Sharpe Ratio', -1.0):.3f}")
    logger.info(f"  其完整优化集表现: {primary_metric}={get_metric_value(full_opt_summary_best, primary_metric):.3f}, Trades={get_metric_value(full_opt_summary_best, 'Total Trades', 0):.0f}, MaxDD={get_metric_value(full_opt_summary_best, 'Max Drawdown', 100.0):.2f}%, Sharpe={get_metric_value(full_opt_summary_best, 'Sharpe Ratio', -1.0):.3f}")
                
    return best_candidate_overall['genes']

if __name__ == '__main__':
    DO_PROFILING = False 
    profiler = None
    if DO_PROFILING: profiler = cProfile.Profile(); profiler.enable()
    try:
        logger.info("Trading Bot Initializing...")
        okx_conn = None; data_hdl = None
        try: okx_conn = OKXConnector(); data_hdl = DataHandler(okx_conn)
        except Exception as e_init: logger.critical(f"Init Error: {e_init}", exc_info=True); sys.exit(1)

        DEFAULT_MODE = "optimize"
        MODE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ["optimize", "backtest_best", "backtest_oos", "live_best"] else DEFAULT_MODE
        logger.info(f"\n--- Running in Mode: {MODE} ---")

        if MODE == "optimize":
            logger.info(f"\n--- Running Genetic Optimization Mode (with new fitness & selection) ---")
            total_optimization_df = pd.DataFrame()
            try:
                total_optimization_df = data_hdl.fetch_klines_to_df(instId=config.SYMBOL, bar=config.TIMEFRAME, total_limit_needed=config.KLINE_LIMIT_FOR_OPTIMIZATION, kline_type="history_market")
            except Exception as e_fetch_main: logger.critical(f"CRITICAL: Failed to fetch data: {e_fetch_main}", exc_info=True); sys.exit(1)
            
            max_period = 0 
            if hasattr(config, 'GENE_SCHEMA') and isinstance(config.GENE_SCHEMA, dict):
                for _, detail in config.GENE_SCHEMA.items():
                    if isinstance(detail, dict) and detail.get('type') == 'int' and 'range' in detail and isinstance(detail['range'], (list, tuple)) and len(detail['range']) == 2:
                        max_period = max(max_period, detail['range'][1])
            min_data_needed = max_period + 100 
            if total_optimization_df.empty or len(total_optimization_df) < min_data_needed:
                logger.critical(f"Not enough data ({len(total_optimization_df)}) for GA. Need ~{min_data_needed}. Exiting."); sys.exit(1)
            logger.info(f"Fetched {len(total_optimization_df)} klines for GA.")

            passed_validation_candidates_list: List[Dict[str, Any]] = []
            validation_metric_name_used = ""
            best_genes_final_selection_raw: Optional[Dict[str, Any]] = None
            try:
                optimizer = GeneticOptimizerPhase1(total_optimization_df=total_optimization_df) 
                passed_validation_candidates_list, validation_metric_name_used = optimizer.run_evolution()
            except Exception as e_ga_run: logger.critical(f"GA Run Error: {e_ga_run}", exc_info=True); sys.exit(1)

            if passed_validation_candidates_list:
                logger.info(f"GA finished. Found {len(passed_validation_candidates_list)} candidates passing initial validation.")
                best_genes_final_selection_raw = select_final_best_gene_from_candidates( 
                    passed_validation_candidates_list,
                    validation_metric_name_used, 
                    optimization_dataset_for_final_check=total_optimization_df 
                )
            else: logger.warning("GA did not yield any candidates passing all initial validation criteria.")
            
            if best_genes_final_selection_raw: 
                constrained_genes_to_save: Optional[Dict[str, Any]] = None
                final_strat_instance: Optional[EvolvableStrategy] = None
                try:
                    temp_strat = EvolvableStrategy(copy.deepcopy(best_genes_final_selection_raw))
                    constrained_genes_to_save = temp_strat.genes
                    final_strat_instance = temp_strat
                    logger.info(f"应用约束后的最终最佳基因: {constrained_genes_to_save}")
                    with open(BEST_GENES_FILE, 'w') as f: json.dump(constrained_genes_to_save, f, indent=4)
                    logger.info(f"Final CONSTRAINED best genes saved to {BEST_GENES_FILE}")
                except Exception as e_save:
                    logger.error(f"Error saving/constraining: {e_save}", exc_info=True)
                    constrained_genes_to_save = copy.deepcopy(best_genes_final_selection_raw)
                    try: 
                        with open(BEST_GENES_FILE, 'w') as f_raw: json.dump(constrained_genes_to_save, f_raw, indent=4)
                        final_strat_instance = EvolvableStrategy(copy.deepcopy(constrained_genes_to_save))
                    except: final_strat_instance = None

                if final_strat_instance and constrained_genes_to_save:
                    logger.info("\n--- Final Backtest (on ENTIRE Optimization Dataset) ---")
                    final_bt = Backtester(data_df=total_optimization_df.copy(), strategy_instance=final_strat_instance,
                                          initial_capital=config.INITIAL_CAPITAL, fee_rate=config.FEE_RATE,
                                          risk_per_trade_percent=config.RISK_PER_TRADE_PERCENT, min_position_size=config.MIN_POSITION_SIZE)
                    summary_f, _ = final_bt.run_backtest()
                    if summary_f: final_bt.print_summary(summary_f)
            else: logger.warning("GA did not select a final best gene set.")
        
        elif MODE == "backtest_best":
            logger.info(f"\n--- Running Backtest with Best Genes from {BEST_GENES_FILE} (on ENTIRE Optimization Dataset) ---")
            if not os.path.exists(BEST_GENES_FILE): logger.critical(f"{BEST_GENES_FILE} not found."); sys.exit(1)
            best_genes = {}; dataset_for_backtest_best = pd.DataFrame()
            try:
                with open(BEST_GENES_FILE, 'r') as f: best_genes = json.load(f)
                dataset_for_backtest_best = data_hdl.fetch_klines_to_df(instId=config.SYMBOL, bar=config.TIMEFRAME,total_limit_needed=config.KLINE_LIMIT_FOR_OPTIMIZATION, kline_type="history_market")
                strategy_instance = EvolvableStrategy(copy.deepcopy(best_genes))
                backtester = Backtester(data_df=dataset_for_backtest_best.copy(), strategy_instance=strategy_instance, initial_capital=config.INITIAL_CAPITAL, fee_rate=config.FEE_RATE, risk_per_trade_percent=config.RISK_PER_TRADE_PERCENT, min_position_size=config.MIN_POSITION_SIZE)
                summary, _ = backtester.run_backtest()
                if summary: backtester.print_summary(summary)
            except Exception as e_bt_best: logger.error(f"Error in 'backtest_best': {e_bt_best}", exc_info=True)

        elif MODE == "backtest_oos":
            logger.info(f"\n--- Running OUT-OF-SAMPLE Backtest with Best Genes from {BEST_GENES_FILE} ---")
            if not os.path.exists(BEST_GENES_FILE): logger.critical(f"{BEST_GENES_FILE} not found."); sys.exit(1)
            best_genes_oos = {}; all_historical_data = pd.DataFrame()
            try:
                with open(BEST_GENES_FILE, 'r') as f: best_genes_oos = json.load(f)
                total_data_for_oos_setup = config.KLINE_LIMIT_FOR_OPTIMIZATION + config.KLINE_LIMIT_FOR_OOS
                all_historical_data = data_hdl.fetch_klines_to_df(instId=config.SYMBOL, bar=config.TIMEFRAME, total_limit_needed=total_data_for_oos_setup, kline_type="history_market")
                oos_data_df = all_historical_data.iloc[:config.KLINE_LIMIT_FOR_OOS]
                # Add min data checks for OOS data
                oos_strategy_instance = EvolvableStrategy(copy.deepcopy(best_genes_oos))
                oos_backtester = Backtester(data_df=oos_data_df.copy(), strategy_instance=oos_strategy_instance, initial_capital=config.INITIAL_CAPITAL, fee_rate=config.FEE_RATE, risk_per_trade_percent=config.RISK_PER_TRADE_PERCENT, min_position_size=config.MIN_POSITION_SIZE)
                oos_summary, oos_trades_df = oos_backtester.run_backtest() 
                if oos_summary: oos_backtester.print_summary(oos_summary)
                if oos_trades_df is not None and not oos_trades_df.empty: 
                    try:
                        oos_data_df.to_csv("oos_kline_data.csv"); oos_trades_df.to_csv("oos_trades_log.csv", index=False)
                        logger.info("OOS K-line data and trades log saved.")
                    except Exception as e_save_oos: logger.error(f"Error saving OOS results: {e_save_oos}")
            except Exception as e_bt_oos: logger.error(f"Error during 'backtest_oos': {e_bt_oos}", exc_info=True)

        elif MODE == "live_best":
            logger.info(f"\n--- Running Live Trading with Best Genes from {BEST_GENES_FILE} (Conceptual) ---")
            if not os.path.exists(BEST_GENES_FILE): logger.critical(f"{BEST_GENES_FILE} not found."); sys.exit(1)
            best_genes_from_file_live = {}
            try:
                with open(BEST_GENES_FILE, 'r') as f: best_genes_from_file_live = json.load(f)
                live_strategy_instance = EvolvableStrategy(copy.deepcopy(best_genes_from_file_live))
                run_live_trading_loop(okx_conn, data_hdl, live_strategy_instance.genes) 
            except Exception as e_live_setup: logger.error(f"Error setting up 'live_best': {e_live_setup}", exc_info=True)
        else: logger.error(f"Unknown mode: {MODE}.")
        logger.info("\nTrading Bot Script Finished.")
    finally:
        if DO_PROFILING and profiler is not None: 
            profiler.disable(); s = io.StringIO(); sortby_key = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(profiler, stream=s).sort_stats(sortby_key); num_stats_to_print = 40
            ps.print_stats(num_stats_to_print); profiling_results_text = s.getvalue()
            print(f"\n--- Profiling Results (Top {num_stats_to_print} by {sortby_key.name}) ---\n{profiling_results_text}")
            profile_output_file = "main_optimize_profile_stats.prof"; profile_summary_file = "main_optimize_profile_summary.txt"
            try:
                profiler.dump_stats(profile_output_file); logger.info(f"Full profiling stats saved to {profile_output_file}")
                with open(profile_summary_file, "w") as f_prof_sum: f_prof_sum.write(profiling_results_text)
                logger.info(f"Profiling summary text saved to {profile_summary_file}")
            except Exception as e_prof_save: logger.error(f"Error saving profiling data: {e_prof_save}")
