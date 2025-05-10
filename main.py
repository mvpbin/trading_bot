# trading_bot/main.py
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
import json
import os
import sys
import logging
from typing import Optional, List, Dict, Any, Tuple # 引入更多类型提示

import config
from okx_connector import OKXConnector
from data_handler import DataHandler
from strategy import EvolvableStrategy
from backtester import Backtester
from genetic_optimizer_phase1 import GeneticOptimizerPhase1

BEST_GENES_FILE = "best_genes.json"

logger = logging.getLogger('trading_bot')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)

def run_live_trading_loop(connector: OKXConnector, data_handler: DataHandler, best_genes_from_ga: Dict[str, Any]):
    # (与上一版本相同)
    logger.info(f"\n--- Starting Live Trading Loop for {config.SYMBOL} ({config.TIMEFRAME}) ---")
    if not best_genes_from_ga: 
        logger.error("Error: No best genes provided for live trading. Exiting.")
        return
    try: 
        strategy = EvolvableStrategy(best_genes_from_ga)
        logger.info(f"Using Strategy with Genes for live trading: {strategy.genes}")
    except KeyError as ke:
        logger.error(f"ERROR: KeyError during EvolvableStrategy initialization for live trading with genes {best_genes_from_ga}: {ke}. ")
        return
    except Exception as e: 
        logger.error(f"ERROR: Init EvolvableStrategy for live trading failed: {e}", exc_info=True)
        return
    if not config.IS_DEMO_TRADING: logger.warning("!!! WARNING: REAL TRADING MODE !!!")
    else: logger.info("Running in DEMO TRADING MODE for live loop.")
    logger.warning("\nWARNING: Live trading loop is conceptual.")
    pass

def select_final_best_gene_from_candidates(
    candidates: List[Dict[str, Any]], 
    primary_metric: str, 
    secondary_metrics_desc: Optional[List[Tuple[str, bool]]] = None
) -> Optional[Dict[str, Any]]:
    """
    从通过验证的候选基因列表中选择最终的最佳基因。
    candidates: 候选列表，每个元素是一个包含 'genes' 和 'validation_summary' 的字典。
    primary_metric: 用于主要排序的验证集指标名称 (越大越好)。
    secondary_metrics_desc: 可选的次要排序标准列表，每个元素是 (metric_name, ascending_bool)。
                           例如 [('Max Drawdown', True)] 表示回撤越小越好。
    """
    if not candidates:
        logger.warning("候选基因列表为空，无法选择最终基因。")
        return None

    # 1. 按主要指标排序 (越大越好)
    #    确保指标存在且为数值，对于不存在或非数值的给予最差排序
    def get_primary_metric_value(candidate):
        val = candidate.get('validation_summary', {}).get(primary_metric, -float('inf'))
        return val if isinstance(val, (int, float)) and not pd.isna(val) else -float('inf')

    sorted_candidates = sorted(candidates, key=get_primary_metric_value, reverse=True)
    
    logger.info(f"\n--- 从 {len(sorted_candidates)} 个通过验证的候选中选择最终基因 (主要指标: {primary_metric}) ---")
    # 打印前几个候选者的主要指标
    for i, cand in enumerate(sorted_candidates[:min(5, len(sorted_candidates))]): # 最多打印前5个
        logger.info(f"  候选 {i+1}: 基因={cand.get('genes')}, {primary_metric}="
                    f"{cand.get('validation_summary', {}).get(primary_metric, 'N/A'):.4f}, "
                    f"训练适应度={cand.get('train_fitness', 'N/A'):.2f}")


    if not secondary_metrics_desc: # 如果没有次要标准，直接返回主要标准最优的
        logger.info(f"选择主要指标 '{primary_metric}' 最高的候选者。")
        return sorted_candidates[0]['genes']

    # 2. 应用次要排序标准 (如果主要指标值相近或需要进一步筛选)
    # 简单起见，我们这里只取主要指标最优的那个。
    # 更复杂的逻辑可以比较主要指标相近的一批候选者，再用次要指标排序。
    # 例如，取主要指标排名前N%的，再用次要指标细分。
    # 目前我们先简化，直接用主要指标筛选。
    
    best_candidate = sorted_candidates[0]
    
    # 如果需要更复杂的次要排序：
    # top_n_primary = [c for c in sorted_candidates 
    #                  if abs(get_primary_metric_value(c) - get_primary_metric_value(best_candidate)) < 0.001] # 主要指标相近
    # if len(top_n_primary) > 1 and secondary_metrics_desc:
    #     logger.info(f"主要指标相近，应用次要标准: {secondary_metrics_desc}")
    #     for metric_name, ascending in secondary_metrics_desc:
    #         def get_secondary_metric_value(cand):
    #             val_s = cand.get('validation_summary', {}).get(metric_name, float('inf') if ascending else -float('inf'))
    #             return val_s if isinstance(val_s, (int, float)) and not pd.isna(val_s) else (float('inf') if ascending else -float('inf'))
            
    #         top_n_primary.sort(key=get_secondary_metric_value, reverse=not ascending)
    #     best_candidate = top_n_primary[0]
    #     logger.info(f"通过次要标准选择的基因: {best_candidate['genes']}")


    logger.info(f"最终选择的基因 (基于主要指标 '{primary_metric}'): {best_candidate['genes']}")
    return best_candidate['genes']


if __name__ == '__main__':
    logger.info("Trading Bot Initializing...")
    # ... (okx_conn, data_hdl 初始化保持不变) ...
    okx_conn = None; data_hdl = None
    try:
        okx_conn = OKXConnector(); data_hdl = DataHandler(okx_conn)
    except ValueError as ve: logger.critical(f"Init Error: {ve}"); exit()
    except Exception as e_init: logger.critical(f"Unexpected init error: {e_init}", exc_info=True); exit()

    DEFAULT_MODE = "optimize"
    MODE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ["optimize", "backtest_best", "backtest_oos", "live_best"] else DEFAULT_MODE
    logger.info(f"Mode: {MODE}")
    
    logger.info(f"\n--- Running in Mode: {MODE} ---")

    if MODE == "optimize":
        logger.info(f"\n--- Running Genetic Optimization Mode (with Enhanced Validation Selection) ---")
        logger.info(f"Fetching data for GA (Train+Validation): {config.SYMBOL} ({config.TIMEFRAME}). "
                    f"Target: {config.KLINE_LIMIT_FOR_OPTIMIZATION} klines.")
        
        total_optimization_df = data_hdl.fetch_klines_to_df(
            instId=config.SYMBOL, bar=config.TIMEFRAME,
            total_limit_needed=config.KLINE_LIMIT_FOR_OPTIMIZATION, 
            kline_type="history_market"
        )
        
        max_period = 0
        for _, detail in config.GENE_SCHEMA.items():
            if detail['type'] == 'int' and 'range' in detail: max_period = max(max_period, detail['range'][1])
        min_data_needed = max_period + 100
        
        if total_optimization_df.empty or len(total_optimization_df) < min_data_needed :
            logger.critical(f"Not enough data ({len(total_optimization_df)}) for GA. Need ~{min_data_needed}. Exiting."); exit()
        logger.info(f"Fetched {len(total_optimization_df)} klines for GA.")
        
        passed_validation_candidates_list: List[Dict[str, Any]] = []
        validation_metric_name_used = ""

        try:
            optimizer = GeneticOptimizerPhase1(total_optimization_df=total_optimization_df)
            # run_evolution 返回所有通过验证的候选者列表和使用的主要验证指标名
            passed_validation_candidates_list, validation_metric_name_used = optimizer.run_evolution() 
        except ValueError as ve_ga_init: logger.critical(f"GA Init Error: {ve_ga_init}. Exiting."); exit()
        except Exception as e_ga_run: logger.critical(f"GA Run Error: {e_ga_run}", exc_info=True); exit()

        best_genes_final_selection: Optional[Dict[str, Any]] = None
        if passed_validation_candidates_list:
            logger.info(f"GA finished. Found {len(passed_validation_candidates_list)} candidates that passed validation criteria.")
            # 定义次要排序标准 (可选)
            secondary_sort_criteria = [
                ('Max Drawdown', True),      # True表示越小越好
                ('Profit Factor', False),    # False表示越大越好
                ('Total Trades', False)      # 交易次数越多越好 (在满足其他条件下)
            ]
            best_genes_final_selection = select_final_best_gene_from_candidates(
                passed_validation_candidates_list,
                validation_metric_name_used, # 这个应该是 config.VALIDATION_PRIMARY_METRIC
                secondary_metrics_desc=secondary_sort_criteria 
            )
        else:
            logger.warning("GA did not yield any candidates passing all validation criteria.")

        if best_genes_final_selection:
            logger.info(f"\n--- Genetic Optimization Finished ---")
            logger.info(f"Final Best Genes selected: {best_genes_final_selection}")
            # (这里不再打印单一的 best_metric_value，因为选择标准可能更复杂)
            try:
                with open(BEST_GENES_FILE, 'w') as f: json.dump(best_genes_final_selection, f, indent=4)
                logger.info(f"Final best genes saved to {BEST_GENES_FILE}")
            except Exception as e_save: logger.error(f"Error saving {BEST_GENES_FILE}: {e_save}")
            
            logger.info("\n--- Final Backtest with Selected Genes (on ENTIRE Optimization Dataset) ---")
            try:
                final_strategy = EvolvableStrategy(best_genes_final_selection)
                final_backtester = Backtester(
                    data_df=total_optimization_df.copy(), strategy_instance=final_strategy,
                    initial_capital=config.INITIAL_CAPITAL, fee_rate=config.FEE_RATE
                )
                summary_final, trades_df_final = final_backtester.run_backtest()
                if summary_final: 
                    logger.info("Summary for Selected Genes on Entire Optimization Dataset:")
                    final_backtester.print_summary(summary_final)
                if trades_df_final is not None and not trades_df_final.empty: 
                    logger.info("\nTrades (Selected Genes on Entire Optimization Dataset, first 5):\n" + \
                                str(trades_df_final.head()))
            except KeyError as ke_final_bt:
                 logger.error(f"KeyError in final backtest with genes {best_genes_final_selection}: {ke_final_bt}.")
            except Exception as e_final_bt: 
                logger.error(f"Error in final backtest: {e_final_bt}", exc_info=True)
        else: 
            logger.warning("GA did not select a final best gene set.")

    # ... (backtest_best, backtest_oos, live_best 模式基本保持不变, 它们都从 BEST_GENES_FILE 加载基因) ...
    elif MODE == "backtest_best": 
        logger.info(f"\n--- Running Backtest with Best Genes from {BEST_GENES_FILE} (on ENTIRE Optimization Dataset) ---")
        if not os.path.exists(BEST_GENES_FILE): 
            logger.critical(f"Error: {BEST_GENES_FILE} not found. Run 'optimize' mode first. Exiting.")
            exit()
        try:
            with open(BEST_GENES_FILE, 'r') as f: best_genes = json.load(f)
            logger.info(f"Loaded best genes: {best_genes}")
            dataset_for_backtest_best = data_hdl.fetch_klines_to_df(
                instId=config.SYMBOL, bar=config.TIMEFRAME,
                total_limit_needed=config.KLINE_LIMIT_FOR_OPTIMIZATION, 
                kline_type="history_market"
            )
            if dataset_for_backtest_best.empty or len(dataset_for_backtest_best) < config.KLINE_LIMIT_FOR_OPTIMIZATION * 0.9:
                logger.critical(f"Not enough data for backtest_best (fetched {len(dataset_for_backtest_best)}). Exiting.")
                exit()
            logger.info(f"Fetched {len(dataset_for_backtest_best)} klines for backtest_best.")
            strategy_instance = EvolvableStrategy(best_genes)
            backtester = Backtester(
                data_df=dataset_for_backtest_best.copy(), strategy_instance=strategy_instance,
                initial_capital=config.INITIAL_CAPITAL, fee_rate=config.FEE_RATE
            )
            summary, trades_df = backtester.run_backtest()
            if summary: 
                logger.info("Summary for 'backtest_best':")
                backtester.print_summary(summary)
            if trades_df is not None and not trades_df.empty: 
                logger.info("\nTrades ('backtest_best', first 5):\n" + str(trades_df.head()))
        except KeyError as ke_bt_best: logger.error(f"KeyError in 'backtest_best' strategy init: {ke_bt_best}.")
        except Exception as e_bt_best: logger.error(f"Error in 'backtest_best': {e_bt_best}", exc_info=True)

    elif MODE == "backtest_oos":
        logger.info(f"\n--- Running OUT-OF-SAMPLE Backtest with Best Genes from {BEST_GENES_FILE} ---")
        if not os.path.exists(BEST_GENES_FILE):
            logger.critical(f"Error: {BEST_GENES_FILE} not found. Run 'optimize' mode first. Exiting.")
            exit()
        try:
            with open(BEST_GENES_FILE, 'r') as f: best_genes = json.load(f)
            logger.info(f"Loaded best genes for OOS test: {best_genes}")
            logger.info(f"Optimization (Train+Validation) used approx. {config.KLINE_LIMIT_FOR_OPTIMIZATION} klines.")
            logger.info(f"OOS test will use {config.KLINE_LIMIT_FOR_OOS} klines PRECEDING the Optimization data.")
            total_data_for_oos_setup = config.KLINE_LIMIT_FOR_OPTIMIZATION + config.KLINE_LIMIT_FOR_OOS
            logger.info(f"Fetching a total of {total_data_for_oos_setup} klines...")
            all_historical_data = data_hdl.fetch_klines_to_df(
                instId=config.SYMBOL, bar=config.TIMEFRAME,
                total_limit_needed=total_data_for_oos_setup, kline_type="history_market"
            )
            if all_historical_data.empty or len(all_historical_data) < total_data_for_oos_setup * 0.95:
                logger.critical(f"Not enough data fetched ({len(all_historical_data)}) for OOS setup. Need ~{total_data_for_oos_setup}."); exit()
            logger.info(f"Fetched {len(all_historical_data)} total klines.")
            oos_data_end_index = config.KLINE_LIMIT_FOR_OOS
            if len(all_historical_data) < config.KLINE_LIMIT_FOR_OOS:
                logger.critical(f"Fetched data {len(all_historical_data)} < KLINE_LIMIT_FOR_OOS {config.KLINE_LIMIT_FOR_OOS}."); exit()
            oos_data_df = all_historical_data.iloc[:oos_data_end_index]
            if oos_data_df.empty or len(oos_data_df) < config.KLINE_LIMIT_FOR_OOS * 0.9:
                 logger.critical(f"Not enough data for OOS after slicing (got {len(oos_data_df)})."); exit()
            
            oos_start_time_str = str(oos_data_df.index.min()) if isinstance(oos_data_df.index, pd.DatetimeIndex) else "N/A"
            oos_end_time_str = str(oos_data_df.index.max()) if isinstance(oos_data_df.index, pd.DatetimeIndex) else "N/A"
            logger.info(f"Using {len(oos_data_df)} klines for OOS backtest (from {oos_start_time_str} to {oos_end_time_str}).")

            oos_strategy_instance = EvolvableStrategy(best_genes)
            oos_backtester = Backtester(
                data_df=oos_data_df.copy(), strategy_instance=oos_strategy_instance,
                initial_capital=config.INITIAL_CAPITAL, fee_rate=config.FEE_RATE
            ) 
            oos_summary, oos_trades_df = oos_backtester.run_backtest()
            logger.info("\n--- OUT-OF-SAMPLE Backtest Summary ---")
            if oos_summary: oos_backtester.print_summary(oos_summary)
            if oos_trades_df is not None and not oos_trades_df.empty:
                 logger.info("\nTrades Log from OOS backtest (first 5):\n" + str(oos_trades_df.head()))
                 try:
                    oos_data_df.to_csv("oos_kline_data.csv") 
                    logger.info("OOS K-line data (used in this test) saved to oos_kline_data.csv")
                    oos_trades_df.to_csv("oos_trades_log.csv", index=False) 
                    logger.info("OOS trades log (from this test) saved to oos_trades_log.csv")
                 except Exception as e_save_oos: logger.error(f"Error saving OOS results: {e_save_oos}")
            else: logger.info("No trades executed in OOS backtest.")
        except KeyError as ke_bt_oos: logger.error(f"KeyError in 'backtest_oos' strategy init: {ke_bt_oos}.")
        except Exception as e_bt_oos: logger.error(f"Error during 'backtest_oos' mode: {e_bt_oos}", exc_info=True)

    elif MODE == "live_best":
        # (与上一版本相同)
        logger.info(f"\n--- Running Live Trading with Best Genes from {BEST_GENES_FILE} (Conceptual) ---")
        if not os.path.exists(BEST_GENES_FILE): 
            logger.critical(f"Error: {BEST_GENES_FILE} not found. Run 'optimize' mode first. Exiting.")
            exit()
        try:
            with open(BEST_GENES_FILE, 'r') as f: best_genes = json.load(f)
            logger.info(f"Loaded best genes: {best_genes}")
            run_live_trading_loop(okx_conn, data_hdl, best_genes)
        except KeyError as ke_live: logger.error(f"KeyError in live trading strategy init: {ke_live}.")
        except Exception as e_live_setup: logger.error(f"Error setting up 'live_best': {e_live_setup}", exc_info=True)
    else:
        logger.error(f"Unknown mode: {MODE}.")

    logger.info("\nTrading Bot Script Finished.")
