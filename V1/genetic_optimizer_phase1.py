# trading_bot/genetic_optimizer_phase1.py
import random
import copy
import numpy as np
import pandas as pd
import json
import logging
from typing import Optional, List, Tuple, Dict, Any
import multiprocessing

import config
from strategy import EvolvableStrategy
from backtester import Backtester

logger = logging.getLogger(f"trading_bot.{__name__}")

GA_LOG_FILE = "ga_evolution_log.jsonl"
VALIDATION_CANDIDATES_FILE = "validation_candidates_log.jsonl"
ALL_VALIDATION_ATTEMPTS_LOG_FILE = "all_validation_attempts_log.jsonl"

# calculate_fitness_for_individual_process (REMAINS THE SAME as the version from my response timestamped 2023-03-15 00:00:00 UTC, 
# i.e., the one that correctly uses the new FITNESS_WEIGHTS from config.py)
def calculate_fitness_for_individual_process(args_tuple: Tuple) -> float:
    genes_copy, training_df_copy, initial_capital, fee_rate, \
    risk_per_trade_pct, min_pos_size, fitness_weights_cfg_copy, _ = args_tuple
    fw = fitness_weights_cfg_copy
    try:
        strategy_instance_p = EvolvableStrategy(genes_copy)
        if not strategy_instance_p._validate_parameters_after_constraints():
            return fw['base_penalty'] - 1000 
        backtester_p = Backtester(
            data_df=training_df_copy, strategy_instance=strategy_instance_p,
            initial_capital=initial_capital, fee_rate=fee_rate,
            risk_per_trade_percent=risk_per_trade_pct, min_position_size=min_pos_size
        )
        summary, _ = backtester_p.run_backtest()
        if summary is None: return fw['base_penalty']
        fitness_score = 0.0
        num_trades = summary.get('Total Trades', 0)
        if num_trades < fw['train_min_trades_threshold']:
            missing_trades = fw['train_min_trades_threshold'] - num_trades
            fitness_score += missing_trades * fw['penalty_per_missing_trade']
        max_dd_summary = summary.get('Max Drawdown', 100.0) / 100.0
        if max_dd_summary > fw['train_ideal_max_drawdown']:
            exceed_dd = max_dd_summary - fw['train_ideal_max_drawdown'] # Define exceed_dd here
            if max_dd_summary > fw['train_acceptable_max_drawdown']:
                fitness_score += fw['base_penalty'] / 10
            else:
                penalty_factor = (exceed_dd / (fw['train_acceptable_max_drawdown'] - fw['train_ideal_max_drawdown'])) \
                                  if (fw['train_acceptable_max_drawdown'] - fw['train_ideal_max_drawdown']) > 0.001 else exceed_dd * 100
                drawdown_penalty = fw['train_drawdown_penalty_factor'] * (penalty_factor ** fw['train_drawdown_exponent'])
                fitness_score += drawdown_penalty
        profit_factor = summary.get('Profit Factor', 0.0)
        if profit_factor < fw['train_min_profit_factor']:
            fitness_score += (profit_factor - fw['train_min_profit_factor']) * abs(fw['train_profit_factor_weight'] * 2)
        else:
            pf_bonus = min( (profit_factor - fw['train_min_profit_factor']) / \
                            (fw['train_profit_factor_target'] - fw['train_min_profit_factor']), 1.0) \
                       if (fw['train_profit_factor_target'] - fw['train_min_profit_factor']) > 0.01 else 0
            fitness_score += pf_bonus * fw['train_profit_factor_weight']
        sharpe_ratio = summary.get('Sharpe Ratio', -5.0)
        if sharpe_ratio < 0:
            fitness_score += sharpe_ratio * fw['train_sharpe_ratio_weight'] * 1.5
        else:
            fitness_score += sharpe_ratio * fw['train_sharpe_ratio_weight']
        pnl_pct_summary = summary.get('Total PnL Pct', -100.0) / 100.0
        fitness_score += pnl_pct_summary * fw['train_pnl_pct_weight']
        if np.isinf(fitness_score) or np.isnan(fitness_score): return fw['base_penalty'] / 100
        if fitness_score < (fw['base_penalty'] / 1000) and num_trades >= fw['train_min_trades_threshold']:
             return (fw['base_penalty'] / 1000) + fitness_score
        return float(fitness_score)
    except Exception:
        return fitness_weights_cfg_copy['base_penalty']


# Helper function for validation in multiprocessing - RETURNS TUPLE: (task_id, genes_originally_sent, validation_summary)
def evaluate_validation_for_individual_process(args_tuple: Tuple) -> Tuple[Any, Dict[str, Any], Optional[Dict[str, Any]]]:
    task_id, genes_val_copy, validation_df_copy, initial_capital, fee_rate, \
    risk_per_trade_pct_val, min_pos_size_val, gene_schema_val_copy = args_tuple

    try:
        strategy_instance_v = EvolvableStrategy(genes_val_copy) 
        if not strategy_instance_v._validate_parameters_after_constraints():
            return task_id, genes_val_copy, None 

        backtester_v = Backtester(
            data_df=validation_df_copy, strategy_instance=strategy_instance_v,
            initial_capital=initial_capital, fee_rate=fee_rate,
            risk_per_trade_percent=risk_per_trade_pct_val, min_position_size=min_pos_size_val
        )
        summary, _ = backtester_v.run_backtest()
        
        if summary and config.VALIDATION_PRIMARY_METRIC == 'Calmar_like' and 'Calmar_like' not in summary:
            pnl_val = summary.get('Total PnL', 0.0)
            pnl_pct_val = pnl_val / initial_capital if initial_capital > 0 else 0.0
            max_dd_pct_val = summary.get('Max Drawdown', 100.0) / 100.0 
            if max_dd_pct_val < 0.00001:
                summary['Calmar_like'] = pnl_pct_val * 1000 if pnl_pct_val > 0 else pnl_pct_val * 100
            else:
                summary['Calmar_like'] = pnl_pct_val / max_dd_pct_val
                
        return task_id, genes_val_copy, summary 
    except Exception:
        return task_id, genes_val_copy, None


class GeneticOptimizerPhase1:
    # ... (__init__, _create_individual_genes, initialize_population, etc. REMAINS THE SAME as the one with new fitness func) ...
    def __init__(self, total_optimization_df: pd.DataFrame):
        self.total_optimization_df = total_optimization_df.copy()
        val_ratio = config.VALIDATION_SET_RATIO
        if not (0 < val_ratio < 1): logger.critical(f"..."); raise ValueError("...")
        if not isinstance(self.total_optimization_df.index, pd.DatetimeIndex):
             try: self.total_optimization_df.index = pd.to_datetime(self.total_optimization_df.index)
             except Exception as e_conv: logger.critical(f"..."); raise TypeError("...")
        if not self.total_optimization_df.index.is_monotonic_increasing:
            logger.warning("..."); self.total_optimization_df.sort_index(inplace=True)
            if not self.total_optimization_df.index.is_monotonic_increasing: logger.critical("..."); raise ValueError("...")
        split_point_index = int(len(self.total_optimization_df) * (1 - val_ratio))
        self.training_df = self.total_optimization_df.iloc[:split_point_index].copy()
        self.validation_df = self.total_optimization_df.iloc[split_point_index:].copy()
        longest_ma_period = config.GENE_SCHEMA['long_ma_period']['range'][1] if 'long_ma_period' in config.GENE_SCHEMA else 0
        longest_atr_period = config.GENE_SCHEMA['atr_period']['range'][1] if 'atr_period' in config.GENE_SCHEMA else 0
        min_data_len = max(max(longest_ma_period, longest_atr_period) + 20, 50)
        if len(self.training_df) < min_data_len or len(self.validation_df) < min_data_len:
            logger.critical(f"Data too short..."); raise ValueError("Data too short...")
        logger.info(f"Data loaded: Total={len(self.total_optimization_df)}, Train={len(self.training_df)}, Val={len(self.validation_df)}")
        self.population_size = config.POPULATION_SIZE; self.num_generations = config.NUM_GENERATIONS
        self.mutation_rate = config.MUTATION_RATE; self.crossover_rate = config.CROSSOVER_RATE
        self.elitism_count = config.ELITISM_COUNT; self.gene_schema = config.GENE_SCHEMA
        self.fitness_weights_config = config.FITNESS_WEIGHTS
        self.population: List[Dict[str, Any]] = []; self.fitness_scores_on_train: List[float] = []
        self.ga_log: List[Dict[str, Any]] = []; self.passed_validation_candidates: List[Dict[str, Any]] = []
        self.all_validation_attempts_log: List[Dict[str, Any]] = []

    def _create_individual_genes(self) -> Dict[str, Any]: # Remains same
        genes: Dict[str, Any] = {}; # ... (logic as before)
        for param_name, details in self.gene_schema.items():
            if details['type'] == 'int': genes[param_name] = random.randint(details['range'][0], details['range'][1])
            elif details['type'] == 'float': genes[param_name] = random.uniform(details['range'][0], details['range'][1])
            elif details['type'] == 'bool':
                if not isinstance(details.get('range'), list) or not all(isinstance(item, bool) for item in details['range']): genes[param_name] = True
                else: genes[param_name] = random.choice(details['range'])
        return genes

    def initialize_population(self): # Remains same
        self.population = [self._create_individual_genes() for _ in range(self.population_size)]; logger.info(f"Population initialized: {len(self.population)} individuals.")

    def _calculate_fitness_on_train_serial(self, genes: Dict[str, Any]) -> float: # Remains same
        args = (copy.deepcopy(genes), self.training_df, config.INITIAL_CAPITAL, config.FEE_RATE,
                config.RISK_PER_TRADE_PERCENT, config.MIN_POSITION_SIZE, self.fitness_weights_config, self.gene_schema)
        return calculate_fitness_for_individual_process(args)

    def _evaluate_population_on_train(self): # Remains same (parallel logic)
        try: num_cores = multiprocessing.cpu_count() - 1; num_cores = max(1, num_cores)
        except NotImplementedError: num_cores = 1
        if num_cores > 1 and self.population_size >= num_cores : # condition to use pool
            logger.info(f"Parallel train fitness eval using {num_cores} cores...")
            tasks_args = [(copy.deepcopy(g), self.training_df, config.INITIAL_CAPITAL, config.FEE_RATE,
                           config.RISK_PER_TRADE_PERCENT, config.MIN_POSITION_SIZE, self.fitness_weights_config, self.gene_schema)
                          for g in self.population]
            try:
                with multiprocessing.Pool(processes=num_cores) as pool: self.fitness_scores_on_train = pool.map(calculate_fitness_for_individual_process, tasks_args)
                logger.info("Parallel train fitness eval complete.")
            except Exception as e_pool:
                logger.error(f"Pool error: {e_pool}", exc_info=True); logger.info("Fallback to serial eval...")
                self.fitness_scores_on_train = [self._calculate_fitness_on_train_serial(g) for g in self.population]
        else:
            logger.info("Serial train fitness eval..."); self.fitness_scores_on_train = [self._calculate_fitness_on_train_serial(g) for g in self.population]

    def _evaluate_on_validation_serial(self, genes: Dict[str, Any], task_id_placeholder=0) -> Optional[Dict[str, Any]]: # Added task_id_placeholder
        args = (task_id_placeholder, copy.deepcopy(genes), self.validation_df, config.INITIAL_CAPITAL, config.FEE_RATE,
                config.RISK_PER_TRADE_PERCENT, config.MIN_POSITION_SIZE, self.gene_schema)
        _, _, summary = evaluate_validation_for_individual_process(args)
        return summary
        
    def _is_validation_result_acceptable(self, validation_summary: Optional[Dict[str, Any]]) -> bool:
        if validation_summary is None:
            return False
        
        passed_all = True # 初始化 passed_all 为 True

        # 检查交易次数
        if validation_summary.get('Total Trades', 0) < config.MIN_TRADES_FOR_VALIDATION:
            # logger.debug(f"Validation Fail: Trades {validation_summary.get('Total Trades', 0)} < {config.MIN_TRADES_FOR_VALIDATION}")
            passed_all = False
        
        # 检查最大回撤
        # Max Drawdown 在 summary 中是百分比 (e.g., 15.0 for 15%), config 中是小数 (e.g., 0.25 for 25%)
        current_max_dd_pct = validation_summary.get('Max Drawdown', 100.0) 
        if (current_max_dd_pct / 100.0) > config.MAX_DRAWDOWN_FOR_VALIDATION:
            # logger.debug(f"Validation Fail: MaxDD {current_max_dd_pct/100.0:.2%} > {config.MAX_DRAWDOWN_FOR_VALIDATION:.2%}")
            passed_all = False
        
        # 检查 PnL 百分比
        # Total PnL Pct 在 summary 中是百分比 (e.g., 10.0 for 10%), config 中是小数 (e.g., 0.01 for 1%)
        current_pnl_pct = validation_summary.get('Total PnL Pct', -100.0)
        if (current_pnl_pct / 100.0) < config.MIN_PNL_PCT_FOR_VALIDATION:
            # logger.debug(f"Validation Fail: PnLPct {current_pnl_pct/100.0:.2%} < {config.MIN_PNL_PCT_FOR_VALIDATION:.2%}")
            passed_all = False
            
        # 检查盈利因子
        current_profit_factor = validation_summary.get('Profit Factor', 0.0)
        if current_profit_factor < config.MIN_PROFIT_FACTOR_FOR_VALIDATION:
            # logger.debug(f"Validation Fail: ProfitFactor {current_profit_factor:.2f} < {config.MIN_PROFIT_FACTOR_FOR_VALIDATION:.2f}")
            passed_all = False
            
        # 检查夏普比率
        current_sharpe_ratio = validation_summary.get('Sharpe Ratio', -float('inf'))
        if current_sharpe_ratio < config.MIN_SHARPE_FOR_VALIDATION:
            # logger.debug(f"Validation Fail: Sharpe {current_sharpe_ratio:.2f} < {config.MIN_SHARPE_FOR_VALIDATION:.2f}")
            passed_all = False
            
        return passed_all

    def _selection(self) -> List[Dict[str, Any]]: # Remains same
        parents = []; population_indices = list(range(self.population_size))
        for _ in range(self.population_size):
            competitors = random.sample(population_indices, min(3, self.population_size))
            winner_idx = competitors[0] # Default winner
            for comp_idx in competitors[1:]:
                if self.fitness_scores_on_train[comp_idx] is not None and \
                   (self.fitness_scores_on_train[winner_idx] is None or \
                    self.fitness_scores_on_train[comp_idx] > self.fitness_scores_on_train[winner_idx]):
                    if not (np.isnan(self.fitness_scores_on_train[comp_idx]) or np.isinf(self.fitness_scores_on_train[comp_idx])):
                         winner_idx = comp_idx
            parents.append(copy.deepcopy(self.population[winner_idx]))
        return parents

    def _crossover_individual(self, p1_g: Dict[str, Any], p2_g: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]: # Remains same
        c1, c2 = copy.deepcopy(p1_g), copy.deepcopy(p2_g)
        if random.random() < self.crossover_rate:
            keys = list(self.gene_schema.keys())
            if len(keys) >= 2:
                p1, p2 = sorted(random.sample(range(len(keys)), 2))
                for i in range(p1, p2 + 1): k = keys[i]; c1[k], c2[k] = c2.get(k, c1[k]), c1.get(k, c2[k])
            elif len(keys) == 1 and random.random() < 0.5: k = keys[0]; c1[k], c2[k] = c2.get(k,c1[k]), c1.get(k,c2[k])
        return c1, c2

    def _mutate_individual(self, genes: Dict[str, Any]) -> Dict[str, Any]: # Remains same
        mut_g = copy.deepcopy(genes)
        for name in mut_g:
            if name not in self.gene_schema: continue
            if random.random() < self.mutation_rate:
                det = self.gene_schema[name]; curr_v = mut_g[name]
                if det['type'] == 'int':
                    r_min, r_max = det['range']; sig = max(1.0, float(r_max - r_min) * 0.1)
                    chg = int(random.gauss(0, sig)); new_v = curr_v + chg; mut_g[name] = max(r_min, min(new_v, r_max))
                elif det['type'] == 'float':
                    r_min, r_max = det['range']; sig = max(float(r_max - r_min) * 0.01, float(r_max - r_min) * 0.1)
                    if sig == 0: sig = 0.1
                    chg = random.gauss(0, sig); new_v = curr_v + chg; mut_g[name] = max(r_min, min(new_v, r_max))
                elif det['type'] == 'bool': mut_g[name] = not curr_v
        return mut_g

    # *** KEY MODIFICATIONS IN run_evolution for task_id based matching ***
    def run_evolution(self) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        logger.info("开始进化过程 (使用新的适应度函数和TASK_ID修复的并行验证)...")
        self.initialize_population()
        self.ga_log = []
        self.passed_validation_candidates = []
        self.all_validation_attempts_log = []
        best_fitness_on_train_overall = -float('inf')
        best_genes_on_train_overall: Optional[Dict[str, Any]] = None

        try:
            num_cores_val_pool = multiprocessing.cpu_count() - 1
            if num_cores_val_pool < 1: num_cores_val_pool = 1
        except NotImplementedError:
            num_cores_val_pool = 1
        
        # Store candidates for validation with their original train_fitness and a unique task_id
        # This list will be [{ "task_id": unique_int, "genes": {...}, "train_fitness": float, "original_pop_index": int }, ...]
        
        for generation in range(self.num_generations):
            self._evaluate_population_on_train()

            # Collect valid individuals from the current population along with their fitness and original index
            # List of tuples: (genes_dict, fitness_score, original_population_index)
            current_gen_pop_with_fitness_and_idx = []
            for i, fitness_val in enumerate(self.fitness_scores_on_train):
                if fitness_val is not None and not np.isnan(fitness_val) and not np.isinf(fitness_val) \
                   and fitness_val > config.FITNESS_WEIGHTS['base_penalty'] * 0.9: # Filter out extreme penalties early
                    current_gen_pop_with_fitness_and_idx.append((self.population[i], fitness_val, i))
            
            if not current_gen_pop_with_fitness_and_idx:
                logger.warning(f"第 {generation + 1} 代 - 没有个体具有有效或足够好的训练适应度。")
                if generation < self.num_generations - 1: self.initialize_population()
                continue

            # Sort by fitness to find the best for logging
            current_gen_pop_with_fitness_and_idx.sort(key=lambda item: item[1], reverse=True)
            
            current_best_genes_this_gen_on_train, current_best_train_fitness_val, _ = current_gen_pop_with_fitness_and_idx[0]

            if current_best_train_fitness_val > best_fitness_on_train_overall:
                best_fitness_on_train_overall = current_best_train_fitness_val
                best_genes_on_train_overall = copy.deepcopy(current_best_genes_this_gen_on_train)
                logger.info(f"第 {generation + 1} 代：新的训练集最佳总体适应度：{best_fitness_on_train_overall:.4f}")

            # Log generation stats
            valid_scores = [item[1] for item in current_gen_pop_with_fitness_and_idx]
            avg_train_fitness = np.mean(valid_scores) if valid_scores else -float('inf')
            std_train_fitness = np.std(valid_scores) if len(valid_scores) > 1 else 0.0
            logger.info(f"第 {generation + 1}/{self.num_generations} 代 (训练集)：最佳适应度={current_best_train_fitness_val:.2f}, "
                        f"平均适应度={avg_train_fitness:.2f}, 标准差={std_train_fitness:.2f}")

            # Prepare candidates for validation
            # validation_input_list will store dicts: {"task_id": ..., "genes": ..., "train_fitness": ...}
            validation_input_list = []
            task_id_counter = 0
            for genes_cand, train_fitness_cand, _ in current_gen_pop_with_fitness_and_idx:
                if train_fitness_cand >= config.MIN_TRAIN_FITNESS_FOR_VALIDATION:
                    validation_input_list.append({
                        "task_id": task_id_counter,
                        "genes": copy.deepcopy(genes_cand), # Deepcopy for safety with multiprocessing
                        "train_fitness": train_fitness_cand
                    })
                    task_id_counter += 1
            
            # Ensure the best of the current generation is always considered for validation if not already included
            if current_gen_pop_with_fitness_and_idx: # If there are any valid individuals
                best_genes_of_gen, best_fitness_of_gen, _ = current_gen_pop_with_fitness_and_idx[0]
                is_best_in_val_list = any(
                    # Simple dict comparison might fail due to float precision after deepcopy.
                    # Comparing by a unique aspect or if task_ids were assigned based on original_pop_index
                    # For now, rely on MIN_TRAIN_FITNESS_FOR_VALIDATION. If best doesn't meet it, it won't be added.
                    # If it does meet, it should be in validation_input_list.
                    # This check is a bit redundant if the list is built correctly from sorted individuals.
                    item['train_fitness'] == best_fitness_of_gen and \
                    json.dumps(item['genes'], sort_keys=True) == json.dumps(best_genes_of_gen, sort_keys=True) # More robust check
                    for item in validation_input_list
                )
                if not is_best_in_val_list and best_fitness_of_gen >= config.MIN_TRAIN_FITNESS_FOR_VALIDATION:
                    logger.debug(f"第 {generation + 1} 代：本代最佳个体之前未包含在验证列表中，现加入。")
                    validation_input_list.append({
                        "task_id": task_id_counter, # Ensure unique task_id
                        "genes": copy.deepcopy(best_genes_of_gen),
                        "train_fitness": best_fitness_of_gen
                    })


            logger.debug(f"第 {generation + 1} 代：{len(validation_input_list)} 个候选个体将进行验证集评估。")

            if validation_input_list:
                validation_tasks_args = [
                    (item["task_id"], item["genes"], self.validation_df, config.INITIAL_CAPITAL,
                     config.FEE_RATE, config.RISK_PER_TRADE_PERCENT, config.MIN_POSITION_SIZE, self.gene_schema)
                    for item in validation_input_list
                ]
                
                processed_validation_results_with_id = [] # List of (task_id, genes_from_process, summary_from_process)
                if num_cores_val_pool > 1 and len(validation_tasks_args) >= num_cores_val_pool:
                    logger.info(f"并行评估验证集，使用 {num_cores_val_pool} 个核心 (任务数: {len(validation_tasks_args)})...")
                    try:
                        with multiprocessing.Pool(processes=num_cores_val_pool) as pool_val:
                            processed_validation_results_with_id = pool_val.map(evaluate_validation_for_individual_process, validation_tasks_args)
                        logger.info("验证集并行评估完成。")
                    except Exception as e_pool_val:
                        logger.error(f"验证集多进程池执行错误: {e_pool_val}", exc_info=True); logger.info("回退到串行验证模式...")
                        processed_validation_results_with_id = [evaluate_validation_for_individual_process(args) for args in validation_tasks_args]
                else:
                    logger.info(f"串行评估验证集 (任务数: {len(validation_tasks_args)})...")
                    processed_validation_results_with_id = [evaluate_validation_for_individual_process(args) for args in validation_tasks_args]

                # Process results using task_id for matching
                for task_id_res, genes_from_process_res, validation_summary_res in processed_validation_results_with_id:
                    original_candidate_data = next((item for item in validation_input_list if item["task_id"] == task_id_res), None)
                    
                    if original_candidate_data is None:
                        logger.error(f"CRITICAL: 无法通过 task_id '{task_id_res}' 找到原始验证候选数据。跳过此验证结果。")
                        continue # This should not happen if task_ids are managed correctly
                    
                    # Genes from process should match original_candidate_data['genes'] because we sent a copy.
                    # We use original_candidate_data['genes'] for consistency.
                    original_genes_for_this_task = original_candidate_data['genes']
                    train_fitness_of_candidate = original_candidate_data["train_fitness"]

                    attempt_log_entry = {
                        "generation": generation + 1, "genes": original_genes_for_this_task, # Use genes that were sent
                        "train_fitness": float(train_fitness_of_candidate),
                        "validation_summary_attempted": copy.deepcopy(validation_summary_res) if validation_summary_res else None,
                        "passed_validation": False
                    }
                    if validation_summary_res and self._is_validation_result_acceptable(validation_summary_res):
                        attempt_log_entry["passed_validation"] = True
                        candidate_data_for_pool = {
                            "generation": generation + 1, "genes": original_genes_for_this_task,
                            "train_fitness": float(train_fitness_of_candidate),
                            "validation_summary": copy.deepcopy(validation_summary_res)
                        }
                        self.passed_validation_candidates.append(candidate_data_for_pool)
                        val_metric_value = validation_summary_res.get(config.VALIDATION_PRIMARY_METRIC, 'N/A')
                        val_metric_str = f"{val_metric_value:.4f}" if isinstance(val_metric_value, (float, np.floating)) else str(val_metric_value)
                        logger.info(f"  第 {generation + 1} 代：个体通过验证 (基因 short_ma={original_genes_for_this_task.get('short_ma_period')}). "
                                    f"训练适应度: {train_fitness_of_candidate:.2f}, 验证 {config.VALIDATION_PRIMARY_METRIC}: {val_metric_str}")
                    self.all_validation_attempts_log.append(attempt_log_entry)

            # --- GA Log and Next Generation (structure remains same) ---
            ga_gen_log = {
                "generation": generation + 1, "best_train_fitness_this_gen": float(current_best_train_fitness_val),
                "avg_train_fitness": float(avg_train_fitness), "std_train_fitness": float(std_train_fitness),
                "best_genes_on_train_this_gen": copy.deepcopy(current_best_genes_this_gen_on_train),
                "num_passed_validation_this_gen": sum(1 for c in self.passed_validation_candidates if c['generation'] == generation + 1)
            }
            self.ga_log.append(ga_gen_log)
            new_population = []
            if self.elitism_count > 0 and self.population and current_gen_pop_with_fitness_and_idx:
                # Elites are chosen from current_gen_pop_with_fitness_and_idx (already sorted by fitness)
                for elite_genes, _, _ in current_gen_pop_with_fitness_and_idx[:self.elitism_count]:
                    new_population.append(copy.deepcopy(elite_genes))
            
            selected_parents = self._selection() # Selection is based on self.population and self.fitness_scores_on_train
            num_offspring_to_create = self.population_size - len(new_population)
            if not selected_parents and num_offspring_to_create > 0:
                for _ in range(num_offspring_to_create): new_population.append(self._create_individual_genes())
            elif selected_parents:
                children_created_count = 0; parent_idx_counter = 0
                while children_created_count < num_offspring_to_create:
                    p1 = selected_parents[parent_idx_counter % len(selected_parents)]
                    p2 = selected_parents[(parent_idx_counter + 1) % len(selected_parents)] 
                    parent_idx_counter = (parent_idx_counter + 2) # Iterate through parents
                    if parent_idx_counter >= len(selected_parents) * 2 and len(selected_parents) > 1: # Avoid infinite loop for small parent pool by reshuffling
                        random.shuffle(selected_parents)
                        parent_idx_counter = 0


                    child1, child2 = self._crossover_individual(p1, p2)
                    new_population.append(self._mutate_individual(child1))
                    children_created_count += 1
                    if children_created_count >= num_offspring_to_create: break
                    new_population.append(self._mutate_individual(child2))
                    children_created_count += 1
            self.population = new_population[:self.population_size]
            while len(self.population) < self.population_size: self.population.append(self._create_individual_genes())
        
        logger.info("\n--- 进化完成 ---")
        if not self.passed_validation_candidates:
            logger.warning("在整个进化过程中，未能找到任何满足所有验证标准的候选基因。")
            if best_genes_on_train_overall: logger.info(f"训练集上的最佳表现来自：Fitness={best_fitness_on_train_overall:.4f}, Genes={best_genes_on_train_overall}")
            else: logger.info("训练集上也没有找到有效的最佳基因。")
        else: logger.info(f"共找到 {len(self.passed_validation_candidates)} 个通过验证的候选基因。")
        try:
            with open(GA_LOG_FILE, 'w') as f: [f.write(json.dumps(self._serialize_dict(e)) + '\n') for e in self.ga_log]
            logger.info(f"GA 进化日志已保存到 {GA_LOG_FILE}")
            with open(VALIDATION_CANDIDATES_FILE, 'w') as f: [f.write(json.dumps(self._serialize_dict(e)) + '\n') for e in self.passed_validation_candidates]
            logger.info(f"通过验证的候选基因已保存到 {VALIDATION_CANDIDATES_FILE}")
            with open(ALL_VALIDATION_ATTEMPTS_LOG_FILE, 'w') as f: [f.write(json.dumps(self._serialize_dict(e)) + '\n') for e in self.all_validation_attempts_log]
            logger.info(f"所有验证尝试日志已保存到 {ALL_VALIDATION_ATTEMPTS_LOG_FILE}")
        except Exception as e_log: logger.error(f"保存日志时出错: {e_log}", exc_info=True)
        return self.passed_validation_candidates, config.VALIDATION_PRIMARY_METRIC

    def _serialize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        serialized = {}
        if not isinstance(data, dict): return data 
        for key, value in data.items():
            if isinstance(value, dict): serialized[key] = self._serialize_dict(value)
            elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): serialized[key] = int(value)
            elif isinstance(value, (np.float_, np.float16, np.float32, np.float64, float)): # Added float
                if np.isnan(value): serialized[key] = None 
                elif np.isinf(value): serialized[key] = "Infinity" if value > 0 else "-Infinity" 
                else: serialized[key] = float(value)
            elif isinstance(value, (np.complex_, np.complex64, np.complex128)): serialized[key] = {'real': value.real, 'imag': value.imag}
            elif isinstance(value, (np.bool_, bool)): serialized[key] = bool(value) # Added bool
            elif isinstance(value, (np.void)): serialized[key] = None
            elif isinstance(value, pd.Timestamp): serialized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta): serialized[key] = str(value)
            elif isinstance(value, list): serialized[key] = [self._serialize_dict(item) if isinstance(item, dict) else (None if isinstance(item, float) and np.isnan(item) else item) for item in value]
            else:
                 try: json.dumps({key: value}); serialized[key] = value
                 except TypeError: serialized[key] = str(value)
        return serialized

if __name__ == '__main__':
    multiprocessing.freeze_support() 
    print("测试 Genetic Optimizer Phase 1 (独立运行 - 包含并行逻辑和新适应度函数)...")
    if not logger.hasHandlers():
        ch_test_ga = logging.StreamHandler(); ch_test_ga.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch_test_ga); logger.setLevel(logging.INFO)
        logging.getLogger('trading_bot.strategy').setLevel(logging.INFO); logging.getLogger('trading_bot.backtester').setLevel(logging.INFO)
    try: from okx_connector import OKXConnector; from data_handler import DataHandler
    except ImportError: print("错误：无法导入 OKXConnector 或 DataHandler。"); exit()
    print(f"正在为GA测试获取历史数据: {config.SYMBOL} ({config.TIMEFRAME})..."); historical_df_ga = pd.DataFrame()
    try:
        okx_conn_ga_test = OKXConnector(); data_hdl_ga_test = DataHandler(okx_conn_ga_test)
        historical_df_ga = data_hdl_ga_test.fetch_klines_to_df(instId=config.SYMBOL, bar=config.TIMEFRAME, total_limit_needed=config.KLINE_LIMIT_FOR_OPTIMIZATION, kline_type="history_market")
    except Exception as e_fetch: print(f"获取历史数据时发生错误: {e_fetch}"); exit()
    min_data_len_needed_for_ga_main = 250 + 100 
    if historical_df_ga.empty or len(historical_df_ga) < min_data_len_needed_for_ga_main: print(f"数据不足 ({len(historical_df_ga) if not historical_df_ga.empty else 0})。"); exit()
    print(f"已获取 {len(historical_df_ga)} 条K线数据。")
    try:
        optimizer_test_instance = GeneticOptimizerPhase1(total_optimization_df=historical_df_ga)
        passed_candidates_list, primary_metric_name = optimizer_test_instance.run_evolution()
    except Exception as e_init_run: print(f"GA 初始化或运行时发生意外错误: {e_init_run}"); import traceback; traceback.print_exc(); exit()
    final_best_genes = None
    if passed_candidates_list:
        print(f"\n--- 从 {len(passed_candidates_list)} 个候选中选择 (基于 {primary_metric_name}) ---")
        if passed_candidates_list[0].get('validation_summary', {}).get(primary_metric_name):
            best_c = max(passed_candidates_list, key=lambda c: c.get('validation_summary', {}).get(primary_metric_name, -float('inf')))
            final_best_genes = best_c['genes']; print(f"选定基因: {final_best_genes}")
    else: print("GA未能产生任何通过验证的候选。")
    if final_best_genes:
        print("\n--- 使用最终基因回测 ---")
        try:
            final_strategy_instance = EvolvableStrategy(final_best_genes)
            final_backtester_instance = Backtester(data_df=historical_df_ga.copy(), strategy_instance=final_strategy_instance, initial_capital=config.INITIAL_CAPITAL, fee_rate=config.FEE_RATE, risk_per_trade_percent=config.RISK_PER_TRADE_PERCENT, min_position_size=config.MIN_POSITION_SIZE)
            summary_final, _ = final_backtester_instance.run_backtest()
            if summary_final: final_backtester_instance.print_summary(summary_final)
        except Exception as e_final_bt: print(f"最终回测出错: {e_final_bt}")
    print("\nGenetic Optimizer Phase 1 独立测试完成。")
