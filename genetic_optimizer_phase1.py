# trading_bot/genetic_optimizer_phase1.py
import random
import copy
import numpy as np
import pandas as pd
import json
import logging
from typing import Optional, List, Tuple, Dict, Any

import config
from strategy import EvolvableStrategy
from backtester import Backtester

logger = logging.getLogger(f"trading_bot.{__name__}")

GA_LOG_FILE = "ga_evolution_log.jsonl" # 日志：基于训练集的GA进化过程
VALIDATION_CANDIDATES_FILE = "validation_candidates_log.jsonl" # 日志：所有通过验证的候选者
ALL_VALIDATION_ATTEMPTS_LOG_FILE = "all_validation_attempts_log.jsonl" # 新日志：所有验证尝试，无论通过与否

class GeneticOptimizerPhase1:
    def __init__(self, total_optimization_df: pd.DataFrame):
        self.total_optimization_df = total_optimization_df.copy()
        val_ratio = config.VALIDATION_SET_RATIO
        if not (0 < val_ratio < 1):
            logger.critical(f"配置错误: VALIDATION_SET_RATIO ({val_ratio})。")
            raise ValueError("VALIDATION_SET_RATIO 必须在 0 和 1 之间。")
        if not isinstance(self.total_optimization_df.index, pd.DatetimeIndex) or \
           not self.total_optimization_df.index.is_monotonic_increasing:
            logger.warning("总优化数据的索引不是单调递增的DatetimeIndex。建议先排序。")

        split_point_index = int(len(self.total_optimization_df) * (1 - val_ratio))
        self.training_df = self.total_optimization_df.iloc[:split_point_index]
        self.validation_df = self.total_optimization_df.iloc[split_point_index:]

        min_data_len = 50
        if len(self.training_df) < min_data_len or len(self.validation_df) < min_data_len:
            msg = (f"训练集 ({len(self.training_df)}) 或验证集 ({len(self.validation_df)}) "
                   f"数据过少 (少于 {min_data_len} 条)。")
            logger.critical(msg)
            raise ValueError(msg)

        logger.info(f"总优化数据: {len(self.total_optimization_df)} 条, "
                    f"从 {self.total_optimization_df.index.min()} 到 {self.total_optimization_df.index.max()}")
        logger.info(f"训练集数据: {len(self.training_df)} 条, "
                    f"从 {self.training_df.index.min()} 到 {self.training_df.index.max()}")
        logger.info(f"验证集数据: {len(self.validation_df)} 条, "
                    f"从 {self.validation_df.index.min()} 到 {self.validation_df.index.max()}")

        self.population_size = config.POPULATION_SIZE
        self.num_generations = config.NUM_GENERATIONS
        self.mutation_rate = config.MUTATION_RATE
        self.crossover_rate = config.CROSSOVER_RATE
        self.elitism_count = config.ELITISM_COUNT
        self.gene_schema = config.GENE_SCHEMA
        self.fitness_weights_config = config.FITNESS_WEIGHTS

        self.population: List[Dict[str, Any]] = []
        self.fitness_scores_on_train: List[float] = []
        self.ga_log: List[Dict[str, Any]] = []
        
        self.passed_validation_candidates: List[Dict[str, Any]] = []
        self.all_validation_attempts_log: List[Dict[str, Any]] = [] # 新增：用于记录所有验证尝试


    def _ensure_gene_constraints(self, genes: Dict[str, Any]) -> Dict[str, Any]:
        if 'short_ma_period' in genes and 'long_ma_period' in genes:
            s_min, s_max = self.gene_schema['short_ma_period']['range']
            l_min, l_max = self.gene_schema['long_ma_period']['range']
            genes['short_ma_period'] = max(s_min, min(int(genes['short_ma_period']), s_max))
            genes['long_ma_period'] = max(l_min, min(int(genes['long_ma_period']), l_max))
            if genes['short_ma_period'] >= genes['long_ma_period']:
                genes['long_ma_period'] = min(genes['short_ma_period'] + random.randint(max(1, int((l_max-s_min)*0.05) if l_max > s_min else 1), 
                                                                          max(5, int((l_max-s_min)*0.1)) if l_max > s_min else 5), 
                                          l_max)
                if genes['short_ma_period'] >= genes['long_ma_period']:
                    genes['short_ma_period'] = max(s_min, genes['long_ma_period'] - random.randint(max(1, int((l_max-s_min)*0.05) if l_max > s_min else 1), 
                                                                                      max(5, int((l_max-s_min)*0.1)) if l_max > s_min else 5))
                    genes['short_ma_period'] = max(s_min, genes['short_ma_period']) 
                    if genes['short_ma_period'] >= genes['long_ma_period'] and l_max > s_min +1 :
                         genes['short_ma_period'] = (s_min + s_max) // 2
                         genes['long_ma_period'] = min(genes['short_ma_period'] + int((l_max-s_min)*0.2) if (l_max-s_min)*0.2 >=1 else 5 , l_max)
                         genes['long_ma_period'] = max(genes['long_ma_period'], genes['short_ma_period'] + 1)
        if 'atr_take_profit_multiplier' in genes and 'atr_stop_loss_multiplier' in genes:
            sl_mult_min, sl_mult_max = self.gene_schema['atr_stop_loss_multiplier']['range']
            tp_mult_min, tp_mult_max = self.gene_schema['atr_take_profit_multiplier']['range']
            genes['atr_stop_loss_multiplier'] = max(sl_mult_min, min(genes['atr_stop_loss_multiplier'], sl_mult_max))
            genes['atr_take_profit_multiplier'] = max(tp_mult_min, min(genes['atr_take_profit_multiplier'], tp_mult_max))
            if genes['atr_take_profit_multiplier'] <= genes['atr_stop_loss_multiplier']:
                genes['atr_take_profit_multiplier'] = genes['atr_stop_loss_multiplier'] + 0.1
                genes['atr_take_profit_multiplier'] = min(genes['atr_take_profit_multiplier'], tp_mult_max)
                if genes['atr_take_profit_multiplier'] <= genes['atr_stop_loss_multiplier'] and \
                   genes['atr_stop_loss_multiplier'] > sl_mult_min + 0.01:
                    genes['atr_stop_loss_multiplier'] = genes['atr_take_profit_multiplier'] - 0.1
                    genes['atr_stop_loss_multiplier'] = max(genes['atr_stop_loss_multiplier'], sl_mult_min)
        return genes

    def _create_individual_genes(self) -> Dict[str, Any]:
        genes: Dict[str, Any] = {}
        for param_name, details in self.gene_schema.items():
            if details['type'] == 'int':
                genes[param_name] = random.randint(details['range'][0], details['range'][1])
            elif details['type'] == 'float':
                genes[param_name] = random.uniform(details['range'][0], details['range'][1])
            elif details['type'] == 'bool': 
                if not isinstance(details['range'], list) or not all(isinstance(item, bool) for item in details['range']):
                    logger.error(f"基因 '{param_name}' 类型为 'bool'，但其 'range' 定义不正确 ({details['range']})。将使用默认值 True。")
                    genes[param_name] = True 
                else:
                    genes[param_name] = random.choice(details['range'])
            else:
                logger.warning(f"未知的基因类型 '{details['type']}' 用于基因 '{param_name}'。")
        return self._ensure_gene_constraints(genes)

    def initialize_population(self):
        self.population = [self._create_individual_genes() for _ in range(self.population_size)]
        logger.info(f"已初始化种群，包含 {len(self.population)} 个个体。")

    def _calculate_fitness_on_train(self, genes: Dict[str, Any]) -> float:
        try:
            strategy_instance = EvolvableStrategy(genes) 
            backtester = Backtester(
                data_df=self.training_df.copy(), 
                strategy_instance=strategy_instance,
                initial_capital=config.INITIAL_CAPITAL,
                fee_rate=config.FEE_RATE
            )
            summary, _ = backtester.run_backtest()
            fw_conf = self.fitness_weights_config
            base_penalty = fw_conf['hard_penalty_base']
            if summary is None: return base_penalty - 20000 
            pnl_total = summary.get('Total PnL', 0.0)
            max_dd_pct = summary.get('Max Drawdown', 1.0)
            num_trades = summary.get('Total Trades', 0)
            profit_factor = summary.get('Profit Factor', 0.0)
            if num_trades < fw_conf['min_trades_threshold']:
                return base_penalty + (num_trades * (abs(base_penalty) / (fw_conf['min_trades_threshold'] * 1000 + 1)))
            if max_dd_pct > fw_conf['max_allowed_drawdown']:
                return base_penalty * 0.8 - ((max_dd_pct - fw_conf['max_allowed_drawdown']) * abs(base_penalty) * 2)
            pnl_percentage_return = pnl_total / config.INITIAL_CAPITAL
            if pnl_percentage_return < fw_conf['significant_loss_threshold_pct']:
                return base_penalty * 0.7 - (abs(pnl_percentage_return - fw_conf['significant_loss_threshold_pct']) * abs(base_penalty))
            fitness_score: float
            if pnl_total <= 0:
                fitness_score = pnl_percentage_return * 100 
                if pnl_total == 0 and num_trades > 0 : fitness_score = -0.01
                elif num_trades == 0: fitness_score = -1.0 
            else:
                if max_dd_pct < 0.001: 
                    fitness_score = pnl_percentage_return * 1000 
                else:
                    fitness_score = pnl_percentage_return / max_dd_pct
                if profit_factor > 1.5: 
                    fitness_score += (profit_factor - 1.5) * 0.5 
            if np.isinf(fitness_score) or np.isnan(fitness_score):
                return base_penalty - 10000
            return float(fitness_score)
        except KeyError as ke: 
            logger.error(f"计算训练集适应度时发生 KeyError，基因 {genes}: {ke}。")
            return config.FITNESS_WEIGHTS['hard_penalty_base'] - 40000 
        except Exception as e:
            logger.error(f"计算训练集适应度时出错，基因 {genes}: {e}", exc_info=False) # 改为False，避免过多追溯信息刷屏
            return config.FITNESS_WEIGHTS['hard_penalty_base'] - 30000

    def _evaluate_on_validation(self, genes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            strategy_instance = EvolvableStrategy(genes) 
            backtester = Backtester(
                data_df=self.validation_df.copy(),
                strategy_instance=strategy_instance,
                initial_capital=config.INITIAL_CAPITAL,
                fee_rate=config.FEE_RATE
            )
            summary, _ = backtester.run_backtest() 
            if summary is None:
                logger.warning(f"验证集回测失败，基因: {genes}, 回测摘要为 None。")
                return None
            if config.VALIDATION_PRIMARY_METRIC == 'Calmar_like' and 'Calmar_like' not in summary:
                pnl_pct_val = summary.get('Total PnL', 0.0) / config.INITIAL_CAPITAL
                max_dd_pct_val = summary.get('Max Drawdown', 1.0)
                if max_dd_pct_val < 0.001 : 
                    summary['Calmar_like'] = pnl_pct_val * 1000 if pnl_pct_val > 0 else pnl_pct_val * 100
                else:
                    summary['Calmar_like'] = pnl_pct_val / max_dd_pct_val
            return summary 
        except KeyError as ke:
            logger.error(f"验证集评估时发生 KeyError，基因 {genes}: {ke}。")
            return None
        except Exception as e:
            logger.error(f"验证集评估时出错，基因 {genes}: {e}", exc_info=False) # 改为False
            return None

    def _is_validation_result_acceptable(self, validation_summary: Optional[Dict[str, Any]]) -> bool:
        if validation_summary is None: return False
        num_trades_val = validation_summary.get('Total Trades', 0)
        max_dd_val = validation_summary.get('Max Drawdown', 1.0)
        pnl_val_pct = validation_summary.get('Total PnL', 0.0) / config.INITIAL_CAPITAL
        pf_val = validation_summary.get('Profit Factor', 0.0)
        sharpe_val = validation_summary.get('Sharpe Ratio', -float('inf'))

        if num_trades_val < config.MIN_TRADES_FOR_VALIDATION: return False
        if max_dd_val > config.MAX_DRAWDOWN_FOR_VALIDATION: return False
        if pnl_val_pct < config.MIN_PNL_PCT_FOR_VALIDATION: return False
        if pf_val < config.MIN_PROFIT_FACTOR_FOR_VALIDATION: return False
        if sharpe_val < config.MIN_SHARPE_FOR_VALIDATION: return False
        return True

    def _evaluate_population_on_train(self):
        self.fitness_scores_on_train = [self._calculate_fitness_on_train(genes) for genes in self.population]

    def _selection(self) -> List[Dict[str, Any]]:
        parents = []
        for _ in range(self.population_size): 
            tournament_size = 3 
            valid_indices = [i for i, score in enumerate(self.fitness_scores_on_train) 
                             if score is not None and not np.isnan(score) and not np.isinf(score)]
            if not valid_indices: parents.append(self._create_individual_genes()); continue
            actual_tournament_size = min(tournament_size, len(valid_indices))
            if actual_tournament_size == 0 : parents.append(self._create_individual_genes()); continue
            competitor_indices_from_valid = random.sample(valid_indices, actual_tournament_size)
            winner_original_index = max(competitor_indices_from_valid, key=lambda idx: self.fitness_scores_on_train[idx])
            parents.append(self.population[winner_original_index])
        return parents

    def _crossover_individual(self, parent1_genes: Dict[str, Any], parent2_genes: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        child1_genes = copy.deepcopy(parent1_genes); child2_genes = copy.deepcopy(parent2_genes)
        if random.random() < self.crossover_rate:
            keys = list(self.gene_schema.keys())
            if len(keys) >= 2:
                point1, point2 = sorted(random.sample(range(len(keys)), 2))
                for i in range(point1, point2 + 1):
                    key_to_swap = keys[i]
                    if key_to_swap in child1_genes and key_to_swap in child2_genes:
                        child1_genes[key_to_swap], child2_genes[key_to_swap] = child2_genes[key_to_swap], child1_genes[key_to_swap]
            elif len(keys) == 1:
                 key_to_swap = keys[0]
                 if key_to_swap in child1_genes and key_to_swap in child2_genes:
                    child1_genes[key_to_swap], child2_genes[key_to_swap] = child2_genes[key_to_swap], child1_genes[key_to_swap]
        return self._ensure_gene_constraints(child1_genes), self._ensure_gene_constraints(child2_genes)

    def _mutate_individual(self, genes: Dict[str, Any]) -> Dict[str, Any]:
        mutated_genes = copy.deepcopy(genes)
        for param_name in mutated_genes:
            if random.random() < self.mutation_rate:
                details = self.gene_schema[param_name]
                current_value = mutated_genes[param_name]
                if details['type'] == 'int':
                    range_width = details['range'][1] - details['range'][0]; sigma = max(1.0, float(range_width) * 0.1) 
                    change = int(random.gauss(0, sigma)); new_val = current_value + change
                    mutated_genes[param_name] = max(details['range'][0], min(new_val, details['range'][1]))
                elif details['type'] == 'float':
                    range_width = details['range'][1] - details['range'][0]; sigma = range_width * 0.1
                    change = random.gauss(0, sigma); new_val = current_value + change
                    mutated_genes[param_name] = max(details['range'][0], min(new_val, details['range'][1]))
                elif details['type'] == 'bool':
                    mutated_genes[param_name] = not current_value
        return self._ensure_gene_constraints(mutated_genes)

    def run_evolution(self) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        logger.info("开始进化过程 (强化验证集筛选)...")
        self.initialize_population()
        self.ga_log = [] 
        self.passed_validation_candidates = [] 
        self.all_validation_attempts_log = [] # 初始化新日志列表

        best_fitness_on_train_overall = -float('inf')
        best_genes_on_train_overall: Optional[Dict[str, Any]] = None 

        for generation in range(self.num_generations):
            self._evaluate_population_on_train() 
            
            current_gen_valid_train_scores_data = [
                (i, score) for i, score in enumerate(self.fitness_scores_on_train) 
                if score is not None and not np.isnan(score) and not np.isinf(score)
            ]
            if not current_gen_valid_train_scores_data:
                 logger.warning(f"第 {generation + 1} 代 - 所有训练集适应度均无效。")
                 if generation < self.num_generations -1 : self.initialize_population()
                 continue
            
            if not current_gen_valid_train_scores_data: 
                logger.warning(f"第 {generation + 1} 代 - 过滤后没有有效的训练适应度分数。")
                continue

            best_item_in_valid_list_with_idx = max(enumerate(current_gen_valid_train_scores_data), key=lambda item_with_idx: item_with_idx[1][1])
            best_train_local_idx_in_valid_list = best_item_in_valid_list_with_idx[0]
            current_best_train_original_idx, current_best_train_fitness_val = \
                current_gen_valid_train_scores_data[best_train_local_idx_in_valid_list]
            current_best_genes_this_gen_on_train = self.population[current_best_train_original_idx]

            if current_best_train_fitness_val > best_fitness_on_train_overall:
                best_fitness_on_train_overall = current_best_train_fitness_val
                best_genes_on_train_overall = copy.deepcopy(current_best_genes_this_gen_on_train)
                logger.info(f"第 {generation + 1} 代：新的训练集最佳总体适应度：{best_fitness_on_train_overall:.4f}")
            
            valid_train_scores_numeric = [score for _, score in current_gen_valid_train_scores_data]
            avg_train_fitness = np.mean(valid_train_scores_numeric) if valid_train_scores_numeric else -float('inf')
            std_train_fitness = np.std(valid_train_scores_numeric) if len(valid_train_scores_numeric) > 1 else 0.0
            
            logger.info(f"第 {generation + 1}/{self.num_generations} 代 (训练集)：最佳适应度={current_best_train_fitness_val:.2f}, "
                        f"平均适应度={avg_train_fitness:.2f}, 标准差={std_train_fitness:.2f}")
            
            candidate_indices_for_validation = [
                i for i, train_fitness in enumerate(self.fitness_scores_on_train)
                if train_fitness is not None and not np.isnan(train_fitness) and not np.isinf(train_fitness) and \
                   train_fitness >= config.MIN_TRAIN_FITNESS_FOR_VALIDATION
            ]
            if not candidate_indices_for_validation and current_gen_valid_train_scores_data:
                if current_best_train_original_idx not in candidate_indices_for_validation:
                    candidate_indices_for_validation.append(current_best_train_original_idx)
            
            logger.debug(f"第 {generation + 1} 代：{len(candidate_indices_for_validation)} 个候选个体将进行验证集评估。")

            for candidate_original_idx in candidate_indices_for_validation:
                candidate_genes = self.population[candidate_original_idx]
                validation_summary = self._evaluate_on_validation(candidate_genes)
                train_fitness_of_candidate = self.fitness_scores_on_train[candidate_original_idx]

                attempt_log_entry = {
                    "generation": generation + 1,
                    "genes": copy.deepcopy(candidate_genes),
                    "train_fitness": float(train_fitness_of_candidate) if train_fitness_of_candidate is not None else None,
                    "validation_summary_attempted": copy.deepcopy(validation_summary) if validation_summary else None,
                    "passed_validation": False 
                }

                if validation_summary is not None:
                    if self._is_validation_result_acceptable(validation_summary):
                        attempt_log_entry["passed_validation"] = True
                        candidate_data_for_pool = {
                            "generation": generation + 1,
                            "genes": copy.deepcopy(candidate_genes),
                            "train_fitness": float(train_fitness_of_candidate) if train_fitness_of_candidate is not None else None,
                            "validation_summary": copy.deepcopy(validation_summary)
                        }
                        self.passed_validation_candidates.append(candidate_data_for_pool)
                        logger.info(f"  第 {generation + 1} 代：基因 {candidate_original_idx} 通过验证。 "
                                    f"训练适应度: {train_fitness_of_candidate:.2f}, "
                                    f"验证 {config.VALIDATION_PRIMARY_METRIC}: "
                                    f"{validation_summary.get(config.VALIDATION_PRIMARY_METRIC, 'N/A'):.4f}")
                self.all_validation_attempts_log.append(attempt_log_entry)

            ga_gen_log = {
                "generation": generation + 1, "best_train_fitness_this_gen": float(current_best_train_fitness_val),
                "avg_train_fitness": float(avg_train_fitness), "std_train_fitness": float(std_train_fitness),
                "best_genes_on_train_this_gen": current_best_genes_this_gen_on_train,
                "num_passed_validation_this_gen": sum(1 for c in self.passed_validation_candidates if c['generation'] == generation + 1)
            }
            self.ga_log.append(ga_gen_log)

            new_population = []
            if self.elitism_count > 0 and self.population and current_gen_valid_train_scores_data:
                sorted_train_indices = sorted(
                    [original_idx for original_idx, _ in current_gen_valid_train_scores_data], 
                    key=lambda idx: self.fitness_scores_on_train[idx], reverse=True
                )
                for idx in sorted_train_indices[:self.elitism_count]:
                    new_population.append(copy.deepcopy(self.population[idx]))
            selected_parents = self._selection() 
            num_offspring_to_create = self.population_size - len(new_population)
            if not selected_parents and num_offspring_to_create > 0:
                for _ in range(num_offspring_to_create): new_population.append(self._create_individual_genes())
            elif selected_parents:
                children_created_count = 0
                while children_created_count < num_offspring_to_create:
                    if not selected_parents : break
                    parent1 = random.choice(selected_parents); parent2 = random.choice(selected_parents) 
                    child1, child2 = self._crossover_individual(parent1, parent2)
                    new_population.append(self._mutate_individual(child1)); children_created_count += 1
                    if children_created_count >= num_offspring_to_create: break
                    new_population.append(self._mutate_individual(child2)); children_created_count += 1
                    if children_created_count >= num_offspring_to_create: break
            self.population = new_population[:self.population_size] 
            while len(self.population) < self.population_size: self.population.append(self._create_individual_genes())

        logger.info("\n--- 进化完成 (所有通过验证的候选者已记录) ---")
        if not self.passed_validation_candidates:
            logger.warning("在整个进化过程中，未能找到任何满足所有验证标准的候选基因。")
        else:
            logger.info(f"共找到 {len(self.passed_validation_candidates)} 个通过验证的候选基因。")

        try:
            with open(GA_LOG_FILE, 'w') as f:
                for entry in self.ga_log:
                    serializable_entry = copy.deepcopy(entry)
                    if 'best_genes_on_train_this_gen' in serializable_entry and isinstance(serializable_entry['best_genes_on_train_this_gen'], dict):
                         for k_g, v_g in serializable_entry['best_genes_on_train_this_gen'].items():
                            if isinstance(v_g, (np.float32, np.float64)): serializable_entry['best_genes_on_train_this_gen'][k_g] = float(v_g)
                            elif isinstance(v_g, (np.int32, np.int64)): serializable_entry['best_genes_on_train_this_gen'][k_g] = int(v_g)
                    f.write(json.dumps(serializable_entry) + '\n')
            logger.info(f"GA 进化日志已保存到 {GA_LOG_FILE}")

            # 保存所有通过验证的候选者（之前叫做 validation_log，现在是 passed_validation_candidates）
            with open(VALIDATION_CANDIDATES_FILE, 'w') as f:
                for entry in self.passed_validation_candidates: # 使用 self.passed_validation_candidates
                    serializable_entry = copy.deepcopy(entry) 
                    if 'genes' in serializable_entry and isinstance(serializable_entry['genes'], dict):
                         for k_g, v_g in serializable_entry['genes'].items():
                            if isinstance(v_g, (np.float32, np.float64)): serializable_entry['genes'][k_g] = float(v_g)
                            elif isinstance(v_g, (np.int32, np.int64)): serializable_entry['genes'][k_g] = int(v_g)
                    if 'validation_summary' in serializable_entry and isinstance(serializable_entry['validation_summary'], dict):
                        for k_s, v_s in serializable_entry['validation_summary'].items():
                            if isinstance(v_s, pd.Timedelta): serializable_entry['validation_summary'][k_s] = str(v_s)
                            elif isinstance(v_s, (np.generic, float, int)): serializable_entry['validation_summary'][k_s] = float(v_s)
                    f.write(json.dumps(serializable_entry) + '\n')
            logger.info(f"所有通过验证的候选基因表现已保存到 {VALIDATION_CANDIDATES_FILE}")

            # 保存所有验证尝试的日志
            with open(ALL_VALIDATION_ATTEMPTS_LOG_FILE, 'w') as f:
                for entry in self.all_validation_attempts_log:
                    serializable_entry = copy.deepcopy(entry)
                    if 'genes' in serializable_entry and isinstance(serializable_entry['genes'], dict):
                         for k_g, v_g in serializable_entry['genes'].items():
                            if isinstance(v_g, (np.float32, np.float64)): serializable_entry['genes'][k_g] = float(v_g)
                            elif isinstance(v_g, (np.int32, np.int64)): serializable_entry['genes'][k_g] = int(v_g)
                    if 'validation_summary_attempted' in serializable_entry and serializable_entry['validation_summary_attempted'] and isinstance(serializable_entry['validation_summary_attempted'], dict):
                        for k_s, v_s in serializable_entry['validation_summary_attempted'].items():
                            if isinstance(v_s, pd.Timedelta): serializable_entry['validation_summary_attempted'][k_s] = str(v_s)
                            elif isinstance(v_s, (np.generic, float, int)): serializable_entry['validation_summary_attempted'][k_s] = float(v_s)
                    f.write(json.dumps(serializable_entry) + '\n')
            logger.info(f"所有验证尝试日志已保存到 {ALL_VALIDATION_ATTEMPTS_LOG_FILE}")

        except Exception as e_log:
            logger.error(f"保存日志时出错: {e_log}", exc_info=True)
            
        return self.passed_validation_candidates, config.VALIDATION_PRIMARY_METRIC

if __name__ == '__main__':
    # --- 用于独立测试 GeneticOptimizerPhase1 的代码块 ---
    print("测试 Genetic Optimizer Phase 1 (独立运行)...")
    
    # 模拟从 DataHandler 获取数据 (你需要确保 OKXConnector 和 DataHandler 可用)
    try:
        from okx_connector import OKXConnector 
        from data_handler import DataHandler   
    except ImportError:
        print("错误：无法导入 OKXConnector 或 DataHandler。请确保它们在 PYTHONPATH 中。")
        exit()

    # 模拟获取历史数据
    print(f"正在为GA测试获取历史数据: {config.SYMBOL} ({config.TIMEFRAME})...")
    try:
        okx_conn_ga_test = OKXConnector() # 这会使用 config.py 中的API密钥设置
        data_hdl_ga_test = DataHandler(okx_conn_ga_test)
        
        historical_df_ga = data_hdl_ga_test.fetch_klines_to_df(
            instId=config.SYMBOL, 
            bar=config.TIMEFRAME,
            total_limit_needed=config.KLINE_LIMIT_FOR_OPTIMIZATION, # 使用优化所需的数据量
            kline_type="history_market" # 或者你用来获取稳定历史数据的方法
        )
    except Exception as e_fetch:
        print(f"获取历史数据时发生错误: {e_fetch}")
        exit()

    # 检查获取的数据是否足够
    # (基于策略中可能用到的最大指标周期来估算最小数据需求)
    max_period_in_schema_main = 0
    # 假设 ATR 和 MA 周期是主要的指标周期需求
    if 'long_ma_period' in config.GENE_SCHEMA:
        max_period_in_schema_main = max(max_period_in_schema_main, config.GENE_SCHEMA['long_ma_period']['range'][1])
    if 'atr_period' in config.GENE_SCHEMA:
        max_period_in_schema_main = max(max_period_in_schema_main, config.GENE_SCHEMA['atr_period']['range'][1])
    
    min_data_needed_for_ga_main = max_period_in_schema_main + 50 # 额外50条作为缓冲
    
    if historical_df_ga.empty or len(historical_df_ga) < min_data_needed_for_ga_main:
        print(f"用于GA测试的数据不足 ({len(historical_df_ga) if not historical_df_ga.empty else 0})。"
              f"至少需要约 {min_data_needed_for_ga_main} 条。正在退出。")
        exit()
    else:
        print(f"已为GA回测获取 {len(historical_df_ga)} 条K线数据。")

    # 创建并运行优化器实例
    optimizer_test_instance = GeneticOptimizerPhase1(historical_df=historical_df_ga)
    best_genes_found, best_fitness_achieved = optimizer_test_instance.run_evolution()

    # 如果找到了最佳基因，用它们运行一次最终回测 (可选)
    if best_genes_found:
        print("\n--- 使用最佳基因运行最终回测 (来自GA独立测试) ---")
        try:
            final_strategy_instance = EvolvableStrategy(best_genes_found)
            final_backtester_instance = Backtester(
                data_df=historical_df_ga, # 在相同数据上回测
                strategy_instance=final_strategy_instance,
                initial_capital=config.INITIAL_CAPITAL,
                fee_rate=config.FEE_RATE
            )
            summary_final, trades_df_final = final_backtester_instance.run_backtest()
            if summary_final:
                final_backtester_instance.print_summary(summary_final)
            if trades_df_final is not None and not trades_df_final.empty:
                 print("\n最终回测交易记录 (前5条):\n", trades_df_final.head())
        except Exception as e_final_bt_standalone:
            print(f"独立测试的最终回测出错: {e_final_bt_standalone}")
            # import traceback; traceback.print_exc() # 取消注释以查看详细错误
    else:
        print("GA独立测试未能产生最佳基因组。")
        
    print("\nGenetic Optimizer Phase 1 独立测试完成。")