# 🚀 基于"基因锁"概念的OKX永续合约交易程序 🧬📈

**🎯 核心目标：** 利用进化算法（遗传算法GA、遗传编程GP等）自动发现和优化OKX永续合约（如BTC-USDT-SWAP）的交易策略，实现更智能、更自适应的量化交易。

---

## 🌟 项目当前状态与进展

**当前项目阶段：阶段一成果显著，准备迈向阶段二！**

### ✅ 阶段一：参数优化 + 简单规则组合进化 (已取得关键成果)

*   **主要成就** 🎉:
    *   **稳健的GA框架和回测引擎** 💪：已成功搭建并验证了包含数据获取、API连接、指标计算、并行回测、GA优化器（支持参数进化、训练/验证集划分）的完整系统。
    *   **可进化的策略模板** 📝：实现了基于均线交叉和ATR动态止损/止盈的 `EvolvableStrategy` 模板，并加入了严格的基因约束（如均线差、TP/SL风险回报比）。
    *   **问题攻克** 🛠️：通过多轮迭代，有效解决了API超时、连接不稳定、基因映射错误、GA运行速度慢等多个棘手问题。
    *   **高质量策略产出** ✨：最新的GA运行已能稳定产出在优化期表现优异的策略，并通过严格的OOS（样本外）测试，验证了部分策略（如基于Run 4和Run 8基因的策略）具有出色的稳健性和盈利能力！例如，Run 4策略在OOS期间表现出高达92%的PnL和2.6的盈利因子，与优化期高度一致！
    *   **可靠的评估工具** 📏：`Backtester.py` 经过改进，现在能够对相同的交易行为产出稳定一致的回测摘要，并新增了如SQN、Calmar Ratio等多个实用指标。

*   **脚本文件清单 (核心)**:
    *   `config.py` ⚙️: 所有可配置参数的中央枢纽。
    *   `okx_connector.py` 🔗: OKX API的坚实桥梁，已加入强大的超时重试机制。
    *   `data_handler.py` 📊: 可靠的K线数据获取与处理模块。
    *   `indicators.py` 📉: 常用的技术指标计算器。
    *   `strategy.py` (EvolvableStrategy) 🧠: 可进化的策略模板核心。
    *   `backtester.py` ⏱️: 经过强化的回测引擎，结果稳定可靠。
    *   `genetic_optimizer_phase1.py` 🧬: GA参数优化的核心实现，支持并行计算。
    *   `main.py` ▶️: 程序主入口，统领全局。
    *   `run_multiple_ga_optimizer_sessions.py` 🚀🚀: 批量运行GA优化，效率倍增。
    *   `run_selected_oos_tests.py` 🧪: 严格的OOS测试助手。
    *   `best_genes.json` 🏆: 保存每一轮GA选出的最佳基因。
    *   各种日志文件 📜: 记录实验过程的点点滴滴。

### ⏳ 下一步：阶段二 - 引入遗传编程 (GP) - 蓄势待发！

*   **目标**: 使用GP直接进化出策略的规则结构（买卖条件、运算逻辑、指标组合方式等），力求发现更创新、更强大的交易逻辑，不再局限于预设的均线交叉模板。
*   **当前状态**: 已完成阶段一的成果验证，目前AI助手正在积极进行GP阶段的详细设计，包括：
    *   GP库的选择 (`gplearn` 初步选定)。
    *   函数集 (算术、逻辑、比较运算等) 和终端集 (价格数据、技术指标、可进化常数等) 的具体定义。
    *   GP树如何转化为交易信号的逻辑。
    *   包含树复杂度惩罚和训练/验证一致性的GP适应度函数设计。
    *   GP优化器与现有框架的整合方案。
*   **敬请期待** ✨: 一个能够自主“编写”交易策略的AI即将登场！

### 🔭 远期规划：

*   **阶段三：多策略组合与元学习** 🤝 集群智慧，更上一层楼！
*   **阶段四：探索“策略调控网络”** 🕸️ 迈向策略生态的深度理解！

---

## 🛠️ 项目脚本使用指南：轻松上手你的OKX基因锁交易程序！

本指南将帮助你了解如何使用本项目中的核心脚本来优化、测试和（未来）运行你的自动化交易策略。

### ⚠️ 重要前提：

1.  **环境配置** 🌲: 确保你已经安装了所有必要的Python库（请参考 `requirements.txt` 文件，如果项目中有的话，或者根据脚本中的 `import` 语句安装）。
2.  **API密钥配置** 🔑:
    *   打开 `config.py` 文件。
    *   如果你想使用OKX模拟盘进行测试和优化（强烈推荐！），请确保 `IS_DEMO_TRADING = True`，并正确填写 `DEMO_API_KEY`, `DEMO_SECRET_KEY`, 和 `DEMO_PASSPHRASE`。
    *   如果准备进行实盘交易（风险自负！），请设置 `IS_DEMO_TRADING = False`，并确保已配置真实的API密钥（目前脚本中这部分可能还是占位符，你需要自行添加并安全管理）。
3.  **其他配置检查** 🧐: 根据你的需求，检查 `config.py` 中的其他参数，如交易对 `SYMBOL` (例如 "BTC-USDT-SWAP")、K线时间周期 `TIMEFRAME` (例如 "1H")、GA参数、回测初始资金等。

---

### ✨ 主要使用流程与脚本功能：

**Step 1: 🚀 运行遗传算法 (GA) 优化策略参数 (推荐流程)**

*   **脚本**: `run_multiple_ga_optimizer_sessions.py`
*   **目的**: 这是最核心的步骤！此脚本会自动多次运行 `main.py` 的优化模式，以寻找最佳的策略基因参数。它会处理结果的保存和汇总。
*   **用法**:
    ```bash
    python run_multiple_ga_optimizer_sessions.py
    ```
*   **参数设置**:
    *   打开 `run_multiple_ga_optimizer_sessions.py` 文件，可以修改顶部的 `NUMBER_OF_RUNS` 来决定运行多少次独立的GA优化会话（默认为10次）。
    *   GA相关的参数（如种群大小、代数、变异率等）在 `config.py` 中设置。
*   **产出**:
    *   会在 `ga_multiple_runs_results/` 目录下创建一个带有时间戳的新子目录（例如 `ga_runs_YYYYMMDD_HHMMSS/`）。
    *   该子目录中包含：
        *   `all_runs_summary.csv`: 所有GA优化会话的结果摘要，非常重要！👍
        *   每个独立运行会话的详细日志和产出文件（如 `run_X_optimize_output.txt`, `best_genes.json`, `ga_evolution_log.jsonl` 等）。

**Step 2: 📊 分析GA优化结果**

*   **目的**: 从多次GA运行中挑选出表现最佳、最稳健的策略基因。
*   **操作**:
    *   打开上一步生成的 `all_runs_summary.csv` 文件。
    *   关注关键指标，如 `Total PnL Pct` (总盈亏百分比), `Max Drawdown` (最大回撤), `Sharpe Ratio` (夏普比率), `Profit Factor` (盈利因子), `Total Trades` (总交易次数), 以及 `status` (确保是 "Success")。
    *   选择那些在这些指标上综合表现优异的 `run_id`。

**Step 3: 🧪 对选定的策略进行严格的样本外 (OOS) 测试**

*   **脚本**: `run_selected_oos_tests.py`
*   **目的**: 验证在优化期表现良好的策略，在它们从未见过的新数据（OOS数据）上的真实表现和稳健性。
*   **用法**:
    1.  打开 `run_selected_oos_tests.py` 文件。
    2.  **修改 `SUMMARY_CSV_FILE`**: 将其路径设置为你在 Step 1 中生成的 `all_runs_summary.csv` 文件的**完整路径**。
    3.  **修改 `RUN_IDS_TO_TEST_OOS`**: 这是一个列表，填入你在 Step 2 中挑选出的优秀 `run_id`。例如 `RUN_IDS_TO_TEST_OOS = [4, 8, 2]`。
    4.  保存文件。
    5.  运行脚本：
        ```bash
        python run_selected_oos_tests.py
        ```
*   **产出**:
    *   会在 `oos_test_results_latest_ga/` (或你在脚本中配置的目录) 下为每个测试的 `run_id` 创建一个子目录。
    *   每个子目录中包含：
        *   `oos_output_run_X.txt`: OOS回测的详细日志和回测摘要。
        *   `oos_kline_data_run_X.csv`: 用于该OOS测试的K线数据。
        *   `oos_trades_log_run_X.csv`: OOS测试期间的所有交易记录。
        *   `best_genes_run_X.json`: 该次OOS测试使用的策略基因。

**Step 4: 🧐 分析OOS测试结果，做出决策**

*   **目的**: 对比策略在优化期和OOS期的表现，评估其真实盈利能力和泛化能力。
*   **操作**:
    *   查看 `oos_output_run_X.txt` 中的回测摘要。
    *   对比其OOS表现（PnL, MaxDD, PF, Sharpe等）与该策略在 `all_runs_summary.csv` 中的优化期表现。
    *   如果OOS表现依然强劲（例如，Run 4策略的最新结果！），那么恭喜你，找到了一个有潜力的稳健策略！🎉
    *   如果OOS表现大幅下滑，则可能需要回到 `config.py` 调整GA的适应度函数权重、验证标准，或重新审视策略逻辑。

---

### ⚙️ 其他辅助模式 (通过 `main.py` 直接调用)：

虽然推荐使用上面的流程脚本，但 `main.py` 也支持直接运行特定模式：

*   **单独运行一次GA优化**:
    ```bash
    python main.py optimize
    ```
    这会进行一次完整的GA优化，并将最佳基因（如果找到）保存到项目根目录下的 `best_genes.json`，同时输出日志到终端。

*   **对当前 `best_genes.json` 在优化期数据上进行回测**:
    ```bash
    python main.py backtest_best
    ```
    这会加载根目录的 `best_genes.json`，并在完整的优化期数据集 (由 `config.py` 中的 `KLINE_LIMIT_FOR_OPTIMIZATION` 定义) 上进行回测。

*   **对当前 `best_genes.json` 进行一次OOS回测**:
    ```bash
    python main.py backtest_oos
    ```
    这会加载根目录的 `best_genes.json`，并获取用于OOS测试的数据 (由 `config.py` 中的 `KLINE_LIMIT_FOR_OOS` 定义，数据会从优化期数据之前的时间段获取) 进行回测。结果会保存到根目录的 `oos_kline_data.csv` 和 `oos_trades_log.csv`。

*   **（概念性）使用 `best_genes.json` 运行实时交易循环**:
    ```bash
    python main.py live_best
    ```
    目前这部分功能更多是概念性的，实际的实时交易下单、仓位管理、错误处理等需要更完善的实现。

---

### 💡 建议使用流程小结：

1.  **配置好 `config.py`** (特别是API密钥和 `IS_DEMO_TRADING`)。
2.  运行 `python run_multiple_ga_optimizer_sessions.py` 进行多轮GA优化。
3.  分析 `all_runs_summary.csv`，挑选出优胜者。
4.  配置并运行 `python run_selected_oos_tests.py` 对优胜者进行OOS测试。
5.  仔细评估OOS结果，决定下一步是迭代优化GA参数/逻辑，还是准备将优秀策略投入（模拟）实盘观察，或是像我们现在这样，准备进入更高级的GP阶段！

---

**👨‍💻 贡献与参与：**

欢迎对量化交易、进化算法、OKX平台感兴趣的朋友关注、讨论和贡献！让我们一起探索AI在金融交易领域的无限可能！🌟
