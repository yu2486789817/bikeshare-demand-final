# 团队分工声明

本文档用于课程期末项目提交时说明四名成员的具体贡献。最终提交前，请每位成员确认自己负责的代码文件、报告章节、展示材料和工作量百分比。

## 1. 项目基本信息

- 项目名称：共享单车需求预测与调度分析：基于骑行记录与历史天气的多源数据挖掘
- 课程名称：数据挖掘与分析
- 项目类型：期末项目，C类，端到端应用系统开发
- 团队成员：
  - 任泓臻（2351458）
  - 虞澍（2352979）
  - 黄彦炜（2353117）
  - 马小龙（2353814）

## 2. 总体分工

本项目按照项目计划书中的分工展开，共分为数据工程、算法建模、深度学习和工程展示四个模块。代码模块与报告章节尽量保持一致，使每名成员都有明确的可考核输出。

| 成员 | 角色 | 工作量占比 | 签字确认 |
|---|---|:---|----|
| 任泓臻 | 数据工程负责人 | 25% |  |
| 虞澍 | 算法负责人 | 25% |  |
| 黄彦炜 | 深度学习负责人 | 25% |  |
| 马小龙 | 工程展示负责人 | 25% |  |

## 3. 成员具体贡献

### 3.1 任泓臻（2351458）：数据工程负责人

主要职责：

- 确认 Capital Bikeshare 和 Open-Meteo 数据源。
- 实现骑行数据下载、读取和清洗流程。
- 完成时间字段解析、异常骑行时长过滤、缺失站点处理。
- 将骑行记录聚合为城市级小时需求和站点级小时需求。
- 融合历史天气数据，并生成处理后数据表。
- 维护项目配置、目录结构和主 pipeline。

负责代码文件：

- `src/bikeshare/config.py`
- `src/bikeshare/data.py`
- `src/bikeshare/pipeline.py`
- `tests/test_data_processing.py`

负责结果文件：

- `data/processed/hourly_demand.csv`
- `data/processed/weather_hourly.csv`
- `data/processed/station_hourly.csv`

负责报告章节：

- 第 2 章：数据来源与预处理。
- 第 3.1 到 3.3 节：时间、节假日、天气和结构特征说明。
- 附录中的数据字段说明和运行流程。

GitHub 提交记录：

- `feat(data): add trip download cleaning and hourly aggregation`
- `feat(weather): merge Open-Meteo weather and holiday-ready data`
- `test(data): cover trip cleaning and hourly aggregation`

### 3.2 虞澍（2352979）：算法负责人

主要职责：

- 设计城市总体需求预测任务。
- 构造时间、周期、节假日、滞后和滚动窗口特征。
- 实现 Seasonal Naive、Ridge、Random Forest、HistGradientBoosting 和 XGBoost 模型。
- 使用时间顺序切分进行模型训练和测试。
- 输出统一格式的指标表和预测结果。
- 编写可复现实验脚本。

负责代码文件：

- `src/bikeshare/features.py`
- `src/bikeshare/modeling.py`
- `scripts/run_experiment.py`
- `tests/test_features.py`
- `tests/test_modeling.py`

负责结果文件：

- `data/processed/model_features.csv`
- `models/metrics.json`
- `models/predictions.csv`
- `reports/tables/model_metrics.csv`

负责报告章节：

- 第 3 章：特征工程与任务构建。
- 第 4.1 节：城市总体需求预测模型。
- 第 5 章：实验设计与评价指标。
- 第 6.3 节：城市级模型结果。

GitHub 提交记录：

- `feat(features): add temporal holiday lag and rolling features`
- `feat(models): train baseline ridge tree and xgboost models`
- `feat(experiment): add reproducible leaderboard script`
- `test(models): cover feature columns splits and metrics`

### 3.3 黄彦炜（2353117）：深度学习负责人

主要职责：

- 实现或整理 LSTM 时序模型实验。
- 构造过去 24 小时序列输入。
- 适配 GPU/CPU 训练环境，并记录训练设备。
- 分析 LSTM 与传统机器学习模型的表现差异。
- 在报告中解释深度模型在本任务中的作用和局限。

负责代码文件：

- `src/bikeshare/lstm_model.py`
- `src/bikeshare/modeling.py` 中 LSTM 接入部分。
- `README.md` 中 LSTM 与 GPU 说明部分。
- `tests/test_modeling.py` 中序列构造相关测试。

负责结果文件：

- `models/lstm.joblib`
- `models/training_info.json`
- `models/predictions.csv` 中 `lstm` 预测列。

负责报告章节：

- 第 4.2 节：LSTM 时序建模。
- 第 6.3 节中 LSTM 与其他模型的对比分析。
- 第 7.1 节：模型表现讨论。
- 第 7.4 节：深度模型局限性。

GitHub 提交记录：

- `feat(lstm): add pytorch sequence model for hourly demand`
- `feat(lstm): record training device and sequence metadata`
- `docs(lstm): document gpu cpu training setup`

### 3.4 马小龙（2353814）：工程展示负责人

主要职责：

- 实现站点调度建议、站点聚类和 Top 20 站点预测展示。
- 开发 Streamlit Dashboard。
- 整理模型对比、站点详情、热力图和项目说明页面。
- 导出报告图表和展示材料。
- 负责最终演示系统、海报和答辩材料的视觉组织。

负责代码文件：

- `app/dashboard.py`
- `src/bikeshare/dispatch.py`
- `src/bikeshare/station_clustering.py`
- `src/bikeshare/station_modeling.py`
- `src/bikeshare/reporting.py`
- `README.md`
- `docs/*.md`
- `reports/poster/*.html`

负责结果文件：

- `reports/tables/dispatch_recommendations.csv`
- `reports/tables/station_clusters.csv`
- `reports/tables/station_model_metrics.csv`
- `models/station_predictions.csv`
- `reports/figures/*.html`
- `output/pdf/共享单车项目海报_A1.pdf`

负责报告章节：

- 第 4.4 节：站点聚类与调度建议。
- 第 4.5 节：Dashboard 系统设计。
- 第 6.4 节：站点级模型结果。
- 第 6.5 节：调度建议与站点聚类结果。
- 附录中的 Dashboard 截图和展示说明。

GitHub 提交记录：

- `feat(dispatch): add station imbalance recommendations`
- `feat(stations): add station clustering and net demand models`
- `feat(dashboard): build streamlit dashboard pages`
- `feat(reporting): export report tables figures and posters`
- `docs(project): add run guide contribution and submission docs`
