# GitHub仓库提交方案建议

课程要求每名成员有 GitHub 代码提交记录。虽然实际工作中，2名同学主要负责代码类工作，另外2名同学主要负责非代码类工作，但仍可以按照项目计划第 8 节的四个角色，把现有代码拆成 4 个相对独立的提交包。每个提交包对应一名成员的职责边界，并尽量做到文件归属清晰。

目标：每个人都有清晰的技术角色、代码文件和可量化输出，commit 数量与代码量尽量接近。

项目计划的第8节中的团队分工如下：

| 成员   | 学号    | 角色           | 主要职责                               | 可考核输出                             |
| ------ | ------- | -------------- | -------------------------------------- | -------------------------------------- |
| 任泓臻 | 2351458 | 数据工程负责人 | 数据下载、清洗、天气融合、数据质量检查 | 数据处理代码、处理后数据、数据质量说明 |
| 虞澍   | 2352979 | 算法负责人     | 特征工程、传统模型、XGBoost、实验脚本  | 特征构造代码、训练代码、模型指标       |
| 黄彦炜 | 2353117 | 深度学习负责人 | LSTM、GPU/CPU 训练适配、深度模型解释   | LSTM 代码、训练记录、模型对比分析      |
| 马小龙 | 2353814 | 工程展示负责人 | Dashboard、调度建议、图表导出、文档    | Streamlit 页面、调度表、报告图表       |

因此，可以按照下面的划分方式和提交顺序进行提交

## 任泓臻：数据工程与多源融合

建议 commit message：

1. `feat(data): add trip download cleaning and hourly aggregation`
2. `feat(weather): merge Open-Meteo weather and holiday-ready data`
3. `test(data): cover trip cleaning and hourly aggregation`

提交文件：

- `src/bikeshare/data.py`
- `src/bikeshare/config.py`
- `src/bikeshare/pipeline.py`
- `tests/test_data_processing.py`
- `data/`

## 虞澍：特征工程与传统模型

建议 commit message：

1. `feat(features): add temporal lag and rolling demand features`
2. `feat(models): train baseline and tree-based regressors`
3. `feat(experiment): add reproducible model leaderboard script`
4. `test(models): cover scoring and temporal split behavior`

提交文件：

- `src/bikeshare/features.py`
- `src/bikeshare/modeling.py`
- `scripts/`
- `tests/test_features.py`
- `tests/test_modeling.py`
- `models/metrics.json`、`models/predictions.csv`

## 黄彦炜：LSTM 与 GPU 实验

建议 commit message：

1. `feat(lstm): add pytorch sequence model for hourly demand`
2. `feat(lstm): record training device and sequence metadata`
3. `docs(lstm): document gpu cpu training setup`

提交文件：

- `src/bikeshare/lstm_model.py`
- `src/bikeshare/dispatch.py`
- `tests/test_modeling.py`
- `tests/test_station_enhancements.py`
- `models/station_predictions.csv`
- `models/training_info.json`

## 马小龙：Dashboard、调度与报告产物

建议 commit message：

1. `feat(dispatch): add station imbalance recommendations`
2. `feat(dashboard): build Streamlit analysis dashboard`
3. `feat(reporting): export report tables and figures`
4. `docs(project): add run guide and team handoff notes`

主要文件：

- `app/dashboard.py`
- `src/bikeshare/station_clustering.py`
- `src/bikeshare/station_modeling.py`
- `src/bikeshare/reporting.py`
- `docs/`
- `reports/`

## 建议提交顺序

1. 数据工程提交：先提交 `config.py`、`data.py`、`pipeline.py` 和数据处理测试。
2. 算法模型提交：提交 `features.py`、`modeling.py`、实验脚本和模型测试。
3. 深度学习提交：提交 `lstm_model.py`，并补充 `modeling.py` 中 LSTM 接入和 README 的训练说明。
4. 工程展示提交：提交 Dashboard、调度、站点聚类、站点预测、报告导出和文档。

如果需要每个人多次 commit，可以按上面的内容每人分多次提交。
