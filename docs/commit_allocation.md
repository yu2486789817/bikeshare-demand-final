# 四人提交拆分建议

目标：每个人都有清晰的技术角色、代码文件和可量化输出，commit 数量与代码量尽量接近。

## 成员 A：数据工程与多源融合

建议 commit：

1. `feat(data): add Capital Bikeshare download and trip cleaning`
2. `feat(data): aggregate city and station hourly demand`
3. `feat(weather): merge Open-Meteo hourly weather features`
4. `test(data): cover mixed timestamp parsing and hourly gaps`

主要文件：

- `src/bikeshare/data.py`
- `src/bikeshare/config.py`
- `tests/test_data_processing.py`

## 成员 B：特征工程与传统模型

建议 commit：

1. `feat(features): add temporal lag and rolling demand features`
2. `feat(models): train baseline and tree-based regressors`
3. `feat(experiment): add reproducible model leaderboard script`
4. `test(models): cover scoring and temporal split behavior`

主要文件：

- `src/bikeshare/features.py`
- `src/bikeshare/modeling.py`
- `scripts/run_experiment.py`
- `tests/test_features.py`
- `tests/test_modeling.py`

## 成员 C：LSTM 与 GPU 实验

建议 commit：

1. `feat(lstm): add PyTorch sequence model for hourly demand`
2. `fix(lstm): scale regression target for stable training`
3. `chore(gpu): document CUDA wheelhouse setup`
4. `test(lstm): cover sequence window construction`

主要文件：

- `src/bikeshare/lstm_model.py`
- `src/bikeshare/modeling.py`
- `README.md`
- `tests/test_modeling.py`

## 成员 D：Dashboard、调度与报告产物

建议 commit：

1. `feat(dispatch): add station imbalance recommendations`
2. `feat(dashboard): build Streamlit analysis dashboard`
3. `feat(reporting): export report tables and figures`
4. `docs(project): add run guide and team handoff notes`

主要文件：

- `app/dashboard.py`
- `src/bikeshare/dispatch.py`
- `src/bikeshare/reporting.py`
- `README.md`
- `docs/commit_allocation.md`

## 建议实际提交顺序

1. A 先提交数据管道。
2. B 基于处理后数据提交特征与传统模型。
3. C 在模型接口稳定后提交 LSTM/GPU。
4. D 最后提交 Dashboard、报告图表和文档。

如果需要每个人多次 commit，可以按上面的列表每人拆成 3-4 个 commit；如果只需要大致均衡，每人保留 2 个较大的 commit 也可以。
