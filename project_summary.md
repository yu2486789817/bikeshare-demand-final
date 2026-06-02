# 项目摘要：共享单车需求预测与调度分析

## 数据与范围

- 骑行数据：Capital Bikeshare 月度 trip history。
- 天气数据：Open-Meteo Historical Weather API。
- 时间范围：2023-01-01 00:00 至 2025-12-31 23:00。
- 小时级样本数：26,304。

## 方法

- 将骑行记录聚合为小时级总需求，并提取会员比例、车辆类型比例、历史滞后需求和滚动均值。
- 按小时融合温度、湿度、降水、风速等天气变量。
- 使用时间切分评估 Seasonal Naive、Ridge、Random Forest、HistGradientBoosting、XGBoost 和 LSTM。
- 增加美国联邦节假日特征、Top 站点聚类和 Top 20 站点净需求预测。

## 结果

- 当前 RMSE 最低模型：hist_gradient_boosting。
- 站点调度建议已导出到 `reports/tables/dispatch_recommendations.csv`。
- 站点聚类画像已导出到 `reports/tables/station_clusters.csv`。
- Top 20 站点预测指标已导出到 `reports/tables/station_model_metrics.csv`。
- 模型指标已导出到 `reports/tables/model_metrics.csv`。
- 报告图表已导出到 `reports/figures/`：daily_demand.html, model_metrics.html, prediction_comparison.html, holiday_demand.html, station_clusters.html, station_model_metrics.html。

## 可展示结论

- 需求存在明显小时周期和工作日/周末差异。
- 天气变量和历史滞后需求共同影响短期需求预测。
- Top 站点存在稳定的净流出或净流入模式，可转化为补车/清车建议。
- 站点聚类可区分通勤流出、通勤流入、景点休闲和混合站点。
