# 数据来源、署名和改编记录

核验日期：2026-10-20。此目录中的真实数据按原数据许可提供；根目录 MIT 许可不适用于这些数据。下列第一方 / UCI 页面均明确标注 [Creative Commons Attribution 4.0 International（CC BY 4.0）](https://creativecommons.org/licenses/by/4.0/)。再分享时保留相应作者、数据名、来源、许可与改编记录；不要只署名课程作者。

## Jena 日级气象数据

- 文件：`jena_daily.csv`。
- 原数据提供方：**Max Planck Institute for Biogeochemistry（MPI-BGC）**，Jena，WS Beutenberg 气象站。
- 第一方来源与许可：[MPI-BGC Weather Data Download](https://www.bgc-jena.mpg.de/wetter/weather_data.html)。页面同时列出历史年份下载与 CC BY 4.0 条款；不是从 TensorFlow 教程代码许可推定数据许可。
- 下载链路：[TensorFlow 时间序列教程](https://www.tensorflow.org/tutorials/structured_data/time_series)使用由 **François Chollet** 准备的 2009—2016 子集，[原始 ZIP 镜像](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)位于 TensorFlow 官方存储。公开仓库只附日级派生 CSV，不附 ZIP。
- 改编者：**SCUliujiacheng and contributors**。选择 `T (degC)`、`p (mbar)`、`rh (%)`，按原始时间字符串的日历日期分组；温度计算最大值、最小值、均值，气压与相对湿度计算均值；丢弃少于 100 条观测的日期。没有另外进行时区转换。
- 输出字段：`date`（日期）、`temp_max` / `temp_min` / `temp_mean`（°C）、`pressure`（mbar）、`humidity`（%）。课程的明日最高温预测任务是新增教学设计，不是数据提供方发布的预测产品。

建议署名：MPI-BGC, Jena WS Beutenberg weather observations, 2009–2016; CC BY 4.0. Subset prepared by François Chollet and distributed through the TensorFlow tutorial. Daily aggregation and observation-count filtering by SCUliujiacheng and contributors. 同时保留上述来源与许可链接。

## SMS Spam Collection

- 文件：`sms.tsv`、`sms_README.txt`。
- 作者及原始版权人：**Tiago Agostinho de Almeida、José María Gómez Hidalgo**。
- 数据引用：Almeida, T. & Hidalgo, J.（2011），*SMS Spam Collection*，UCI Machine Learning Repository，[DOI: 10.24432/C5CC84](https://doi.org/10.24432/C5CC84)。[UCI 来源及 CC BY 4.0 声明](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)。
- 改动：给原始 `SMSSpamCollection` 添加 `label` 和 `text` 两列表头，并保存为 `sms.tsv`；标签与短信正文保留。原始 `readme` 以原字节保存为 `sms_README.txt`，其中保留更详细的语料来源、版权和免责声明。其旧字符编码在部分阅读器中可能显示不完整，本文件提供作者姓名的 Unicode 写法。
- 研究引用：Almeida, T. A.; Gómez Hidalgo, J. M.; Yamakami, A.（2011），*Contributions to the Study of SMS Spam Filtering: New Collection and Results*，ACM Symposium on Document Engineering。
- 原始下载：[UCI ZIP](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip)。ZIP 不在仓库中；测试成绩仅针对该数据及案例划分，不保证中文短信表现。

## Bike Sharing

- 文件：`bike_hour.csv`、`bike_day.csv`、`bike_README.txt`。
- 数据作者：**Hadi Fanaee-T**；记录来自 2011—2012 年 Capital Bikeshare，并附相关天气与日历字段。
- 数据引用：Fanaee-T, H.（2013），*Bike Sharing*，UCI Machine Learning Repository，[DOI: 10.24432/C5W894](https://doi.org/10.24432/C5W894)。[UCI 来源及 CC BY 4.0 声明](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)。
- 改动：将 `hour.csv`、`day.csv`、`Readme.txt` 分别重命名为 `bike_hour.csv`、`bike_day.csv`、`bike_README.txt`；文件内容未改动。各案例后续筛选、划分及特征处理在代码中明确实施。
- 依原 README 的引用要求，出版物还应引用：**Hadi Fanaee-T、João Gama**，*Event labeling combining ensemble detectors and background knowledge*（2013），[DOI: 10.1007/s13748-013-0040-3](https://doi.org/10.1007/s13748-013-0040-3)。
- 原始下载：[UCI ZIP](https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip)。ZIP 不在仓库中。预测 `cnt` 时不能用同一时点构成 `cnt` 的 `casual` / `registered` 作为可提前获知的预测特征。

## 由 scikit-learn 读取的其他真实数据

这些数据通过已安装的 scikit-learn 读取，不在此目录另存完整副本。原始样本与衍生图表的署名见 [根目录第三方记录](../../THIRD_PARTY_NOTICES.md)，包括：

| 数据集 | 作者与年份 | UCI DOI / 许可 |
|---|---|---|
| Optical Recognition of Handwritten Digits | E. Alpaydin、C. Kaynak，1998 | [10.24432/C50P49](https://doi.org/10.24432/C50P49)，CC BY 4.0 |
| Breast Cancer Wisconsin (Diagnostic) | 按 UCI 引用记录：W. Wolberg、O. Mangasarian、N. Street、W. Street，1993 | [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B)，CC BY 4.0 |
| Wine | S. Aeberhard、M. Forina，1992 | [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)，CC BY 4.0 |

模拟数据会在对应案例中明确标注，其生成规则、参数和随机种子见代码。准备依赖及离线运行的方法见 [安装指南](../../安装指南.md)与[离线准备指南](../../离线准备指南.md)。
