# 第三方来源、数据与许可记录

核验日期：2026-10-20。课程中的算法是公开方法的原创教学实现与示例改编，不代表任何机构已经部署的项目。各案例的具体研究与实现参考见其 `来源与延伸阅读.md`。

## 原创材料与第三方材料的边界

原创作者署名：**SCUliujiacheng and contributors**。

- 原创 Python 脚本、Notebook 代码单元和文档中的原创代码示例采用 [MIT License](LICENSE)。
- 原创教学文字、Notebook Markdown 单元、原创插图与图表设计采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，完整声明与许可原文见 [LICENSE-DOCS](LICENSE-DOCS)。HTML 导出中的各部分沿用对应源文件许可。
- 图表使用的第三方数据、Notebook 中展示的第三方样本、引文、第三方许可文本和模型文件保留原有权利与许可。课程的 MIT / CC BY 声明不改变它们的许可，也不表示原作者认可本课程。

## 随仓库提供的真实数据

以下许可已在数据提供方页面核验。CC BY 4.0 要求合理署名、保留来源与许可链接、标明修改，并保留所提供的相关声明。文件级记录及完整署名建议见 [assets/data/README.md](assets/data/README.md)。

| 本地文件 | 原作者、来源与许可 | 本课程的处理 |
|---|---|---|
| `assets/data/jena_daily.csv` | Max Planck Institute for Biogeochemistry（MPI-BGC），Jena / WS Beutenberg 气象观测；[第一方下载页明确标注 CC BY 4.0](https://www.bgc-jena.mpg.de/wetter/weather_data.html)。使用 [TensorFlow 教程](https://www.tensorflow.org/tutorials/structured_data/time_series)提供的 2009—2016 子集，该子集由 François Chollet 准备。 | 选择温度、气压、湿度；按日期计算温度最大值、最小值、平均值及气压、湿度均值；去掉少于 100 条观测的日期。公开仓库不附原始 ZIP。 |
| `assets/data/sms.tsv`、`sms_README.txt` | Tiago Agostinho de Almeida、José María Gómez Hidalgo；[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)，2011；[DOI: 10.24432/C5CC84](https://doi.org/10.24432/C5CC84)；UCI 标注 CC BY 4.0。 | 原始标签与英文文本保留；添加 `label` / `text` 表头。原始 README 的版权、来源和免责声明保留，原始 ZIP 不附在仓库中。 |
| `assets/data/bike_hour.csv`、`bike_day.csv`、`bike_README.txt` | Hadi Fanaee-T；[UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)，2013；[DOI: 10.24432/C5W894](https://doi.org/10.24432/C5W894)；UCI 标注 CC BY 4.0。 | `hour.csv`、`day.csv` 分别改名为 `bike_hour.csv`、`bike_day.csv`，内容未改动；保留原始 README。原始 ZIP 不附在仓库中。 |

短信数据还应引用 Almeida、Gómez Hidalgo、Yamakami 的 2011 年论文 *Contributions to the Study of SMS Spam Filtering: New Collection and Results*；自行车数据原始 README 要求在出版物中引用 Fanaee-T 与 Gama 的论文 *Event labeling combining ensemble detectors and background knowledge*（2013），[DOI: 10.1007/s13748-013-0040-3](https://doi.org/10.1007/s13748-013-0040-3)。

课程通过已安装的 scikit-learn 读取下列随包数据。仓库不复制 scikit-learn 安装包；案例输出中出现的数据示例和数字图像仍应保留底层数据署名。

| 数据集 | UCI 作者、年份与 DOI | 许可与使用说明 |
|---|---|---|
| [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) | E. Alpaydin、C. Kaynak，1998；[10.24432/C50P49](https://doi.org/10.24432/C50P49) | CC BY 4.0；通过 `load_digits` 读取 scikit-learn 所提供的子集，案例中的缩放、划分及图像变换见代码。 |
| [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 按 UCI 当前引用记录：W. Wolberg、O. Mangasarian、N. Street、W. Street，1993；[10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) | CC BY 4.0；通过 `load_breast_cancer` 读取；教学分类结果不构成临床验证。 |
| [Wine](https://archive.ics.uci.edu/dataset/109/wine) | S. Aeberhard、M. Forina，1992；[10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J) | CC BY 4.0；通过 `load_wine` 读取；特征预处理及训练/测试划分见案例代码。 |

其他房屋、用电、客流、故障、收益、成本、轨道和团队数据，如无另行来源声明，均为教学模拟；生成机制、参数与随机种子见相应案例。不得把它们解释成真实行业调查结果。

## 预训练模型：只发布元数据与下载入口

本公开仓库**不包含预训练权重或训练后权重**。模型目录保留来源记录和已有许可文件；`SOURCE.json` 的 `files` 是原下载清单，不表示 Git 中包含这些文件。下载方式、固定版本与边界见 [assets/models/README.md](assets/models/README.md)。

| 上游模型 | 核验到的上游声明 | 固定版本记录 |
|---|---|---|
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 模型卡标记 Apache-2.0 | `assets/models/minilm/SOURCE.json` |
| [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) | 模型卡标记 Apache-2.0 | `assets/models/flan-t5-small/SOURCE.json` |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 模型卡标记 Apache-2.0；保留上游 LICENSE | `assets/models/qwen3-0.6b/SOURCE.json` |
| [amazon/chronos-bolt-tiny](https://huggingface.co/amazon/chronos-bolt-tiny) | 模型卡标记 Apache-2.0 | `assets/models/chronos-bolt-tiny/SOURCE.json` |
| [PyTorch ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) | 权重来自 PyTorch 官方下载地址；未据 torchvision 代码的 BSD 许可推定权重与全部训练图像的再分发条件 | `resnet18-f37072fd.pth`；下载目标为 `assets/models/resnet18.pth` |

[Apache License 2.0 原文](https://www.apache.org/licenses/LICENSE-2.0)与各模型的具体文件及声明应一并查阅。上述记录描述已核验的上游标记，不保证未公开训练材料的权利状态，也不改变未来上游版本的条件。下载后保留上游许可、模型卡及 NOTICE（如提供）；本课程不对模型作额外再许可。数据与预训练语料可能重叠，案例成绩不等于保证未见数据上的效果。

## Python 与第三方软件

公开仓库提供依赖清单与准备脚本，不附 `.venv`、Python 运行时、`wheelhouse`、缓存或原始下载 ZIP。安装时获得的 Python 与第三方库保留各自许可及版权声明；`requirements-lock.txt` 记录验收环境的精确版本，不是许可授予文件。使用、修改或重新打包这些软件时，应查阅其随包许可证。每例引用的外部论文、文档和实现链接也不因此改为本课程许可。
