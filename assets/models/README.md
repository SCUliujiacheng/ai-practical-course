# 模型来源、固定版本与下载

本公开仓库不上传预训练权重或课程训练后的权重。此目录保留 `SOURCE.json` 来源记录，以及已有的上游许可文件；实际权重、配置和分词器文件需在首次准备时从上游下载。准备完成后，案例可从本地读取。依赖安装见 [安装指南](../../安装指南.md)，完整离线准备步骤见 [离线准备指南](../../离线准备指南.md)。

## 准备入口

在已安装课程依赖的仓库根目录执行以下联网准备命令（Windows）：

```powershell
.venv/Scripts/python.exe tools/prepare_models.py --all
```

脚本入口为 [tools/prepare_models.py](../../tools/prepare_models.py)。不要为离线复现擅自把固定版本改成上游最新版本；更换版本后需重新记录来源、哈希并重新验收相关案例。仓库根目录 `model_files.json` 记录本课程所用文件的预期路径、大小与 SHA-256；它列出了需要准备的文件，不表示 Git 中包含权重。

## 已核验的来源

核验日期：2026-10-20。下列四个 Hugging Face 模型仓库的模型卡均标记 `apache-2.0`；固定 revision 同本地 `SOURCE.json` 一致。许可原文：[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)。

| 本地目录 | 上游仓库与作者组织 | 课程使用的 revision |
|---|---|---|
| `minilm` | Sentence Transformers：[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `flan-t5-small` | Google：[flan-t5-small](https://huggingface.co/google/flan-t5-small) | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` |
| `qwen3-0.6b` | Qwen：[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` |
| `chronos-bolt-tiny` | Amazon：[chronos-bolt-tiny](https://huggingface.co/amazon/chronos-bolt-tiny) | `a0e552de83495b5c28c14c71c374f3e33280b340` |

ResNet18 使用 [PyTorch 官方权重文件 `resnet18-f37072fd.pth`](https://download.pytorch.org/models/resnet18-f37072fd.pth)，准备后保存为 `assets/models/resnet18.pth`；架构与权重说明见 [torchvision 官方 ResNet18 文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。本课程没有把 torchvision 源代码的 BSD 许可推定为全部权重或训练图像的再分发许可；公开仓库不提供该权重副本。

## 元数据的含义与许可边界

- `SOURCE.json` 中的 `files` 是制作时的下载清单，`download_date` 是记录的下载日期；公开仓库只保留其中的元数据和必要许可文件。
- 下载后应保留上游 README / 模型卡、LICENSE 和 NOTICE（如提供）。模型具体条件以上游对应版本的声明为准，本课程的 MIT 与 CC BY 4.0 不替代模型许可。
- 上述许可标记是已核验的发布方声明，不是对未公开训练材料、所有训练数据或所有生成内容的权利保证，也不推定未来版本适用相同条件。
- 课程中的训练、提示词设置和推理结果是教学实验；使用现有权重不表示知道完整训练数据，公开测试数据可能与预训练语料重叠。

更多数据、依赖与原创材料许可见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
