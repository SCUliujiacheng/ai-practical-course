# 离线可学：深度强化学习是否值得升级

先完成主例。此拓展依赖的 Gymnasium、Stable-Baselines3 与 CPU PyTorch 已纳入课程环境，不需要额外数据、权重或网络。

在本目录运行：`../../.venv/Scripts/python.exe extension_dqn.py`。也可在课程的本机 Jupyter 终端里执行同一命令。实际验收结果见根目录 `验收报告.md`。

扩展把路线从5段改为20段，观察归一化位置。DQN用小型神经网络表示Q值，训练2500步，使用3个独立训练种子和同一组独立评价种子。

1. 阅读CourierEnv，确认奖励、终止与截断条件；`check_env`检查接口，不证明问题建模正确。
2. 查看`dqn_metrics.json`的训练时间、样本量、完成率和配送耗时。
3. 和同一20段环境中的固定慢路、已知模型参照比较。已知模型参照拥有更多信息，须解释这一点。
4. 将2500步改为10000步，先预测成本和效果，再运行。效果没有提高也应保留结果。

这个小问题仍可用表格方法处理。拓展是学习标准环境接口和升级成本，不能据此宣称深度模型是必要的或更优。

来源：[SB3 DQN 文档](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)、[Gymnasium 自定义环境](https://gymnasium.farama.org/introduction/create_custom_env/)。无需联网也能读本地代码和本说明。
