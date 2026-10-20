# 验证证据说明

- `original_windows_validation.json`：原离线课程在 Windows x64 / Python 3.12.13 上执行60个脚本和180份Notebook的240条结果；已去掉本机日志路径。该文件记录历史运行，不表示任意操作系统或后续修改均通过。
- `code_equivalence.json`：公开打包时，240个目标的可执行代码与上述原版一致；Notebook中的个人目录输出已匿名化。只比较代码，不把改动说明文字误算为运行验证。
- `publication_windows_validation.json`：公开副本中新建虚拟环境、依赖与素材校验、代表案例重跑的结果。只适用于报告明确列出的检查范围。

新增读者先按安装指南准备自己的环境。要重跑全套，可在根目录执行：

```powershell
.venv/Scripts/python.exe tools/verify_course.py --mode all
```

这会执行60个脚本和180份Notebook，更新Notebook内输出，并在 `reports/` 保存本机日志；可能耗时较长。原练习模板可运行，不代表学生已完成题目。运行器启用 Python socket 防护、空 Hugging Face 缓存和离线加载设置，允许本机 Jupyter 通信；这不等同于物理断网、操作系统防火墙或所有第三方原生组件的网络隔离。

发布前检查运行 `py -3.12 tools/check_publication.py`，它检查待发布文件、案例数量、Python语法和常见个人路径/凭据形状；无法证明不存在所有可能的隐私或安全问题。
