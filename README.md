# LLM4CSI-RS
面向室内非视距场景的可移动 5G 单站指纹定位方法，基于 Llama-3.2-3B 预训练大模型，通过 LoRA 参数高效微调实现 RSRP 时序指纹定位，配套真实场景采集的实验数据集与训练代码。
# 环境依赖
- Python 3.10+
- PyTorch 2.1+
- transformers 4.35+
- peft 0.7+（LoRA 微调）
- numpy, pandas
- 基座模型：[Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)（需自行下载或通过 Hugging Face 加载）
# 数据集说明
本仓库提供真实室内场景采集的 5G CSI-RS 指纹数据集，数据通过搭载 USRP B210 的 AGV 移动平台采集，对应论文实验设置：

- 采集场景：300㎡室内大厅（含桌椅、墙体等 NLOS 遮挡）
- 信号频段：5G NR n78 频段，中心频点 3.5GHz，带宽 40MHz
- 采集方式：AGV 沿外围环绕路径匀速行驶，终端上报 RSRP 时序序列
- 文件说明：
  - `train.json`：训练 + 验证集，包含 30 个参考点的 RSRP 时序指纹样本，按 4:1 划分训练 / 验证
  - `test.json`：独立测试集，包含 10 个完全未参与训练的参考点样本，用于泛化性能评估
- 数据格式：每条样本包含文本化 RSRP 时序序列、参考点分类标签
# 代码说明
`Llama_train.py`：完整的模型训练与评估脚本，功能包括：

1. 加载 JSON 格式指纹数据集
2. 配置 LoRA 微调参数（默认 r=8, alpha=32，目标模块 q,k,v,o）
3. 加载 Llama-3.2-3B 基座模型并冻结主干参数
4. 执行训练、早停验证，输出 Top-4 加权融合定位指标
5. 支持调整学习率、epoch 数、K 值等超参数
# 使用方法
1. 安装依赖：

```
pip install torch transformers peft numpy pandas
```

2. 准备 Llama-3.2-3B-Instruct 基座模型，修改脚本中模型路径为本地路径
3. 运行训练与评估：

```
python3 Llama_train.py
```
python3 evolution_lora.py
```
# 引用说明
若使用本项目的数据或代码，请引用对应期刊论文。
