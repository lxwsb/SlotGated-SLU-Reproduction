# SlotGated-SLU 复现版说明文档

## 项目来源与致谢
- 本项目为复现自中山大学陈韫侬（Yun-Nung Chen）团队的论文及其开源代码：
  - 论文：Goo et al., 2018 "Slot-Gated Modeling for Joint Slot Filling and Intent Prediction"
  - 原始GitHub项目：[https://github.com/MiuLab/SlotGated-SLU/tree/master](https://github.com/MiuLab/SlotGated-SLU/tree/master?tab=readme-ov-file)
- 本地仓库为**个人学习/研究复现用途**，代码和理论归原作者所有。

## 项目简介
本项目用于**联合抽取（Joint Extraction）任务**——自动完成序列标注（Slot Filling）与意图识别（Intent Detection）。核心模型为Slot-Gated机制（支持full attention与intent-only attention两种模式），特别适合ATIS、SNIPS等数据集。

- Slot Filling: 识别语句中的关键实体（如航班、日期、城市等）
- Intent Detection: 预测用户意图（如预定航班/查询天气等）

## 复现的主要依赖及部分环境
- Python 版本：**3.9.19**
- 主要依赖包（仅展示核心组件，更多环境包可参考`pip list`输出）：
  - tensorflow-gpu 2.10.0 或 tensorflow 2.x（建议 >=2.10）
  - tensorflow-estimator 2.10.0
  - numpy 1.24.0
  - matplotlib 3.9.0
  - pandas 2.2.2
  - scikit-learn 1.5.1
  - absl-py、protobuf、h5py、astunparse、keras、six 等均为TF与科学计算生态常规依赖
- 注意事项：
  - 推荐直接运行`trainV2.py`，该文件可自动兼容TF2.x+环境，迁移自TF1.x经典写法。
  - 其它Python 3.8+/3.10版本一般也可正常运行，关键在于TensorFlow与numpy等接口兼容。

如需完整的本地pip依赖可通过如下命令导出：
```bash
pip freeze > requirements.txt
```

其余详见原项目ReadMe和本地仓库源码说明。

## 目录与数据说明
```
SlotGated-SLU-master/
 ├── data/           # 数据集（atis、snips等，需自行准备）
 ├── model/          # 训练生成的模型存放目录
 ├── vocab/          # 词表，会自动生成
 ├── trainV2.py      # 建议主运行入口（支持TF2.x）
 ├── train.py        # 原生TF1.x老版本
 ├── utils.py        # 数据处理与评测工具
 └── README.md       # 本说明文档
```

数据目录下需有`train/`、`test/`及`valid/`三个子文件夹，每个文件夹下有`seq.in`（输入序列）、`seq.out`（slot标签）、`label`（intent标签），格式与原仓库一致。

## 常用命令示例
### 基本用法（复现实验）
- ATIS数据集上使用默认full attention模型：

```bash
python trainV2.py --dataset=atis
```
- SNIPS数据集：
```bash
python trainV2.py --dataset=snips
```
- 指定隐藏单元数/更长训练/不用早停：
```bash
python trainV2.py --num_units=64 --max_epochs=50 --no_early_stop --dataset=atis
```
- 切换为intent attention (`intent_only`模式)：
```bash
python trainV2.py --dataset=atis --model_type=intent_only
```
- 查看全部命令参数：
```bash
python trainV2.py -h
```

> 注：推荐优先使用`trainV2.py`运行（自动兼容TF2.x环境）。TF1.4原生环境可使用`train.py`。

## 实验结果输出
- 训练过程会自动记录loss、slot F1、intent acc等曲线，结果图保存在`./model/training_curves.png`。
- CSV日志位于`./model/training_metrics.csv`。
- 检查点（ckpt模型）会保存在`./model`目录。

---

## Reference

Goo et al., 2018. Slot-Gated Modeling for Joint Slot Filling and Intent Prediction.

GitHub: https://github.com/MiuLab/SlotGated-SLU/tree/master?tab=readme-ov-file

