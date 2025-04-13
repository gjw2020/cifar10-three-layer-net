
# CIFAR-10 三层神经网络分类器

本项目为神经网络课程作业，**不使用深度学习框架（如 PyTorch/TensorFlow）**，基于 `NumPy` 手动实现三层神经网络的前向与反向传播算法，并在 CIFAR-10 数据集上完成图像分类任务。

> ✅ **模型结构**：输入层 (3072) → 隐藏层 (可设定) → 输出层 (10 类别)  
> ✅ **损失函数**：Softmax + 交叉熵 + L2 正则化  
> ✅ **优化算法**：随机梯度下降 (SGD) + 学习率衰减 + Early Stopping

---

## 🎯 最终模型表现

由于训练轮数不高（`epoch=20`），经过超参数调优，最优组合为：

```python
{ 'learning_rate': 0.01, 'hidden_size': 256, 'reg': 0.001 }
```

在该设置下，模型在 **测试集上的分类准确率为：`50.85%`**。

📦 **模型参数下载地址**：  
[点击跳转 Google Drive](https://drive.google.com/file/d/1GnPpTz2bGZZuR8_7qVd2iH6bYwFxocqv/view?usp=sharing)

---

## 💻 实验环境

- 操作系统：Windows 11  
- Python版本：Python 3.11  

---

## 🚀 快速开始

### 1.克隆本项目
```bash
git clone https://github.com/gjw2020/cifar10-three-layer-net.git
cd cifar10-three-layer-net
```

### 2. 创建并激活虚拟环境

```bash
python -m venv myenv
myenv\Scripts\activate  # Windows 环境
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行主程序（训练 + 推理 + 可视化）

```bash
python main.py
```

- 默认执行完整流程，包括加载数据、训练模型、评估精度、保存模型、参数可视化等。
- 如需修改超参数搜索范围，请编辑 `main.py` 中的 `hyperparameter_tuning()` 函数。
- 如需修改模型结构，在model.py中修改。

### 5. 测试训练好的模型

```bash
python predict.py
```
---

- 需将模型参数保存在cifar10-three-layer-net下
- predict部分已经进行了测试集的划分，运行后将展示在最优模型下的acc，以及展示随机取16张图片测试结果。

## 📁 项目结构

```
cifar10-three-layer-net
├── model.py                    # 神经网络模型结构定义
├── train.py                    # 模型训练过程实现
├── data_load.py                # 数据加载与预处理
├── utils.py                    # 工具函数（绘图、保存等）
├── predict.py                  # 推理脚本（载入模型评估）
├── main.py                     # 主程序入口（控制整体流程）
├── hyperparameter_tuning.py    # 网格/随机搜索超参数组合
└── requirements.txt            # 项目依赖文件
```


## 📬 联系方式

如有疑问，欢迎通过 GitHub Issue联系。
