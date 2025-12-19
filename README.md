# 🧬 Zombie User Detection Project

## 项目简介

本项目基于 Twitter 僵尸用户（social spambots）数据集，通过 **数字 DNA（Digital DNA）** 方法，将用户行为序列编码为图像，并使用 **CNN 模型**进行僵尸用户识别。

目标是：

1. 分析用户行为模式
2. 自动识别僵尸用户（spam bot）
3. 可视化训练和评估结果

---

## 数据集

使用 **Cresci-2017 Twitter Bot Dataset**：

* https://botometer.osome.iu.edu/bot-repository/datasets.html?utm_source=chatgpt.com
* **Genuine Users**：真实用户
* **Social Spambots**：僵尸用户
* 数据包含用户信息、推文文本

---

## 项目流程

整个项目分为 **数字 DNA 生成 → 图像转换 → 数据集划分 → CNN 训练 → 模型评估** 五个阶段：

```
Raw Data (CSV)
      │
      ▼
generate_all_dna.py → 生成用户数字 DNA 序列
      │
      ▼
DNA_to_image.py → 将 DNA 序列转换为灰度图像
      │
      ▼
split_train_val_test.py → 划分训练集、验证集、测试集 (.npy)
      │
      ▼
train_cnn.py → 使用训练集和验证集训练 CNN 模型
      │
      ▼
evaluate_model.py → 使用测试集评估模型，生成 ROC 曲线和效果表格
```

> 可选：使用 `run_all.py` 按顺序自动运行以上所有步骤



## 脚本说明

| 脚本文件                        | 功能                                      |
| --------------------------- | --------------------------------------- |
| **generate_all_dna.py**     | 读取用户推文，按行为生成数字 DNA 序列                   |
| **DNA_to_image.py**         | 将 DNA 序列编码为灰度图像，保存为 `.npy`              |
| **split_train_val_test.py** | 按比例划分训练集/验证集/测试集，保存 `.npy` 文件           |
| **train_cnn.py**            | 训练 CNN 模型，使用验证集 EarlyStopping，保存训练曲线和模型 |
| **evaluate_model.py**       | 使用测试集评估模型，生成指标表格和 ROC 曲线                |
| **run_all.py**              | 按顺序调用上述脚本，实现一键运行整个流程                    |

---

## 模型与方法

* **特征**：用户推文序列的数字 DNA（TRC 三种编码）
* **模型**：CNN（卷积神经网络）
* **训练策略**：训练集训练，验证集 EarlyStopping
* **评估指标**：Accuracy, Precision, Recall, F1, ROC-AUC
* **输出**：训练曲线图、ROC 曲线图、效果表格 CSV

---

## 使用说明

1. 克隆项目仓库：

```bash
git clone <your-repo-url>
cd Zombie-user-detection
```

2. 下载并解压 **Cresci-2017 数据集**到 `data/` 文件夹

3. 安装依赖：

```bash
pip install -r requirements.txt
# 或者手动安装 pandas, numpy, tensorflow, scikit-learn, matplotlib
```

4. 运行整个流程：

```bash
python run_all.py
```

5. 查看结果：

* `dna_images/` → 数字 DNA 图像
* `dna_cnn_model.h5` → CNN 模型文件
* `training_curve.png` → 训练曲线
* `roc_curve.png` → ROC曲线
* `test_metrics.csv` → 测试集指标表格

---

## 注意事项

* 数据集较大，生成 DNA 和训练 CNN 可能需要较多内存
* 如果运行报错，请检查 **路径和文件夹结构**
* 可根据实验需求调整训练/验证/测试集比例


