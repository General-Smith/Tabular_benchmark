# Tabular-benchmark
Benchmark of 8 tabular data models (4 tree-based + 4 deep learning) on 10 standard OpenML datasets
# Tabular Data Model Benchmark

## 项目简介
本项目在10个标准OpenML表格数据集上，系统复现并对比了8个主流机器学习模型的性能，包括4种树模型和4种深度学习模型，旨在量化不同模型在表格数据上的表现差异与数据量效应。

## 实验设计
### 数据集（共10个，5个分类 + 5个回归）
#### 分类数据集（5个）
| Dataset | OpenML ID | Samples | Features | Description |
|---------|-----------|---------|----------|-------------|
| diabetes | name: diabetes | 768 | 8 | Pima Indians Diabetes |
| credit-g | name: credit-g | 1,000 | 20 | German Credit |
| MagicTelescope | suite 337, task 361065 | 13,376 | 10 | MAGIC Gamma Telescope |
| credit-default | suite 337, task 361055 | 16,714 | 10 | Credit Default |
| MiniBooNE | suite 337, task 361068 | 72,998 | 50 | MiniBooNE particle identification |

#### 回归数据集（5个）
| Dataset | OpenML ID | Samples | Features | Description |
|---------|-----------|---------|----------|-------------|
| boston | name: boston | 506 | 13 | Boston Housing |
| wine_quality | suite 336, task 361076 | 6,497 | 11 | Wine Quality |
| superconduct | suite 336, task 361088 | 21,263 | 79 | Superconductivity |
| fried | name: fried | 40,768 | 10 | Friedman (synthetic) |
| diamonds | suite 336, task 361080 | 53,940 | 6 | Diamonds Price |

### 评估指标
- 分类任务：ROC-AUC (Area Under the ROC Curve)
- 回归任务：R² Score (Coefficient of Determination)

### 对比模型
#### 树模型
- Random Forest
- XGBoost
- LightGBM
- CatBoost

#### 深度学习模型
- MLP (多层感知机)
- ResNet (残差网络)
- FT-Transformer
- TabNet

## 核心实验结果
### 1. 整体性能对比
![Model Performance Comparison](figures/figure1_model_comparison.png)

### 2. 模型平均排名
![Average Ranking](figures/figure2_average_ranking.png)
- **整体最佳**：XGBoost
- **第二名**：LightGBM
- **最佳深度学习模型**：MLP

### 3. 数据量与性能差距
![Performance Gap vs Dataset Size](figures/figure3_gap_scatter.png)
- 小数据集：树模型全面领先深度学习
- 大数据集：深度学习与树模型的差距显著缩小
- FT-Transformer在大数据集上性能接近树模型

## 运行方式
### 环境配置
```bash
pip install -r requirements.txt
