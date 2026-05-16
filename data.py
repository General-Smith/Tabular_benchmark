import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, r2_score
import time

# =============================================================================
# 你的核心加载函数（写得非常好！）
# =============================================================================
def load_dataset(name, task_type):
    """加载并预处理单个数据集"""
    
    # 数据集映射（100%匹配原任务）
    dataset_map = {
        # 分类数据集
        'diabetes': ('name', 'diabetes'),
        'credit-g': ('name', 'credit-g'),
        'MagicTelescope': ('task', 361065),
        'credit-default': ('task', 361055),
        'MiniBooNE': ('task', 361068),
        # 回归数据集
        'boston': ('name', 'boston'),
        'wine_quality': ('task', 361076),
        'superconduct': ('task', 361088),
        'fried': ('name', 'fried'),
        'diamonds': ('task', 361080),
    }
    
    lookup_type, lookup_value = dataset_map[name]
    
    if lookup_type == 'name':
        dataset = openml.datasets.get_dataset(lookup_value)
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format='dataframe'
        )
    else:  # task
        task = openml.tasks.get_task(lookup_value)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(
            target=task.target_name,
            dataset_format='dataframe'
        )
    
    # ✅ 坑1：清理特征名（LightGBM对$[]特殊字符敏感）
    X.columns = [c.replace('$', '_').replace('[', '_').replace(']', '_') 
                 for c in X.columns]
    
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category').cat.codes
    
    # ✅ 坑2：用SimpleImputer规范处理缺失值
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 处理目标变量
    if task_type == 'classification':
        y = y.astype('category').cat.codes
        y = y.astype(int)
    
    # 划分数据集 80/20，固定随机种子42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == 'classification' else None
    )
    
    # 特征标准化（深度学习需要，树模型不需要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ✅ 坑3：回归目标标准化（TabNet必须，90%的人在这里翻车）
    y_scaler = None
    if task_type == 'regression':
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    else:
        y_train_scaled = y_train.values
        y_test_scaled = y_test.values
    
    return {
        'name': name,
        'X_train': X_train.values.astype(np.float32),
        'X_test': X_test.values.astype(np.float32),
        'X_train_scaled': X_train_scaled.astype(np.float32),
        'X_test_scaled': X_test_scaled.astype(np.float32),
        'y_train': y_train.values,
        'y_test': y_test.values,
        'y_train_scaled': y_train_scaled,
        'y_test_scaled': y_test_scaled,
        'y_scaler': y_scaler,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'task_type': task_type
    }

# =============================================================================
# 批量加载所有10个数据集 + 验证
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("开始加载原任务指定的10个数据集...")
    print("="*60)
    
    # 所有数据集配置
    ALL_DATASETS = [
        ('diabetes', 'classification'),
        ('credit-g', 'classification'),
        ('MagicTelescope', 'classification'),
        ('credit-default', 'classification'),
        ('MiniBooNE', 'classification'),
        ('boston', 'regression'),
        ('wine_quality', 'regression'),
        ('superconduct', 'regression'),
        ('fried', 'regression'),
        ('diamonds', 'regression'),
    ]
    
    all_datasets = {}
    for name, task_type in ALL_DATASETS:
        all_datasets[name] = load_dataset(name, task_type)
    
    print("="*60)
    print("✅ 所有数据集加载完成！")
    print("="*60)
    
    # 验证：和原任务表格100%匹配
    print("\n📊 与原任务对比验证:")
    print(f"{'数据集':20s} | {'类型':12s} | {'原任务样本':>10s} | {'实际样本':>10s} | {'匹配'}")
    print("-"*70)
    
    expected_samples = {
        'diabetes': 768, 'credit-g': 1000, 'MagicTelescope': 13376,
        'credit-default': 16714, 'MiniBooNE': 72998, 'boston': 506,
        'wine_quality': 6497, 'superconduct': 21263, 'fried': 40768, 'diamonds': 53940
    }
    
    for name, data in all_datasets.items():
        match = "✓" if data['n_samples'] == expected_samples[name] else "✗"
        print(f"{name:20s} | {data['task_type']:12s} | {expected_samples[name]:10d} | {data['n_samples']:10d} | {match}")
    
    print("\n💾 数据集已保存到 all_datasets 字典中，训练模型直接调用即可！")



    # 固定所有随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')