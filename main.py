# ===================== 5_main.py 完整内容 =====================
import os
import json
from tqdm import tqdm
from data import set_seed, load_dataset
from baseline_models import train_random_forest, train_xgboost, train_lightgbm, train_catboost
from train import train_mlp, train_resnet, train_ft_transformer, train_tabnet

def run_all_experiments():
    """运行所有实验"""
    
    # 所有数据集配置
    datasets = [
        # 分类数据集
        ('diabetes', 'classification'),
        ('credit-g', 'classification'),
        ('MagicTelescope', 'classification'),
        ('credit-default', 'classification'),
        ('MiniBooNE', 'classification'),
        # 回归数据集
        ('boston', 'regression'),
        ('wine_quality', 'regression'),
        ('superconduct', 'regression'),
        ('fried', 'regression'),
        ('diamonds', 'regression'),
    ]
    
    # 所有模型
    models = {
        # 树模型
        'RandomForest': train_random_forest,
        'XGBoost': train_xgboost,
        'LightGBM': train_lightgbm,
        'CatBoost': train_catboost,
        # 深度学习模型
        'MLP': train_mlp,
        'ResNet': train_resnet,
        'FT-Transformer': train_ft_transformer,
        'TabNet': train_tabnet,
    }
    
    results = {}
    
    for dataset_name, task_type in tqdm(datasets, desc='Datasets'):
        print(f"\n=== Processing {dataset_name} ({task_type}) ===")
        
        # 加载数据
        data = load_dataset(dataset_name, task_type)
        results[dataset_name] = {
            'task_type': task_type,
            'n_samples': data['n_samples'],
            'n_features': data['n_features'],
            'models': {}
        }
        
        for model_name, train_func in tqdm(models.items(), desc='Models', leave=False):
            print(f"  Training {model_name}...")
            try:
                result = train_func(data)
                results[dataset_name]['models'][model_name] = {
                    'score': float(result['score']),
                    'train_time': float(result['train_time']),
                    'model_type': 'tree-based' if model_name in ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost'] else 'deep-learning'
                }
                print(f"    Score: {result['score']:.4f}, Time: {result['train_time']:.2f}s")
            except Exception as e:
                print(f"    Error: {str(e)}")
                results[dataset_name]['models'][model_name] = {
                    'score': None,
                    'train_time': None,
                    'error': str(e)
                }
    
    # 保存结果
    with open('results/all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# 运行所有实验
if __name__ == '__main__':
    set_seed(42)
    
    # 自动创建文件夹
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    results = run_all_experiments()
    print("\n=== All experiments completed! ===")