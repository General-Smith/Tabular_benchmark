# ===================== 3_train.py 完整内容 =====================
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, r2_score
from models import MLP, ResNet, FTTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 通用深度学习训练函数 --------------------
def train_deep_model(model_class, data, model_params, lr=0.001):
    """通用深度学习模型训练函数"""
    task_type = data['task_type']
    input_dim = data['X_train_scaled'].shape[1]
    output_dim = 1 if task_type == 'regression' else len(np.unique(data['y_train']))
    
    # 小数据集调整batch size
    n_samples = len(data['X_train_scaled'])
    batch_size = min(256, max(16, n_samples // 4))
    
    # 准备数据
    X_train = torch.FloatTensor(data['X_train_scaled'])
    y_train = torch.FloatTensor(data['y_train_scaled']) if task_type == 'regression' \
        else torch.LongTensor(data['y_train_scaled'])
    
    X_test = torch.FloatTensor(data['X_test_scaled'])
    y_test = torch.FloatTensor(data['y_test_scaled']) if task_type == 'regression' \
        else torch.LongTensor(data['y_test_scaled'])
    
    # 训练/验证划分 (最后20%作为验证)
    val_size = int(0.2 * len(X_train))
    X_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = model_class(input_dim, output_dim, **model_params).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss() if task_type == 'regression' else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    
    # 训练循环
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(100):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            if task_type == 'regression':
                outputs = outputs.squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                if task_type == 'regression':
                    outputs = outputs.squeeze()
                val_loss += criterion(outputs, batch_y).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    train_time = time.time() - start_time
    
    # 加载最佳模型
    model.load_state_dict(best_state)
    
    # 评估
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        if task_type == 'regression':
            y_pred = outputs.squeeze().cpu().numpy()
            # 反标准化
            y_pred = data['y_scaler'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
            score = r2_score(data['y_test'], y_pred)
        else:
            y_pred = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            score = roc_auc_score(data['y_test'], y_pred)
    
    return {'score': score, 'train_time': train_time, 'model': model}

# -------------------- 各个深度模型的包装函数 --------------------
def train_mlp(data):
    return train_deep_model(MLP, data, {'dropout': 0.2}, lr=0.001)


def train_resnet(data):
    return train_deep_model(ResNet, data, {'dropout': 0.2}, lr=0.001)

def train_ft_transformer(data):
    return train_deep_model(FTTransformer, data, {
        'd_model': 64, 'n_heads': 4, 'd_ffn': 256, 'dropout': 0.2
    }, lr=0.001)

# ===================== 3_train.py 最末尾 =====================
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

def train_tabnet(data):
    task_type = data['task_type']
    
    if task_type == 'classification':
        model = TabNetClassifier(
            n_d=4, n_a=4, n_steps=2,
            gamma=1.3,
            lambda_sparse=1e-4,
            seed=42,
            verbose=0
        )
    else:
        model = TabNetRegressor(
            n_d=4, n_a=4, n_steps=2,
            gamma=1.3,
            lambda_sparse=1e-4,
            seed=42,
            verbose=0
        )
    
    start_time = time.time()
    
    if task_type == 'classification':
        model.fit(
            data['X_train_scaled'], data['y_train'],
            max_epochs=100,
            patience=10,
            batch_size=256,
            virtual_batch_size=128,
            eval_set=[(data['X_test_scaled'], data['y_test'])],
        )
        y_pred = model.predict_proba(data['X_test_scaled'])[:, 1]
        score = roc_auc_score(data['y_test'], y_pred)
    else:
        # 注意：回归必须使用标准化的y！
        model.fit(
            data['X_train_scaled'], data['y_train_scaled'].reshape(-1, 1),
            max_epochs=100,
            patience=10,
            batch_size=256,
            virtual_batch_size=128,
            eval_set=[(data['X_test_scaled'], data['y_test_scaled'].reshape(-1, 1))],
        )
        y_pred_scaled = model.predict(data['X_test_scaled']).flatten()
        # 反标准化
        y_pred = data['y_scaler'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        score = r2_score(data['y_test'], y_pred)
    
    train_time = time.time() - start_time
    
    return {'score': score, 'train_time': train_time, 'model': model}