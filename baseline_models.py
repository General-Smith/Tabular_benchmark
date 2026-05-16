import time
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
def train_random_forest(data):
    task_type = data['task_type']
    
    if task_type == 'classification':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    
    start_time = time.time()
    model.fit(data['X_train'], data['y_train'])
    train_time = time.time() - start_time
    
    y_pred = model.predict_proba(data['X_test'])[:, 1] if task_type == 'classification' \
        else model.predict(data['X_test'])
    
    score = roc_auc_score(data['y_test'], y_pred) if task_type == 'classification' \
        else r2_score(data['y_test'], y_pred)
    
    return {'score': score, 'train_time': train_time, 'model': model}


def train_xgboost(data):
    task_type = data['task_type']
    
    if task_type == 'classification':
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    
    start_time = time.time()
    model.fit(data['X_train'], data['y_train'])
    train_time = time.time() - start_time
    y_pred = model.predict_proba(data['X_test'])[:, 1] if task_type == 'classification' \
        else model.predict(data['X_test'])
    
    score = roc_auc_score(data['y_test'], y_pred) if task_type == 'classification' \
        else r2_score(data['y_test'], y_pred)
    
    return {'score': score, 'train_time': train_time, 'model': model}


def train_lightgbm(data):
    task_type = data['task_type']
    
    if task_type == 'classification':
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=-1,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        model = LGBMRegressor(
            n_estimators=100,
            max_depth=-1,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    start_time = time.time()
    model.fit(data['X_train'], data['y_train'])
    train_time = time.time() - start_time
    
    y_pred = model.predict_proba(data['X_test'])[:, 1] if task_type == 'classification' \
        else model.predict(data['X_test'])
    
    score = roc_auc_score(data['y_test'], y_pred) if task_type == 'classification' \
        else r2_score(data['y_test'], y_pred)
    
    return {'score': score, 'train_time': train_time, 'model': model}


def train_catboost(data):
    task_type = data['task_type']
    
    if task_type == 'classification':
        model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=0,
            thread_count=-1
        )
    else:
        model = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=0,
            thread_count=-1
        )
    
    start_time = time.time()
    model.fit(data['X_train'], data['y_train'])
    train_time = time.time() - start_time
    
    y_pred = model.predict_proba(data['X_test'])[:, 1] if task_type == 'classification' \
        else model.predict(data['X_test'])
    
    score = roc_auc_score(data['y_test'], y_pred) if task_type == 'classification' \
        else r2_score(data['y_test'], y_pred)
    
    return {'score': score, 'train_time': train_time, 'model': model}