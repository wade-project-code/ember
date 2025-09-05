# Ember 惡意軟體分析教學指南

> 🛡️ 本專案專注於防禦性安全和教育研究目的

## 📚 目錄

- [簡介](#簡介)
- [技術規格](#技術規格)
- [技術背景](#技術背景)
- [Ember 資料集概述](#ember-資料集概述)
- [分析方法論](#分析方法論)
- [實務範例](#實務範例)
- [工具與技術](#工具與技術)
- [高級分析技術](#高級分析技術)
- [技術參考](#技術參考)

## 簡介

Ember 資料集是由 Endgame（現為 Elastic Security）開發的大規模惡意軟體研究資料集，包含超過 100 萬個 PE 檔案的 2351 個特徵。本教學指南旨在幫助學生、研究人員和網路安全從業人員理解和應用這個強大的資料集進行防禦性安全研究。

### 為什麼選擇 Ember？

- **標準化**：提供一致的特徵提取和評估基準
- **規模化**：大規模資料集支援深度學習研究
- **實用性**：基於真實世界的惡意軟體樣本
- **開放性**：開源且免費提供給研究社群
- **可重現性**：標準化的方法論確保研究結果可重現

## 技術規格

### 核心技術能力

1. **PE 檔案逆向工程**
   - 檔案格式解析與結構分析
   - 匯入/匯出表分析技術
   - 節區特徵提取和熵值計算

2. **特徵工程技術棧**
   - 2351維特徵向量操作
   - 多維度特徵空間映射
   - 統計學習特徵選擇

3. **機器學習實作**
   - LightGBM/XGBoost 梯度提升
   - 集成學習和模型融合
   - 對抗性樣本檢測

4. **分析技術方法**
   - SHAP/LIME 模型解釋性
   - 異常檢測算法實作
   - 高維度資料視覺化

## 技術背景

### PE 檔案分析基礎

PE（Portable Executable）格式是 Windows 執行檔的標準格式，包含豐富的結構化資訊：

```python
# PE 檔案主要結構
pe_structure = {
    "DOS_Header": "DOS 相容性標頭",
    "NT_Headers": {
        "File_Header": "檔案基本資訊",
        "Optional_Header": "載入資訊和特性"
    },
    "Section_Headers": "節區描述表",
    "Sections": {
        ".text": "程式碼段",
        ".data": "初始化資料段",
        ".rdata": "只讀資料段",
        ".rsrc": "資源段"
    },
    "Import_Table": "匯入函數表",
    "Export_Table": "匯出函數表"
}
```

### 特徵提取方法

Ember 採用多層次特徵提取方法：

1. **結構特徵**：檔案大小、節區數量、入口點位置
2. **匯入特徵**：API 函數呼叫、DLL 依賴關係
3. **字節統計**：位元組頻率分佈、熵值計算
4. **字串特徵**：可列印字串的統計分析
5. **PE 標頭特徵**：編譯時間戳、機器類型、特性標誌

## Ember 資料集概述

### 資料集結構

```
ember_dataset/
├── train_features.jsonl      # 訓練特徵 (800,000 樣本)
├── train_labels.jsonl        # 訓練標籤
├── test_features.jsonl       # 測試特徵 (200,000 樣本)
├── vectorized/              # 向量化特徵
│   ├── X_train.npy         # 訓練特徵矩陣
│   ├── y_train.npy         # 訓練標籤
│   ├── X_test.npy          # 測試特徵矩陣
│   └── metadata.json       # 元資料資訊
└── README.md               # 資料集說明
```

### 特徵類型分析

#### 1. 一般檔案資訊 (10 個特徵)
```python
general_features = [
    "size",              # 檔案大小
    "vsize",             # 虛擬大小  
    "has_debug",         # 是否包含除錯資訊
    "exports",           # 匯出函數數量
    "imports",           # 匯入函數數量
    "has_relocations",   # 是否有重定位表
    "has_resources",     # 是否有資源段
    "has_signature",     # 是否有數位簽章
    "has_tls",           # 是否使用 TLS
    "symbols"            # 符號數量
]
```

#### 2. 標頭特徵 (62 個特徵)
```python
header_features = {
    "coff": {
        "timestamp": "編譯時間戳",
        "machine": "目標機器架構",
        "characteristics": "檔案特性標誌"
    },
    "optional": {
        "subsystem": "子系統類型",
        "dll_characteristics": "DLL 特性",
        "magic": "PE 格式標識",
        "major_image_version": "映像主版本號",
        "minor_image_version": "映像次版本號",
        "major_linker_version": "連結器主版本號",
        "minor_linker_version": "連結器次版本號",
        "major_operating_system_version": "作業系統主版本號",
        "minor_operating_system_version": "作業系統次版本號",
        "major_subsystem_version": "子系統主版本號",
        "minor_subsystem_version": "子系統次版本號",
        "sizeof_code": "程式碼段大小",
        "sizeof_headers": "標頭大小",
        "sizeof_heap_commit": "堆積提交大小"
    }
}
```

#### 3. 匯入特徵 (1280 個特徵)
匯入特徵基於 Windows API 函數的呼叫統計：

```python
# 常見的惡意軟體 API 類別
malware_apis = {
    "Process": ["CreateProcessA", "OpenProcess", "TerminateProcess"],
    "File": ["CreateFileA", "ReadFile", "WriteFile", "DeleteFileA"],
    "Registry": ["RegOpenKeyA", "RegSetValueA", "RegDeleteKeyA"],
    "Network": ["socket", "connect", "send", "recv"],
    "Crypto": ["CryptGenKey", "CryptEncrypt", "CryptDecrypt"],
    "Memory": ["VirtualAlloc", "VirtualProtect", "VirtualFree"],
    "Service": ["OpenSCManagerA", "CreateServiceA", "StartServiceA"],
    "Debug": ["IsDebuggerPresent", "CheckRemoteDebuggerPresent"]
}
```

#### 4. 匯出特徵 (128 個特徵)
```python
export_features = [
    "export_count",      # 匯出函數數量
    "export_entropy",    # 匯出名稱熵值
    "export_names",      # 具名匯出數量
    "export_ordinals"    # 序號匯出數量
]
```

#### 5. 節區特徵 (255 個特徵)
```python
section_features = {
    "entry": "入口點節區",
    "sections": [
        {
            "name": "節區名稱",
            "size": "節區大小", 
            "entropy": "節區熵值",
            "vsize": "虛擬大小",
            "props": "節區屬性"
        }
    ]
}
```

#### 6. 位元組直方圖 (256 個特徵)
```python
# 計算位元組頻率分佈
def compute_byte_histogram(file_data):
    histogram = [0] * 256
    for byte in file_data:
        histogram[byte] += 1
    
    # 正規化
    total = len(file_data)
    return [count / total for count in histogram]
```

#### 7. 位元組熵直方圖 (256 個特徵)
```python
import math

def compute_entropy_histogram(file_data, window_size=1024):
    entropies = []
    
    for i in range(0, len(file_data), window_size):
        window = file_data[i:i+window_size]
        if len(window) < window_size:
            break
            
        # 計算視窗熵值
        histogram = [0] * 256
        for byte in window:
            histogram[byte] += 1
        
        entropy = 0
        for count in histogram:
            if count > 0:
                p = count / len(window)
                entropy -= p * math.log2(p)
        
        entropies.append(entropy)
    
    # 將熵值分佈到 256 個區間
    return create_histogram(entropies, bins=256)
```

#### 8. 字串特徵 (104 個特徵)
```python
string_features = {
    "numstrings": "字串總數",
    "avlength": "平均長度",
    "printabledist": "可列印字元分佈 (96 維)",
    "printables": "可列印字串比例",
    "entropy": "字串熵值",
    "paths": "路徑字串數量",
    "urls": "URL 字串數量", 
    "registry": "註冊表項目數量",
    "MZ": "MZ 標識出現次數"
}
```

### 標籤系統

Ember 採用三分類系統：

```python
labels = {
    0: "良性軟體 (Benign)",
    1: "惡意軟體 (Malicious)", 
    -1: "未知/不確定 (Unknown)"
}

# 標籤分佈 (訓練集)
label_distribution = {
    "benign": 400000,    # 50%
    "malicious": 400000, # 50%
    "unlabeled": 200000  # 在完整資料集中
}
```

## 分析方法論

### 1. 靜態分析流程

```python
import ember
import numpy as np
import pandas as pd

# 載入預訓練的特徵提取器
extractor = ember.FeatureExtractor()

def static_analysis_pipeline(pe_file_path):
    """完整的靜態分析流程"""
    
    # 1. 特徵提取
    with open(pe_file_path, 'rb') as f:
        raw_features = extractor.feature_vector(f.read())
    
    # 2. 特徵正規化
    features = ember.normalize_features(raw_features)
    
    # 3. 載入預訓練模型
    lgbm_model = ember.load_model()
    
    # 4. 預測
    prediction = lgbm_model.predict([features])[0]
    
    # 5. 結果解釋
    result = {
        'prediction': prediction,
        'confidence': abs(prediction - 0.5) * 2,
        'classification': 'malicious' if prediction > 0.5 else 'benign',
        'features': features
    }
    
    return result

# 範例使用
result = static_analysis_pipeline("sample.exe")
print(f"分類結果: {result['classification']}")
print(f"信心度: {result['confidence']:.3f}")
```

### 2. 特徵重要性分析

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

def feature_importance_analysis(X_train, y_train):
    """分析特徵重要性"""
    
    # 訓練隨機森林模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 獲取特徵重要性
    importance = rf.feature_importances_
    
    # 獲取 Ember 特徵名稱
    feature_names = ember.get_feature_names()
    
    # 排序特徵
    indices = np.argsort(importance)[::-1]
    
    # 顯示前 20 個重要特徵
    print("前 20 個最重要特徵:")
    for i in range(20):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:30s} ({importance[idx]:.4f})")
    
    # 視覺化特徵重要性
    plt.figure(figsize=(12, 8))
    plt.title("特徵重要性分析")
    plt.bar(range(20), importance[indices[:20]])
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=45)
    plt.tight_layout()
    plt.show()
    
    return importance, indices
```

### 3. 模型訓練和評估

```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

def train_ember_model(X_train, y_train, X_test, y_test):
    """訓練和評估 Ember 模型"""
    
    # 1. 資料預處理
    # 移除未標記的樣本 (標籤 = -1)
    mask = y_train != -1
    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]
    
    # 2. LightGBM 參數設置
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 1024,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # 3. 建立訓練資料集
    train_data = lgb.Dataset(X_train_clean, label=y_train_clean)
    
    # 4. 訓練模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # 5. 測試集評估
    test_mask = y_test != -1
    X_test_clean = X_test[test_mask]
    y_test_clean = y_test[test_mask]
    
    y_pred_proba = model.predict(X_test_clean, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 6. 評估結果
    print("模型評估結果:")
    print(classification_report(y_test_clean, y_pred, 
                              target_names=['Benign', 'Malicious']))
    
    # 7. 混淆矩陣
    cm = confusion_matrix(y_test_clean, y_pred)
    print("\n混淆矩陣:")
    print(cm)
    
    # 8. 交叉驗證
    cv_scores = cross_val_score(
        lgb.LGBMClassifier(**params), 
        X_train_clean, y_train_clean, 
        cv=5, scoring='accuracy'
    )
    print(f"\n5折交叉驗證準確率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return model
```

### 4. 對抗性樣本檢測

```python
def adversarial_detection(model, X_test, y_test, epsilon=0.01):
    """檢測對抗性樣本攻擊"""
    
    # Fast Gradient Sign Method (FGSM) 攻擊
    def fgsm_attack(x, y, model, epsilon):
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # 前向傳播
        output = model(x_tensor)
        loss = F.cross_entropy(output, y_tensor)
        
        # 反向傳播
        model.zero_grad()
        loss.backward()
        
        # 生成對抗性樣本
        x_adv = x_tensor + epsilon * x_tensor.grad.sign()
        
        return x_adv.detach().numpy()
    
    # 測試對抗性攻擊的效果
    correct_original = 0
    correct_adversarial = 0
    
    for i in range(len(X_test)):
        x_orig = X_test[i:i+1]
        y_true = y_test[i:i+1]
        
        # 原始預測
        pred_orig = model.predict(x_orig)[0] > 0.5
        if pred_orig == y_true:
            correct_original += 1
        
        # 對抗性攻擊
        x_adv = fgsm_attack(x_orig, y_true, model, epsilon)
        pred_adv = model.predict(x_adv)[0] > 0.5
        
        if pred_adv == y_true:
            correct_adversarial += 1
    
    print(f"原始準確率: {correct_original / len(X_test):.4f}")
    print(f"對抗性攻擊後準確率: {correct_adversarial / len(X_test):.4f}")
    
    return correct_original / len(X_test), correct_adversarial / len(X_test)
```

## 實務範例

### 範例 1: 惡意軟體家族分類

```python
def malware_family_classification():
    """惡意軟體家族分類範例"""
    
    # 載入資料和標籤
    X_train = ember.read_vectorized_features("train")
    y_train = ember.read_labels("train")
    
    # 只選擇惡意軟體樣本
    malicious_mask = y_train == 1
    X_malicious = X_train[malicious_mask]
    
    # 假設我們有家族標籤 (需要額外資料)
    # family_labels = load_family_labels()  # 載入家族標籤
    
    # 使用聚類分析推斷家族
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # 標準化特徵
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_malicious)
    
    # K-means 聚類
    n_families = 10  # 假設有 10 個家族
    kmeans = KMeans(n_clusters=n_families, random_state=42)
    family_clusters = kmeans.fit_predict(X_scaled)
    
    # 分析每個聚類的特徵
    for i in range(n_families):
        cluster_samples = X_malicious[family_clusters == i]
        cluster_size = len(cluster_samples)
        
        print(f"\n家族 {i+1} (樣本數: {cluster_size}):")
        
        # 計算平均特徵
        mean_features = cluster_samples.mean(axis=0)
        feature_names = ember.get_feature_names()
        
        # 顯示最突出的特徵
        top_features = np.argsort(mean_features)[-10:]
        for idx in reversed(top_features):
            print(f"  {feature_names[idx]}: {mean_features[idx]:.4f}")
    
    return kmeans, scaler
```

### 範例 2: 零日惡意軟體檢測

```python
def zero_day_detection():
    """零日惡意軟體檢測範例"""
    
    # 載入訓練資料
    X_train = ember.read_vectorized_features("train")
    y_train = ember.read_labels("train")
    
    # 移除未標記資料
    labeled_mask = y_train != -1
    X_labeled = X_train[labeled_mask]
    y_labeled = y_train[labeled_mask]
    
    # 分割已知和未知惡意軟體
    # 假設我們有時間戳資訊
    cutoff_date = "2020-01-01"  # 分界日期
    
    # 使用異常檢測方法
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    
    # 1. 基於已知良性軟體的異常檢測
    benign_mask = y_labeled == 0
    X_benign = X_labeled[benign_mask]
    
    # 訓練 One-Class SVM
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    ocsvm.fit(X_benign)
    
    # 2. 基於隔離森林的異常檢測
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_labeled)
    
    # 測試新樣本
    X_test = ember.read_vectorized_features("test")
    
    # One-Class SVM 預測
    ocsvm_pred = ocsvm.predict(X_test)
    anomaly_count_svm = np.sum(ocsvm_pred == -1)
    
    # 隔離森林預測
    iso_pred = iso_forest.predict(X_test)
    anomaly_count_iso = np.sum(iso_pred == -1)
    
    print(f"One-Class SVM 檢測到 {anomaly_count_svm} 個異常樣本")
    print(f"隔離森林檢測到 {anomaly_count_iso} 個異常樣本")
    
    # 結合兩種方法
    combined_anomalies = (ocsvm_pred == -1) & (iso_pred == -1)
    high_confidence_anomalies = np.sum(combined_anomalies)
    
    print(f"高信心度異常樣本: {high_confidence_anomalies}")
    
    return ocsvm, iso_forest
```

### 範例 3: 特徵工程和選擇

```python
def advanced_feature_engineering():
    """進階特徵工程範例"""
    
    X_train = ember.read_vectorized_features("train")
    y_train = ember.read_labels("train")
    
    # 移除未標記資料
    labeled_mask = y_train != -1
    X_labeled = X_train[labeled_mask]
    y_labeled = y_train[labeled_mask]
    
    # 1. 特徵縮放
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_labeled)
    
    # 2. 特徵選擇
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    
    # 使用互資訊選擇前 1000 個特徵
    selector = SelectKBest(score_func=mutual_info_classif, k=1000)
    X_selected = selector.fit_transform(X_scaled, y_labeled)
    
    # 3. 主成分分析降維
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)  # 保留 95% 變異
    X_pca = pca.fit_transform(X_selected)
    
    print(f"原始特徵數: {X_labeled.shape[1]}")
    print(f"選擇後特徵數: {X_selected.shape[1]}")
    print(f"PCA 後特徵數: {X_pca.shape[1]}")
    print(f"累積解釋變異比: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")
    
    # 4. 特徵重要性視覺化
    selected_indices = selector.get_support(indices=True)
    feature_names = ember.get_feature_names()
    selected_features = [feature_names[i] for i in selected_indices]
    scores = selector.scores_[selected_indices]
    
    # 繪製特徵重要性
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    top_20 = np.argsort(scores)[-20:]
    plt.barh(range(20), scores[top_20])
    plt.yticks(range(20), [selected_features[i] for i in top_20])
    plt.xlabel('互資訊分數')
    plt.title('前 20 個最重要特徵')
    plt.tight_layout()
    plt.show()
    
    return scaler, selector, pca, X_pca, y_labeled

# 執行特徵工程
scaler, selector, pca, X_engineered, y_engineered = advanced_feature_engineering()
```

### 範例 4: 模型解釋性分析

```python
def model_interpretability():
    """模型解釋性分析範例"""
    
    import shap
    import lime
    import lime.tabular
    
    # 載入預訓練模型
    model = ember.load_model()
    X_test = ember.read_vectorized_features("test")
    feature_names = ember.get_feature_names()
    
    # 1. SHAP 解釋
    print("計算 SHAP 值...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:100])  # 前 100 個樣本
    
    # SHAP 摘要圖
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
    plt.title('SHAP 特徵重要性摘要')
    plt.tight_layout()
    plt.show()
    
    # 2. LIME 解釋
    print("設置 LIME 解釋器...")
    lime_explainer = lime.tabular.LimeTabularExplainer(
        X_test,
        feature_names=feature_names,
        class_names=['Benign', 'Malicious'],
        mode='classification'
    )
    
    # 解釋單個樣本
    idx = 0  # 解釋第一個測試樣本
    lime_exp = lime_explainer.explain_instance(
        X_test[idx], 
        model.predict_proba, 
        num_features=10
    )
    
    # 顯示解釋結果
    print(f"\n樣本 {idx} 的 LIME 解釋:")
    for feature, importance in lime_exp.as_list():
        print(f"  {feature}: {importance:.4f}")
    
    # 3. 特徵重要性熱圖
    def plot_feature_importance_heatmap():
        # 計算各特徵類型的平均重要性
        importance = np.abs(shap_values).mean(axis=0)
        
        # Ember 特徵分組
        feature_groups = {
            'General': list(range(10)),
            'Header': list(range(10, 72)),
            'Import': list(range(72, 1352)),
            'Export': list(range(1352, 1480)),
            'Section': list(range(1480, 1735)),
            'Byte Histogram': list(range(1735, 1991)),
            'Byte Entropy': list(range(1991, 2247)),
            'String': list(range(2247, 2351))
        }
        
        group_importance = {}
        for group_name, indices in feature_groups.items():
            group_importance[group_name] = importance[indices].mean()
        
        # 繪製條形圖
        plt.figure(figsize=(10, 6))
        groups = list(group_importance.keys())
        values = list(group_importance.values())
        
        plt.bar(groups, values)
        plt.title('各特徵組平均重要性')
        plt.ylabel('SHAP 重要性')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return group_importance
    
    group_imp = plot_feature_importance_heatmap()
    
    return explainer, lime_explainer, shap_values, group_imp

# 執行解釋性分析
shap_explainer, lime_explainer, shap_vals, group_importance = model_interpretability()
```

## 工具與技術

### 核心工具清單

#### 1. Ember 原生工具
```bash
# 安裝 Ember
pip install ember-ml

# 下載資料集
python -c "import ember; ember.download_data()"

# 基本使用
python -c "import ember; print(ember.version)"
```

#### 2. 靜態分析工具

```python
# PE 檔案分析
import pefile
import pehash

def pe_analysis_tools():
    """PE 檔案分析工具示範"""
    
    # 使用 pefile 分析
    pe = pefile.PE('sample.exe')
    
    # 基本資訊
    print(f"入口點: 0x{pe.OPTIONAL_HEADER.AddressOfEntryPoint:08x}")
    print(f"映像基址: 0x{pe.OPTIONAL_HEADER.ImageBase:08x}")
    print(f"節區數量: {pe.FILE_HEADER.NumberOfSections}")
    
    # 匯入表分析
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            print(f"DLL: {entry.dll.decode()}")
            for imp in entry.imports:
                if imp.name:
                    print(f"  函數: {imp.name.decode()}")
    
    # 節區分析
    for section in pe.sections:
        print(f"節區: {section.Name.decode().strip()}")
        print(f"  虛擬大小: 0x{section.Misc_VirtualSize:08x}")
        print(f"  原始大小: 0x{section.SizeOfRawData:08x}")
        print(f"  熵值: {section.get_entropy():.4f}")
    
    # 計算各種雜湊值
    hashes = {
        'imphash': pe.get_imphash(),
        'md5': hashlib.md5(pe.__data__).hexdigest(),
        'sha256': hashlib.sha256(pe.__data__).hexdigest()
    }
    
    return pe, hashes

# YARA 規則整合
import yara

def yara_scanning():
    """YARA 規則掃描"""
    
    # 載入 YARA 規則
    rules = yara.compile(filepath='malware_rules.yar')
    
    # 掃描檔案
    matches = rules.match('sample.exe')
    
    for match in matches:
        print(f"規則: {match.rule}")
        print(f"標籤: {match.tags}")
        print(f"匹配字串: {match.strings}")
    
    return matches
```

#### 3. 機器學習工具

```python
# 進階機器學習模型
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def ensemble_learning():
    """集成學習方法"""
    
    # 基礎模型
    lgb_model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05)
    xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
    cat_model = CatBoostClassifier(n_estimators=1000, learning_rate=0.05, verbose=False)
    
    # 投票集成
    ensemble = VotingClassifier(
        estimators=[
            ('lightgbm', lgb_model),
            ('xgboost', xgb_model),
            ('catboost', cat_model)
        ],
        voting='soft'
    )
    
    return ensemble

# 深度學習模型
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def deep_learning_model():
    """深度學習惡意軟體檢測模型"""
    
    model = Sequential([
        Dense(1024, input_dim=2351, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

#### 4. 視覺化工具

```python
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def advanced_visualization():
    """進階視覺化分析"""
    
    # 載入資料
    X_train = ember.read_vectorized_features("train")
    y_train = ember.read_labels("train")
    
    # 1. 特徵分佈視覺化
    def plot_feature_distributions():
        # 選擇部分特徵進行視覺化
        feature_indices = [0, 10, 72, 1352, 1480, 1735, 1991, 2247]
        feature_names = ember.get_feature_names()
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i, idx in enumerate(feature_indices):
            benign_values = X_train[y_train == 0, idx]
            malicious_values = X_train[y_train == 1, idx]
            
            axes[i].hist(benign_values, bins=50, alpha=0.7, label='Benign', density=True)
            axes[i].hist(malicious_values, bins=50, alpha=0.7, label='Malicious', density=True)
            axes[i].set_title(feature_names[idx][:30])
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()
    
    # 2. t-SNE 降維視覺化
    def plot_tsne():
        from sklearn.manifold import TSNE
        
        # 隨機選擇樣本以加快計算
        sample_size = 10000
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[indices]
        y_sample = y_train[indices]
        
        # 移除未標記樣本
        labeled_mask = y_sample != -1
        X_labeled = X_sample[labeled_mask]
        y_labeled = y_sample[labeled_mask]
        
        # t-SNE 降維
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_labeled)
        
        # 繪製散點圖
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_labeled, 
                            cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, label='Label')
        plt.title('t-SNE 視覺化 (紅色=惡意軟體, 藍色=良性軟體)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()
    
    # 3. 互動式視覺化
    def interactive_visualization():
        # 使用 Plotly 建立互動式圖表
        sample_size = 5000
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[indices]
        y_sample = y_train[indices]
        
        labeled_mask = y_sample != -1
        X_labeled = X_sample[labeled_mask]
        y_labeled = y_sample[labeled_mask]
        
        # 選擇前三個主成分
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_labeled)
        
        # 3D 散點圖
        fig = go.Figure(data=go.Scatter3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1], 
            z=X_pca[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=y_labeled,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="標籤")
            ),
            text=[f"樣本 {i}: {'惡意' if y_labeled[i] else '良性'}" 
                  for i in range(len(y_labeled))],
            hovertemplate='%{text}<br>PC1: %{x}<br>PC2: %{y}<br>PC3: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Ember 資料集 3D PCA 視覺化',
            scene=dict(
                xaxis_title='第一主成分',
                yaxis_title='第二主成分',
                zaxis_title='第三主成分'
            )
        )
        
        fig.show()
    
    plot_feature_distributions()
    plot_tsne()
    interactive_visualization()

# 執行視覺化
advanced_visualization()
```

### 自動化分析流水線

```python
import json
import logging
from datetime import datetime
from pathlib import Path

class EmberAnalysisPipeline:
    """Ember 分析自動化流水線"""
    
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.model = None
        self.scaler = None
        self.feature_selector = None
    
    def load_config(self, config_path):
        """載入配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def setup_logging(self):
        """設置日誌記錄"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ember_analysis.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def preprocess_data(self):
        """資料預處理"""
        self.logger.info("開始資料預處理...")
        
        # 載入原始資料
        X_train = ember.read_vectorized_features("train")
        y_train = ember.read_labels("train")
        X_test = ember.read_vectorized_features("test")
        y_test = ember.read_labels("test")
        
        # 移除未標記資料
        train_mask = y_train != -1
        test_mask = y_test != -1
        
        self.X_train = X_train[train_mask]
        self.y_train = y_train[train_mask]
        self.X_test = X_test[test_mask]
        self.y_test = y_test[test_mask]
        
        self.logger.info(f"訓練集大小: {self.X_train.shape}")
        self.logger.info(f"測試集大小: {self.X_test.shape}")
    
    def feature_engineering(self):
        """特徵工程"""
        self.logger.info("開始特徵工程...")
        
        # 特徵縮放
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # 特徵選擇
        from sklearn.feature_selection import SelectKBest, f_classif
        self.feature_selector = SelectKBest(
            score_func=f_classif, 
            k=self.config['feature_selection']['k']
        )
        
        self.X_train_processed = self.feature_selector.fit_transform(
            X_train_scaled, self.y_train
        )
        self.X_test_processed = self.feature_selector.transform(X_test_scaled)
        
        self.logger.info(f"特徵選擇後維度: {self.X_train_processed.shape[1]}")
    
    def train_model(self):
        """訓練模型"""
        self.logger.info("開始模型訓練...")
        
        model_type = self.config['model']['type']
        
        if model_type == 'lightgbm':
            import lightgbm as lgb
            
            params = self.config['model']['lightgbm_params']
            train_data = lgb.Dataset(self.X_train_processed, label=self.y_train)
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.config['model']['num_boost_round'],
                valid_sets=[train_data],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
        elif model_type == 'ensemble':
            from sklearn.ensemble import VotingClassifier
            import lightgbm as lgb
            from xgboost import XGBClassifier
            
            lgb_model = lgb.LGBMClassifier(**self.config['model']['lightgbm_params'])
            xgb_model = XGBClassifier(**self.config['model']['xgboost_params'])
            
            self.model = VotingClassifier(
                estimators=[
                    ('lightgbm', lgb_model),
                    ('xgboost', xgb_model)
                ],
                voting='soft'
            )
            
            self.model.fit(self.X_train_processed, self.y_train)
        
        self.logger.info("模型訓練完成")
    
    def evaluate_model(self):
        """模型評估"""
        self.logger.info("開始模型評估...")
        
        from sklearn.metrics import classification_report, roc_auc_score
        
        # 預測
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(self.X_test_processed)[:, 1]
        else:
            y_pred_proba = self.model.predict(
                self.X_test_processed, 
                num_iteration=self.model.best_iteration
            )
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 計算指標
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        report = classification_report(
            self.y_test, y_pred, 
            target_names=['Benign', 'Malicious'],
            output_dict=True
        )
        
        # 記錄結果
        self.logger.info(f"AUC 分數: {auc_score:.4f}")
        self.logger.info(f"準確率: {report['accuracy']:.4f}")
        self.logger.info(f"精確率: {report['Malicious']['precision']:.4f}")
        self.logger.info(f"召回率: {report['Malicious']['recall']:.4f}")
        
        # 保存結果
        results = {
            'timestamp': datetime.now().isoformat(),
            'auc_score': auc_score,
            'classification_report': report,
            'config': self.config
        }
        
        with open('evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def save_model(self, model_path="trained_model"):
        """保存模型"""
        import joblib
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler, 
            'feature_selector': self.feature_selector,
            'config': self.config
        }
        
        joblib.dump(model_data, f"{model_path}.pkl")
        self.logger.info(f"模型已保存至 {model_path}.pkl")
    
    def run_pipeline(self):
        """執行完整分析流水線"""
        try:
            self.preprocess_data()
            self.feature_engineering()
            self.train_model()
            results = self.evaluate_model()
            self.save_model()
            
            self.logger.info("分析流水線執行完成！")
            return results
            
        except Exception as e:
            self.logger.error(f"流水線執行出錯: {str(e)}")
            raise

# 配置文件範例
config_example = {
    "feature_selection": {
        "k": 1000
    },
    "model": {
        "type": "lightgbm",
        "num_boost_round": 1000,
        "lightgbm_params": {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 1024,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1
        },
        "xgboost_params": {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "random_state": 42
        }
    }
}

# 保存配置文件
with open('config.json', 'w', encoding='utf-8') as f:
    json.dump(config_example, f, ensure_ascii=False, indent=2)

# 執行流水線
pipeline = EmberAnalysisPipeline()
results = pipeline.run_pipeline()
```

## 高級分析技術

### 1. 評估指標與統計分析

```python
def comprehensive_evaluation():
    """全面的評估框架"""
    
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, matthews_corrcoef,
        confusion_matrix, classification_report
    )
    
    def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
        """計算所有評估指標"""
        
        metrics = {}
        
        # 基本分類指標
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # 機率相關指標
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['ap'] = average_precision_score(y_true, y_pred_proba)
        
        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        # 計算更多指標
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp'])
        metrics['npv'] = metrics['tn'] / (metrics['tn'] + metrics['fn'])  # 負預測值
        metrics['fpr'] = metrics['fp'] / (metrics['fp'] + metrics['tn'])  # 假陽性率
        metrics['fnr'] = metrics['fn'] / (metrics['fn'] + metrics['tp'])  # 假陰性率
        
        return metrics
    
    def bootstrap_confidence_intervals(y_true, y_pred, y_pred_proba=None, 
                                     n_bootstrap=1000, confidence=0.95):
        """計算指標的信賴區間"""
        
        n_samples = len(y_true)
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            # 自助法採樣
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_pred_proba_boot = y_pred_proba[indices] if y_pred_proba is not None else None
            
            # 計算指標
            metrics = calculate_all_metrics(y_true_boot, y_pred_boot, y_pred_proba_boot)
            bootstrap_metrics.append(metrics)
        
        # 計算信賴區間
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = {}
        for metric in bootstrap_metrics[0].keys():
            values = [m[metric] for m in bootstrap_metrics]
            ci_lower = np.percentile(values, lower_percentile)
            ci_upper = np.percentile(values, upper_percentile)
            confidence_intervals[metric] = (ci_lower, ci_upper)
        
        return confidence_intervals
    
    def statistical_significance_test(results_a, results_b, metric='accuracy'):
        """統計顯著性測試"""
        
        from scipy import stats
        
        # 提取指標值
        values_a = [r[metric] for r in results_a]
        values_b = [r[metric] for r in results_b]
        
        # 配對 t 檢驗
        t_stat, p_value = stats.ttest_rel(values_a, values_b)
        
        # Wilcoxon 符號秩檢驗（非參數）
        w_stat, w_p_value = stats.wilcoxon(values_a, values_b)
        
        return {
            't_test': {'statistic': t_stat, 'p_value': p_value},
            'wilcoxon': {'statistic': w_stat, 'p_value': w_p_value},
            'effect_size': (np.mean(values_a) - np.mean(values_b)) / np.sqrt(
                (np.var(values_a) + np.var(values_b)) / 2
            )
        }
    
    return calculate_all_metrics, bootstrap_confidence_intervals, statistical_significance_test

# 實際使用範例
eval_functions = comprehensive_evaluation()
calculate_all_metrics, bootstrap_ci, significance_test = eval_functions

# 載入測試結果
X_test = ember.read_vectorized_features("test")
y_test = ember.read_labels("test")
model = ember.load_model()

# 預測
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# 移除未標記樣本
mask = y_test != -1
y_test_clean = y_test[mask]
y_pred_clean = y_pred[mask]
y_pred_proba_clean = y_pred_proba[mask]

# 計算指標
metrics = calculate_all_metrics(y_test_clean, y_pred_clean, y_pred_proba_clean)
print("評估指標:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# 計算信賴區間
ci = bootstrap_ci(y_test_clean, y_pred_clean, y_pred_proba_clean)
print("\n95% 信賴區間:")
for metric, (lower, upper) in ci.items():
    if metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"  {metric}: [{lower:.4f}, {upper:.4f}]")
```

### 3. 實驗記錄和管理

```python
class ExperimentTracker:
    """實驗追蹤和管理系統"""
    
    def __init__(self, project_name="ember_experiments"):
        self.project_name = project_name
        self.experiment_id = self.generate_experiment_id()
        self.results_dir = Path(f"experiments/{self.experiment_id}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_log = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'config': {},
            'metrics': {},
            'artifacts': []
        }
    
    def generate_experiment_id(self):
        """生成實驗ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}"
    
    def log_config(self, config):
        """記錄實驗配置"""
        self.experiment_log['config'] = config
        
        # 保存配置文件
        config_path = self.results_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        self.experiment_log['artifacts'].append(str(config_path))
    
    def log_metrics(self, metrics, step=None):
        """記錄評估指標"""
        if step is not None:
            if 'step_metrics' not in self.experiment_log:
                self.experiment_log['step_metrics'] = {}
            self.experiment_log['step_metrics'][step] = metrics
        else:
            self.experiment_log['metrics'].update(metrics)
    
    def log_model(self, model, model_name="model"):
        """記錄模型"""
        import joblib
        
        model_path = self.results_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        
        self.experiment_log['artifacts'].append(str(model_path))
        return model_path
    
    def log_figure(self, fig, name):
        """記錄圖表"""
        fig_path = self.results_dir / f"{name}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        self.experiment_log['artifacts'].append(str(fig_path))
        return fig_path
    
    def log_data(self, data, name, format='npy'):
        """記錄資料"""
        if format == 'npy':
            data_path = self.results_dir / f"{name}.npy"
            np.save(data_path, data)
        elif format == 'csv':
            data_path = self.results_dir / f"{name}.csv"
            data.to_csv(data_path, index=False)
        elif format == 'json':
            data_path = self.results_dir / f"{name}.json"
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.experiment_log['artifacts'].append(str(data_path))
        return data_path
    
    def save_experiment(self):
        """保存實驗記錄"""
        log_path = self.results_dir / "experiment_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, ensure_ascii=False, indent=2)
        
        return log_path
    
    def load_experiment(self, experiment_id):
        """載入實驗記錄"""
        log_path = Path(f"experiments/{experiment_id}/experiment_log.json")
        with open(log_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compare_experiments(self, experiment_ids, metric='accuracy'):
        """比較多個實驗"""
        results = []
        
        for exp_id in experiment_ids:
            exp_log = self.load_experiment(exp_id)
            results.append({
                'experiment_id': exp_id,
                'timestamp': exp_log['timestamp'],
                'metric_value': exp_log['metrics'].get(metric, None),
                'config': exp_log['config']
            })
        
        # 排序
        results.sort(key=lambda x: x['metric_value'] or 0, reverse=True)
        
        return results

# 使用範例
tracker = ExperimentTracker("ember_malware_detection")

# 記錄配置
config = {
    "model": "lightgbm",
    "features": "all_2351",
    "preprocessing": "robust_scaler",
    "feature_selection": "top_1000"
}
tracker.log_config(config)

# 記錄指標
metrics = {
    "accuracy": 0.9234,
    "precision": 0.9187,
    "recall": 0.9289,
    "f1": 0.9238,
    "auc": 0.9756
}
tracker.log_metrics(metrics)

# 記錄模型和結果
# tracker.log_model(trained_model, "lightgbm_model")
# tracker.log_figure(confusion_matrix_plot, "confusion_matrix")

# 保存實驗
log_path = tracker.save_experiment()
print(f"實驗記錄保存至: {log_path}")
```

## 技術參考

### 核心技術參數

#### 1. 模型超參數配置

```python
# LightGBM 最佳參數
LIGHTGBM_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 1024,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1
}

# 特徵選擇參數
FEATURE_SELECTION = {
    'method': 'mutual_info_classif',
    'k_best': 1000,
    'variance_threshold': 0.01
}
```

#### 2. 性能基準

```python
# Ember 官方基準結果
EMBER_BENCHMARKS = {
    'LightGBM': {
        'accuracy': 0.8958,
        'precision': 0.8951,
        'recall': 0.8965,
        'f1_score': 0.8958,
        'auc_score': 0.9761
    },
    'RandomForest': {
        'accuracy': 0.8834,
        'precision': 0.8798,
        'recall': 0.8871,
        'f1_score': 0.8834,
        'auc_score': 0.9572
    }
}
```

#### 3. 技術限制與注意事項

- **資料依賴性**：模型效能受訓練資料時間範圍影響
- **特徵漂移**：惡意軟體演化可能導致特徵分佈變化  
- **對抗性攻擊**：需要定期評估模型對抗攻擊的抗性
- **計算資源**：完整訓練需要16GB+ RAM和多核CPU

### 核心惡意軟體分析概念

#### 靜態分析基礎技術

```python
# 惡意軟體特徵識別
malware_indicators = {
    'packed_executables': {
        'entropy_threshold': 7.0,  # 高熵值表示可能被打包
        'section_names': ['.upx', '.aspack', '.petite'],
        'import_anomalies': 'few_imports_high_entropy'
    },
    'suspicious_apis': {
        'process_manipulation': ['CreateProcess', 'VirtualAlloc', 'WriteProcessMemory'],
        'file_operations': ['CreateFile', 'WriteFile', 'MoveFile', 'DeleteFile'],
        'registry_access': ['RegCreateKey', 'RegSetValue', 'RegDeleteKey'],
        'network_activity': ['socket', 'connect', 'send', 'recv', 'InternetOpen'],
        'anti_analysis': ['IsDebuggerPresent', 'GetTickCount', 'QueryPerformanceCounter']
    },
    'pe_anomalies': {
        'unusual_entry_point': 'entry_point_in_writable_section',
        'section_characteristics': 'executable_writable_sections',
        'timestamp_anomalies': 'future_or_null_timestamps'
    }
}
```

#### 惡意軟體家族特徵模式

```python
# 常見惡意軟體家族的 Ember 特徵模式
family_patterns = {
    'trojans': {
        'high_import_features': ['kernel32.dll', 'advapi32.dll', 'user32.dll'],
        'suspicious_strings': ['\\System32\\', 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run'],
        'section_entropy': 'mixed_high_low_entropy'
    },
    'ransomware': {
        'crypto_apis': ['CryptGenKey', 'CryptEncrypt', 'CryptAcquireContext'],
        'file_extensions': ['.docx', '.pdf', '.jpg', '.mp3'],
        'string_patterns': ['your files have been encrypted', 'bitcoin']
    },
    'backdoors': {
        'network_apis': ['WSAStartup', 'socket', 'bind', 'listen'],
        'persistence_mechanisms': ['CreateService', 'RegSetValue'],
        'process_apis': ['CreateProcess', 'ShellExecute']
    },
    'info_stealers': {
        'browser_paths': ['\\Chrome\\', '\\Firefox\\', '\\Edge\\'],
        'credential_apis': ['LsaEnumerateLogonSessions', 'CredEnumerate'],
        'file_search': ['FindFirstFile', 'FindNextFile']
    }
}
```

#### 進階分析技術

```python
def advanced_malware_analysis(ember_features, pe_file_path):
    """進階惡意軟體分析技術"""
    
    import pefile
    import math
    
    # 1. PE 結構分析
    pe = pefile.PE(pe_file_path)
    
    # 檢查打包跡象
    def detect_packing():
        packing_indicators = {
            'high_entropy_sections': [],
            'suspicious_section_names': [],
            'import_anomalies': False
        }
        
        for section in pe.sections:
            entropy = section.get_entropy()
            section_name = section.Name.decode().strip('\x00')
            
            if entropy > 7.0:
                packing_indicators['high_entropy_sections'].append({
                    'name': section_name,
                    'entropy': entropy,
                    'size': section.SizeOfRawData
                })
            
            # 檢查已知打包程式的節區名稱
            known_packers = ['UPX', 'ASPack', 'PEtite', 'MPRESS']
            if any(packer.lower() in section_name.lower() for packer in known_packers):
                packing_indicators['suspicious_section_names'].append(section_name)
        
        # 檢查匯入表異常（打包後匯入函數通常很少）
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            import_count = sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT)
            if import_count < 10:  # 正常程式通常有更多匯入
                packing_indicators['import_anomalies'] = True
        
        return packing_indicators
    
    # 2. 行為模式分析
    def analyze_behavior_patterns():
        # 基於 Ember 匯入特徵分析行為
        import_features = ember_features[72:1352]  # 匯入特徵索引範圍
        
        behavior_score = {
            'file_manipulation': 0,
            'registry_access': 0,
            'network_activity': 0,
            'process_control': 0,
            'anti_analysis': 0
        }
        
        # 這裡需要映射 Ember 特徵索引到具體 API
        # 實際實作需要 Ember 的特徵名稱映射表
        
        return behavior_score
    
    # 3. 統計異常檢測
    def statistical_anomaly_detection():
        anomalies = {
            'feature_outliers': [],
            'suspicious_patterns': []
        }
        
        # 檢查特徵值異常
        for i, feature_value in enumerate(ember_features):
            if feature_value > 10:  # 假設正常值範圍
                anomalies['feature_outliers'].append({
                    'feature_index': i,
                    'value': feature_value,
                    'category': get_feature_category(i)
                })
        
        return anomalies
    
    # 4. 時間序列分析（如果有時間戳資訊）
    def temporal_analysis():
        if hasattr(pe, 'FILE_HEADER'):
            timestamp = pe.FILE_HEADER.TimeDateStamp
            
            # 檢查時間戳異常
            from datetime import datetime
            compile_time = datetime.fromtimestamp(timestamp)
            current_time = datetime.now()
            
            temporal_indicators = {
                'compile_time': compile_time,
                'is_future': compile_time > current_time,
                'is_null': timestamp == 0,
                'is_suspicious': timestamp < 946684800  # 2000年之前
            }
            
            return temporal_indicators
        return None
    
    # 執行所有分析
    analysis_results = {
        'packing_analysis': detect_packing(),
        'behavior_patterns': analyze_behavior_patterns(),
        'statistical_anomalies': statistical_anomaly_detection(),
        'temporal_analysis': temporal_analysis()
    }
    
    return analysis_results

def get_feature_category(feature_index):
    """根據特徵索引返回特徵類別"""
    if feature_index < 10:
        return 'general'
    elif feature_index < 72:
        return 'header'
    elif feature_index < 1352:
        return 'imports'
    elif feature_index < 1480:
        return 'exports'
    elif feature_index < 1735:
        return 'sections'
    elif feature_index < 1991:
        return 'byte_histogram'
    elif feature_index < 2247:
        return 'byte_entropy'
    else:
        return 'strings'
```

---

**技術聲明**: 本技術指南專注於 Ember 資料集的防禦性惡意軟體分析技術實作。