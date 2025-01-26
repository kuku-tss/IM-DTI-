import pandas as pd
import ast  # 用于将字符串转换为列表
import torch  # 确保导入 PyTorch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
from scipy.stats import pearsonr
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from pathlib import Path
from sklearn.metrics import average_precision_score


def train_random_forest_model(train_output_path,tree_path):
    # 1. 加载数据
    data = pd.read_csv(train_output_path)  # 使用合并后的 CSV 文件

    # 检查缺失值并去掉包含缺失值的行
    if data.isnull().any().any():
        print("数据中存在缺失值，去掉包含缺失值的行。")
        data = data.dropna()

    # 2. 解析数据
    # 由于预测值已经是列表形式，直接使用
    # 这里假设 'label' 列的值是可以直接使用的
    data['label'] = data['label'].apply(lambda x: float(x))  # 确保标签为浮点数

    # 3. 构建特征集
    # 将三个模型的预测值合并为一个 DataFrame
    X = data[['esm_pred', 'pt50_pred', 'probert_pred']]
    # X = data[['esm_pred', 'pt50_pred']]

    # 真实标签
    y = data['label']  # 直接使用标签列

    # 4. 划分数据集（可选，若希望在训练和测试集上评估性能）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建随机森林模型
    model = RandomForestRegressor(random_state=42)

    # 定义搜索空间
    search_space = {
        'n_estimators': Integer(50, 400),
        'max_depth': Integer(1, 25),
        'min_samples_split': Integer(2, 15),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical(['sqrt', 'log2']),  # 修正这里
        'bootstrap': Categorical([True, False])
    }

    # 使用贝叶斯优化进行超参数搜索
    bayes_search = BayesSearchCV(model, search_space, n_iter=50, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    bayes_search.fit(X_train, y_train)

    # 输出最佳参数
    print("Best Parameters:", bayes_search.best_params_)

    # 5. 使用最佳参数训练模型
    best_model = bayes_search.best_estimator_
    best_model.fit(X_train, y_train)  # 在训练集上训练最佳模型

    # 6. 保存模型
    # 6. 保存模型
    joblib.dump(best_model, tree_path)

    # 如果使用 pickle
    # import pickle
    # with open('best_random_forest_model.pkl', 'wb') as f:
    #     pickle.dump(best_model, f)

    # 7. 可选：在测试集上评估性能
    test_score = best_model.score(X_test, y_test)
    print(f"Test Score (R^2): {test_score}")

    # 7. 可选：在测试集上评估性能
    y_pred = best_model.predict(X_test)  # 预测测试集

    # 8. 计算皮尔逊相关系数 (PCC)
    pcc, _ = pearsonr(y_test, y_pred)  # 计算 PCC
    print(f"Pearson Correlation Coefficient (PCC): {pcc}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import joblib
from scipy.stats import pearsonr

def train_random_forest_model2(train_output_path,tree_path):
    # 1. 加载数据
    data = pd.read_csv(train_output_path)  # 使用合并后的 CSV 文件

    # 检查缺失值并去掉包含缺失值的行
    if data.isnull().any().any():
        print("数据中存在缺失值，去掉包含缺失值的行。")
        data = data.dropna()

    # 2. 解析数据
    # 确保标签为浮点数
    data['label'] = data['label'].apply(lambda x: float(x))

    # 3. 构建特征集
    # 只使用 esm_pred 和 probert_pred
    X = data[['esm_pred', 'probert_pred']]
    
    # 真实标签
    y = data['label']  # 直接使用标签列

    # 4. 划分数据集（可选，若希望在训练和测试集上评估性能）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建随机森林模型
    model = RandomForestRegressor(random_state=42)

    # 定义搜索空间
    search_space = {
        'n_estimators': Integer(50, 400),
        'max_depth': Integer(1, 25),
        'min_samples_split': Integer(2, 15),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical(['sqrt', 'log2']),
        'bootstrap': Categorical([True, False])
    }

    # 使用贝叶斯优化进行超参数搜索
    bayes_search = BayesSearchCV(model, search_space, n_iter=50, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    bayes_search.fit(X_train, y_train)

    # 输出最佳参数
    print("Best Parameters:", bayes_search.best_params_)

    # 5. 使用最佳参数训练模型
    best_model = bayes_search.best_estimator_
    best_model.fit(X_train, y_train)  # 在训练集上训练最佳模型

    # 6. 保存模型
    joblib.dump(best_model, tree_path)

    # 7. 可选：在测试集上评估性能
    test_score = best_model.score(X_test, y_test)
    print(f"Test Score (R^2): {test_score}")

    # 8. 预测测试集
    y_pred = best_model.predict(X_test)

    # 9. 计算皮尔逊相关系数 (PCC)
    pcc, _ = pearsonr(y_test, y_pred)  # 计算 PCC
    print(f"Pearson Correlation Coefficient (PCC): {pcc}")

if __name__ == "__main__":
    train_output_path=Path(f"/home/liujin/data/ConPLex-main/sum/bindingdbtrain.txt")
    tree_path=Path(f"/home/liujin/data/ConPLex-main/sum/bindingdb_beiyes_model.pkl")
    train_random_forest_model(train_output_path,tree_path)
