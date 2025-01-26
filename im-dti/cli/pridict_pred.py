import pandas as pd
import joblib
from scipy.stats import pearsonr
from tdc.benchmark_group import dti_dg_group
from sklearn.metrics import average_precision_score
import numpy as np

# 创建基准组
dg_group = dti_dg_group(path='datasets')
dg_benchmark = dg_group.get("bindingdb_patent")

# 用于存储多次运行的预测结果
predictions_list = []

# 获取基准集名称
name = dg_benchmark['name']

# 加载数据集文件
data = pd.read_csv('C:/Users/Tao/Desktop/ConPLex-main/sum/tdc_test.txt')

# 准备特征数据
X = data[['esm_pred', 'pt50_pred', 'probert_pred']]

# 进行至少 5 次独立运行
for run in range(5):
    # 添加随机种子以确保每次运行的随机性
    np.random.seed(run)

    # 动态加载不同的模型
    model_path = f'C:/Users/Tao/Desktop/ConPLex-main/sum/tdc_beiyes_model_run_{run + 1}.pkl'
    best_model = joblib.load(model_path)

    # 使用模型进行预测
    prediction = best_model.predict(X)

    # 提取真实标签
    y_true = data['label'].apply(lambda x: float(x))  # 确保标签为浮点数

    # 计算皮尔逊相关系数 (PCC)
    pcc, _ = pearsonr(y_true, prediction)  # 计算 PCC

    # 将当前运行的预测结果存储到字典中
    predictions = {name: prediction}

    # 将当前预测结果添加到预测列表
    predictions_list.append(predictions)

    # 输出当前运行的预测结果和 PCC
    for i, pred in enumerate(prediction):
        print(f"Run {run + 1} - Prediction for sample {i + 1}: {pred}, True label: {y_true.iloc[i]}")

    print(f"Run {run + 1} - Pearson Correlation Coefficient (PCC): {pcc}")

# 使用 tdc 进行评估
results = dg_group.evaluate_many(predictions_list)

# 保存评估结果到文本文件
with open('C:/Users/Tao/Desktop/ConPLex-main/sum/evaluation_results1.txt', 'w') as f:
    f.write(str(results))

# 计算并输出 AUPR
# 注意：这里的 y_true 应该是在循环外部定义的真实标签
aupr = average_precision_score(y_true, prediction)  # 计算AUPR
print(f"Area Under Precision-Recall Curve (AUPR): {aupr}")
