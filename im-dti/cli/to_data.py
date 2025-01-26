import pandas as pd
import ast
import re
from pathlib import Path

def combine_predictions(esm_file_path, pt50_file_path, probert_file_path, output_file_path):
    # 读取 CSV 文件
    esm_df = pd.read_csv(esm_file_path)
    pt50_df = pd.read_csv(pt50_file_path)
    probert_df = pd.read_csv(probert_file_path)

    # 创建一个空列表来存储合并后的预测和标签
    combined_predictions = []

    # 假设每个模型的预测数量相同
    for esm_row, pt50_row, probert_row in zip(esm_df.iterrows(), pt50_df.iterrows(), probert_df.iterrows()):
        esm_index, esm_row = esm_row
        pt50_index, pt50_row = pt50_row
        probert_index, probert_row = probert_row

        # 将每个模型的预测值转换为列表
        esm_preds = ast.literal_eval(esm_row['pred'])
        pt50_preds = ast.literal_eval(pt50_row['pred'])
        probert_preds = ast.literal_eval(probert_row['pred'])

        # 提取 esm 标签，假设标签是字符串格式的 tensor
        esm_label_str = esm_row['label']  # 获取原始标签字符串
        esm_labels = re.findall(r"[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d*\.\d+|[-+]?\d+", esm_label_str)  # 提取所有数值，包括科学计数法

        # 合并每个模型的预测值和对应的标签
        for i in range(len(esm_preds)):
            label_value = esm_labels[i] if i < len(esm_labels) else None  # 确保索引不超出范围
            combined_predictions.append({
                'esm_pred': esm_preds[i],
                'pt50_pred': pt50_preds[i],
                'probert_pred': probert_preds[i],
                'label': label_value  # 提取的标签值
            })

    # 创建 DataFrame
    predictions_df = pd.DataFrame(combined_predictions)

    # 保存到 CSV 文件
    predictions_df.to_csv(output_file_path, index=False)

    print(f"Combined predictions saved to {output_file_path}")

def combine_space_predictions(drug_file_path, target_file_path, label_file_path, output_file_path):
    # 读取 CSV 文件
    drug_df = pd.read_csv(drug_file_path)
    target_df = pd.read_csv(target_file_path)
    label_df = pd.read_csv(label_file_path)

    # 创建一个空列表来存储合并后的预测和标签
    combined_predictions = []

    # 假设每个模型的预测数量相同
    for drug_row, target_row, label_row in zip(drug_df.iterrows(), target_df.iterrows(), label_df.iterrows()):
        drug_index, drug_row = drug_row
        target_index, target_row = target_row
        label_index, label_row = label_row

        # 将每个模型的预测值转换为列表
        drug_preds = ast.literal_eval(drug_row['pred'])
        target_preds = ast.literal_eval(target_row['pred'])
        
        # 提取标签
        label_str = label_row['label']  # 获取原始标签字符串
        labels = re.findall(r"[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d*\.\d+|[-+]?\d+", label_str)  # 提取所有数值，包括科学计数法

        # 合并每个模型的预测值和对应的标签
        for i in range(len(drug_preds)):
            label_value = labels[i] if i < len(labels) else None  # 确保索引不超出范围
            combined_predictions.append({
                'drug_pred': drug_preds[i],
                'target_pred': target_preds[i],
                'label': label_value  # 提取的标签值
            })

    # 创建 DataFrame
    predictions_df = pd.DataFrame(combined_predictions)

    # 保存到 CSV 文件
    predictions_df.to_csv(output_file_path, index=False)

    print(f"Combined predictions saved to {output_file_path}")


import pandas as pd
import ast
import re

def combine_predictions2(esm_file_path, probert_file_path, output_file_path):
    # 读取 CSV 文件
    esm_df = pd.read_csv(esm_file_path)
    probert_df = pd.read_csv(probert_file_path)

    # 创建一个空列表来存储合并后的预测和标签
    combined_predictions = []

    # 假设每个模型的预测数量相同
    for esm_row, probert_row in zip(esm_df.iterrows(), probert_df.iterrows()):
        esm_index, esm_row = esm_row
        probert_index, probert_row = probert_row

        # 将每个模型的预测值转换为列表
        esm_preds = ast.literal_eval(esm_row['pred'])
        probert_preds = ast.literal_eval(probert_row['pred'])

        # 提取 esm 标签，假设标签是字符串格式的 tensor
        esm_label_str = esm_row['label']  # 获取原始标签字符串
        esm_labels = re.findall(r"[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d*\.\d+|[-+]?\d+", esm_label_str)  # 提取所有数值，包括科学计数法

        # 合并每个模型的预测值和对应的标签
        for i in range(len(esm_preds)):
            label_value = esm_labels[i] if i < len(esm_labels) else None  # 确保索引不超出范围
            combined_predictions.append({
                'esm_pred': esm_preds[i],
                'probert_pred': probert_preds[i],
                'label': label_value  # 提取的标签值
            })

    # 创建 DataFrame
    predictions_df = pd.DataFrame(combined_predictions)

    # 保存到 CSV 文件
    predictions_df.to_csv(output_file_path, index=False)

    print(f"Combined predictions saved to {output_file_path}")

if __name__ == "__main__":
    probert_file_path=Path(f"/home/liujin/data/ConPLex-main/sum/tdc_proberttest.txt")
    esm_file_path=Path(f"/home/liujin/data/ConPLex-main/sum/tdc_esmtest.txt")
    pt50_file_path=Path(f"/home/liujin/data/ConPLex-main/sum/tdc_pt50test.txt")
    output_file_path=Path(f"/home/liujin/data/ConPLex-main/sum/tdc_test.txt")
    combine_predictions(esm_file_path, pt50_file_path,probert_file_path,output_file_path)

