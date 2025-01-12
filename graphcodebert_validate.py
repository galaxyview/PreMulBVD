import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm  # 导入tqdm库显示进度条

from graphcodebert_model import CodeDataset, GraphCodeBERTFineTuner

# 加载GraphCodeBERT模型和分词器
tokenizer = RobertaTokenizer.from_pretrained("pretrained_model/contrabert-base")
graphcodebert = RobertaModel.from_pretrained("pretrained_model/contrabert-base")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
# <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->
data = torch.load('')
checkpoint_path = ''
# <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->

def evaluate(model, val_loader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc="Validating") as pbar:
            for ast_input, code_input, label in val_loader:
                # 将数据移动到 GPU
                ast_input = {key: value.to(device) for key, value in ast_input.items()}
                code_input = {key: value.to(device) for key, value in code_input.items()}
                label = label.to(device)

                outputs = model(ast_input, code_input)
                preds.extend(outputs.cpu().numpy())
                true_labels.extend(label.cpu().numpy())
                pbar.update(1)
    
    preds = np.array(preds).flatten()
    true_labels = np.array(true_labels).flatten()

    for th in range(0,100):
        threshold = th * 0.01

        # 计算指标
        acc = accuracy_score(true_labels, preds > threshold)
        f1 = f1_score(true_labels, preds > threshold)
        auc = roc_auc_score(true_labels, preds)
        precision = precision_score(true_labels, preds > threshold)
        recall = recall_score(true_labels, preds > threshold)
        print(f'Validation: Threshold:{threshold}, Accuracy: {acc}, F1: {f1}, AUC: {auc}, precision: {precision}, recall: {recall}')

def validate():
    # 示例数据格式:
    # [{'ast': 'AST 内容', 'code': '代码片段', 'label': 1}, {...}, ...]

    # 提取 AST, CODE, LABEL 列表
    ast_data = [item['ast'] for item in data]
    code_data = [item['pseudocode'] for item in data]
    labels = [item['label'] for item in data]


    val_dataset = CodeDataset(ast_data, code_data, labels, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GraphCodeBERTFineTuner(graphcodebert).to(device)

    # 初始化模型、优化器和损失函数(checkpoint启用)-----------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluate(model, val_loader, threshold)

def dataparallel_validate():
    # 示例数据格式:
    # [{'ast': 'AST 内容', 'code': '代码片段', 'label': 1}, {...}, ...]

    # 提取 AST, CODE, LABEL 列表
    ast_data = [item['ast'] for item in data]
    code_data = [item['pseudocode'] for item in data]
    labels = [item['label'] for item in data]

    # 按 8:2 比例分割数据
    train_ast_data, val_ast_data, train_code_data, val_code_data, train_labels, val_labels = train_test_split(
        ast_data, code_data, labels, test_size=0.2, random_state=42
    )

    val_dataset = CodeDataset(val_ast_data, val_code_data, val_labels, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GraphCodeBERTFineTuner(graphcodebert).to(device)
    model = DataParallel(model)  # 将模型分布到多个GPU

    # 初始化模型、优化器和损失函数(checkpoint启用)-----------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluate(model, val_loader)

if __name__== '__main__':
    dataparallel_validate()