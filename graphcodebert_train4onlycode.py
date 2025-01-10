import torch
import os
import json
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaModel,RobertaForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 导入tqdm库显示进度条

# 加载GraphCodeBERT模型和分词器
tokenizer = RobertaTokenizer.from_pretrained("pretrained_model/graphcodebert-base")
graphcodebert = RobertaModel.from_pretrained("pretrained_model/graphcodebert-base")
# 指定使用 GPU 
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 1e-5
# 加载 .pth 文件
data = torch.load("datasets/pth/Juliet/CWE_codeonly_mixed.pth")
checkpoint_path = 'checkpoints/finetune_onlycode/mixed_batch64/CWE_mixed_epoch_14.pt'

# <------------------------------微调模型---------------------------------->
# 自定义数据集类
class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data[idx]['pseudocode']
        label = int(self.data[idx]['label'])
        
        code_encoding = self.tokenizer.encode_plus(
            code, padding='max_length', truncation=True, return_tensors='pt'
        )

        return code_encoding, torch.tensor(label)
    
class GraphCodeBERTFineTuner4code(nn.Module):
    def __init__(self, graphcodebert):
        super(GraphCodeBERTFineTuner4code, self).__init__()
        self.graphcodebert = graphcodebert
        
        # 分类器，用于将GraphCodeBERT的输出进行处理并得到分类结果
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),  # 768 是GraphCodeBERT的输出向量维度
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 假设是二分类任务
        )

    def forward(self, code_input):
        # 将输入的形状从 [batch_size, 1, sequence_length] 变为 [batch_size, sequence_length]
        code_input['input_ids'] = code_input['input_ids'].squeeze(1)
        code_input['attention_mask'] = code_input['attention_mask'].squeeze(1)
        code_outputs = self.graphcodebert(**code_input).pooler_output
        
        # 通过分类器得到预测结果
        output = self.classifier(code_outputs)
        
        return output

# 保存模型函数，包含 F1 和 Accuracy
def save_checkpoint(model, optimizer, epoch, loss, f1, accuracy, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'f1': f1,
        'accuracy': accuracy
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at epoch {epoch}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

# 验证函数
def evaluate(model, val_loader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc="Validating") as pbar:
            for code_input, label in val_loader:
                # 将数据移动到 GPU
                code_input = {key: value.to(device) for key, value in code_input.items()}
                label = label.to(device)

                outputs = model(code_input)
                preds.extend(outputs.cpu().numpy())
                true_labels.extend(label.cpu().numpy())
                pbar.update(1)
    
    preds = np.array(preds).flatten()
    true_labels = np.array(true_labels).flatten()

    # 计算指标
    acc = accuracy_score(true_labels, preds > 0.5)
    f1 = f1_score(true_labels, preds > 0.5)
    auc = roc_auc_score(true_labels, preds)
    precision = precision_score(true_labels, preds > 0.5)
    print(f'Validation: Accuracy: {acc}, F1: {f1}, AUC: {auc}, precision: {precision}')
    return acc, f1, auc

# 加载 checkpoint，包含 F1 和 Accuracy
def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    f1 = checkpoint.get('f1', None)  # 如果旧版本 checkpoint 没有 f1
    accuracy = checkpoint.get('accuracy', None)  # 如果旧版本 checkpoint 没有 accuracy
    
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss}, F1: {f1}, Accuracy: {accuracy}")
    return epoch, loss, f1, accuracy

def validate():
    # 示例数据格式:
    # [{'ast': 'AST 内容', 'code': '代码片段', 'label': 1}, {...}, ...]

    # 将数据集划分为训练集 (80%) 和评估集 (20%)
    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)

    val_dataset = CodeDataset(eval_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GraphCodeBERTFineTuner4code(graphcodebert).to(device)
    model = DataParallel(model)  # 将模型分布到多个GPU

    # 初始化模型、优化器和损失函数(checkpoint启用)-----------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluate(model, val_loader)

def train():
    end_epoch = 30
    save_path = 'checkpoints/finetune_onlycode/mixed_batch64/CWE_mixed_epoch_{epoch}.pt'

    # 将数据集划分为训练集 (80%) 和评估集 (20%)
    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)

    # 创建训练集和评估集的数据集实例
    train_dataset = CodeDataset(train_data, tokenizer)
    eval_dataset = CodeDataset(eval_data, tokenizer)
    print(f"训练集大小: {len(train_data)}")
    print(f"评估集大小: {len(eval_data)}")

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = GraphCodeBERTFineTuner4code(graphcodebert).to(device)
    model = DataParallel(model)  # 将模型分布到多个GPU

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 加载指定的 checkpoint
    # checkpoint_path = "checkpoints/finetune/CWE121_epoch_28.pth"
    # start_epoch, start_loss, start_f1, start_accuracy = load_checkpoint(checkpoint_path, model, optimizer)

    # 有关保存checkpoint --------- need to modify
    start_epoch = 0
    best_f1 = 0
    
    for epoch in range(start_epoch, end_epoch):  # 设定训练的 epoch 数
        # 微调模型
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{end_epoch}") as pbar:
            for code_input, label in train_dataloader:
                # 将数据移到当前 GPU 设备
                code_input = {key: value.to(device) for key, value in code_input.items()}
                label = label.to(device)

                optimizer.zero_grad()

                # 前向传播
                outputs = model(code_input)
                loss = criterion(outputs, label.unsqueeze(1).float())
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 更新进度条，显示当前 loss
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)  # 更新进度条
                
            print(f"Epoch {epoch + 1}/{end_epoch}, Loss: {loss.item()}")

            acc, f1, auc = evaluate(model, eval_dataloader)

        print(f'Validation: Accuracy: {acc}, F1: {f1}, AUC: {auc}')

        # 如果F1分数最高，保存当前模型
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, optimizer, epoch, loss.item(), f1, acc, save_path.format(epoch=epoch + 1))

if __name__ == '__main__':
    validate()