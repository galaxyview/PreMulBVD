import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import os
import json
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import DataParallel
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm  # 导入tqdm库显示进度条

from graphcodebert_model import CodeDataset, GraphCodeBERTFineTuner, CodeT5FineTuner, CodeDataset_T5
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# 指定使用 GPU 
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 1e-5

class GraphUnixCoderFineTuner(nn.Module):
    def __init__(self, unixcoder):
        super(GraphUnixCoderFineTuner, self).__init__()
        self.unixcoder = unixcoder
        
        # 分类器，用于将UnixCoder的输出进行处理并得到分类结果
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 128),  # 假设UnixCoder的输出维度与CodeBERT相同（768）
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 二分类任务
        )

    def forward(self, ast_input, code_input):
        # 处理AST输入
        ast_input['input_ids'] = ast_input['input_ids'].squeeze(1)
        ast_input['attention_mask'] = ast_input['attention_mask'].squeeze(1)
        ast_outputs = self.unixcoder(**ast_input).pooler_output

        # 处理代码输入
        code_input['input_ids'] = code_input['input_ids'].squeeze(1)
        code_input['attention_mask'] = code_input['attention_mask'].squeeze(1)
        code_outputs = self.unixcoder(**code_input).pooler_output
        
        # 合并AST和代码的输出
        combined_output = torch.cat((ast_outputs, code_outputs), dim=1)
        
        # 通过分类器得到预测结果
        output = self.classifier(combined_output)
        
        return output

def codeunixcoder_train():
    # 加载UnixCoder模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("pretrained_model/unixcoder-base")
    unixcoder = AutoModel.from_pretrained("pretrained_model/unixcoder-base")

    # 加载数据集, 设置checkpoint路径，设置是否使用checkpoint
    data = torch.load('datasets/pth/Juliet/CWE_astandcode_mixed.pth')
    save_path = 'checkpoints/finetune_withcode/mixed_unixcoder_batch64/CWE_mixed_epoch_{epoch}.pt'
    end_epochs = 30
    use_checkpoint = 0
    checkpoint_path = 'checkpoints/finetune_withcode/CWE121_epoch_6.pt'

    # 数据拆分
    ast_data = [item['ast'] for item in data]
    code_data = [item['pseudocode'] for item in data]
    labels = [item['label'] for item in data]
    train_ast_data, val_ast_data, train_code_data, val_code_data, train_labels, val_labels = train_test_split(
        ast_data, code_data, labels, test_size=0.2, random_state=42
    )

    # 创建数据集和数据加载器
    train_dataset = CodeDataset(train_ast_data, train_code_data, train_labels, tokenizer)
    val_dataset = CodeDataset(val_ast_data, val_code_data, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 使用UnixCoder的模型
    model = GraphUnixCoderFineTuner(unixcoder).to(device)
    model = DataParallel(model)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if use_checkpoint == 0:
        best_f1 = 0
        start_epoch = 0
    elif use_checkpoint == 1:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']

    for epoch in range(start_epoch, end_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{end_epochs}") as pbar:
            for ast_input, code_input, label in train_loader:
                ast_input = {key: value.to(device) for key, value in ast_input.items()}
                code_input = {key: value.to(device) for key, value in code_input.items()}
                label = label.to(device)

                optimizer.zero_grad()

                # 前向传播
                outputs = model(ast_input, code_input)
                loss = criterion(outputs, label.unsqueeze(1).float())
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # 更新进度条
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)

        print(f'Epoch {epoch+1}/{end_epochs}, Loss: {running_loss/len(train_loader)}')

        # 验证模型
        acc, f1, auc = evaluate(model, val_loader)
        print(f'Validation: Accuracy: {acc}, F1: {f1}, AUC: {auc}')

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, optimizer, epoch, loss.item(), f1, acc, save_path.format(epoch=epoch + 1))


class CodeDataset(Dataset):
    def __init__(self, ast_data, code_data, labels, tokenizer):
        self.ast_data = ast_data
        self.code_data = code_data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = 512

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ast = self.ast_data[idx]
        code = self.code_data[idx]
        label = self.labels[idx]
        
        # Tokenizing AST
        ast_encoding = self.tokenizer.encode_plus(
            ast, 
            max_length=self.max_length, 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenizing code
        code_encoding = self.tokenizer.encode_plus(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return ast_encoding, code_encoding, torch.tensor(label)

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
    print(f"Checkpoint saved at epoch {epoch+1}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

# 验证函数
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

    # 计算指标
    acc = accuracy_score(true_labels, preds > 0.5)
    f1 = f1_score(true_labels, preds > 0.5)
    auc = roc_auc_score(true_labels, preds)
    precision = precision_score(true_labels, preds > 0.5)
    recall = recall_score(true_labels, preds > 0.5)

    print(f'Validation: Accuracy: {acc}, F1: {f1}, AUC: {auc}, precision: {precision}, recall: {recall}')
    return acc, f1, auc

if __name__ == '__main__':
    codeunixcoder_train()