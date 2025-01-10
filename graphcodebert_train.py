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

from graphcodebert_model import CodeDataset, GraphCodeBERTFineTuner

# 加载GraphCodeBERT模型和分词器
tokenizer = RobertaTokenizer.from_pretrained("pretrained_model/graphcodebert-base")
graphcodebert = RobertaModel.from_pretrained("pretrained_model/graphcodebert-base")
#torch.cuda.set_device(6)  # 将0替换为你想使用的GPU编号
# 指定使用 GPU 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,5'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 1e-5

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
    
def train():
    # <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->
    data = torch.load('datasets/pth/Juliet/CWE_astandcode_mixed_graphast.pth')
    save_path = 'checkpoints/finetune_withcode/mixed_astgraph_batch64/CWE_mixed_epoch_{epoch}.pt'
    end_epochs = 30
    use_checkpoint = 0 # 1表示使用，0表示不使用
    checkpoint_path = 'checkpoints/finetune_withcode/CWE121_epoch_6.pt'
    # <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->

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

    # 训练和验证数据集
    train_dataset = CodeDataset(train_ast_data, train_code_data, train_labels, tokenizer)
    val_dataset = CodeDataset(val_ast_data, val_code_data, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GraphCodeBERTFineTuner(graphcodebert).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if use_checkpoint == 0:
        # 初始化模型、优化器和损失函数(没有checkpoint启用)------------------------------
        best_f1 = 0
        start_epoch = 0
    elif use_checkpoint == 1:
        # 初始化模型、优化器和损失函数(checkpoint启用)-----------------------------
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']

    for epoch in range(start_epoch ,end_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{end_epochs}") as pbar:
            for ast_input, code_input, label in train_loader:
                # 将数据移到GPU
                ast_input = {key: value.to(device) for key, value in ast_input.items()}
                code_input = {key: value.to(device) for key, value in code_input.items()}
                label = label.to(device)

                optimizer.zero_grad()

                # 前向传播
                outputs = model(ast_input, code_input)
                loss = criterion(outputs, label.unsqueeze(1).float())
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # 更新进度条，显示当前loss和accuracy
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)  # 更新进度条

        print(f'Epoch {epoch+1}/{end_epochs}, Loss: {running_loss/len(train_loader)}')

        # 验证模型
        acc, f1, auc = evaluate(model, val_loader)
        print(f'Validation: Accuracy: {acc}, F1: {f1}, AUC: {auc}')

        # 如果F1分数最高，保存当前模型
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, optimizer, epoch, loss.item(), f1, acc, save_path.format(epoch=epoch + 1))

def mutigpu_train(rank, world_size):
    # <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->
    data = torch.load('datasets/pth/Juliet/CWE_astandcode_mixed_graphast.pth')
    save_path = 'checkpoints/finetune_withcode/mixed_astgraph_batch64/CWE_mixed_epoch_{epoch}.pt'
    end_epochs = 30
    use_checkpoint = 0 # 1表示使用，0表示不使用
    checkpoint_path = 'checkpoints/finetune_withcode/CWE121_epoch_6.pt'
    # <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->

    def set_env_variables():
        # 设置 MASTER_ADDR 和 MASTER_PORT 的默认值
        os.environ['MASTER_ADDR'] = 'localhost'  # 或者主节点的IP地址
        os.environ['MASTER_PORT'] = '12355'      # 一个未被占用的端口号

    def setup(rank, world_size):
        set_env_variables()
        # 初始化进程组，只调用一次
        if not dist.is_initialized():  # 确保进程组没有被初始化过
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()  # 清理进程组

    setup(rank, world_size)  # 初始化分布式进程组

    # 初始化分布式进程组
    torch.cuda.set_device(rank)

    # 提取 AST, CODE, LABEL 列表
    ast_data = [item['ast'] for item in data]
    code_data = [item['pseudocode'] for item in data]
    labels = [item['label'] for item in data]
    
    # 按 8:2 比例分割数据
    train_ast_data, val_ast_data, train_code_data, val_code_data, train_labels, val_labels = train_test_split(
        ast_data, code_data, labels, test_size=0.2, random_state=42
    )

    # 创建数据集和分布式数据采样器
    train_dataset = CodeDataset(train_ast_data, train_code_data, train_labels, tokenizer)
    val_dataset = CodeDataset(val_ast_data, val_code_data, val_labels, tokenizer)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # 初始化模型、损失函数和优化器
    model = GraphCodeBERTFineTuner(graphcodebert).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if use_checkpoint == 0:
        best_f1 = 0
        start_epoch = 0
    elif use_checkpoint == 1:
        checkpoint = torch.load(checkpoint_path, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']

    for epoch in range(start_epoch, end_epochs):
        model.train()
        running_loss = 0.0
        train_sampler.set_epoch(epoch)  # 在每个 epoch 设置随机种子

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{end_epochs}") as pbar:
            for ast_input, code_input, label in train_loader:
                # 将数据移到 GPU
                ast_input = {key: value.to(rank) for key, value in ast_input.items()}
                code_input = {key: value.to(rank) for key, value in code_input.items()}
                label = label.to(rank)

                optimizer.zero_grad()

                # 前向传播
                outputs = model(ast_input, code_input)
                loss = criterion(outputs, label.unsqueeze(1).float())
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)

        print(f'Epoch {epoch+1}/{end_epochs}, Loss: {running_loss/len(train_loader)}')

        # 验证模型
        acc, f1, auc = evaluate(model, val_loader)
        print(f'Validation: Accuracy: {acc}, F1: {f1}, AUC: {auc}')

        # 如果 F1 分数最高，保存当前模型
        if f1 > best_f1:
            best_f1 = f1
            if rank == 0:  # 仅在主进程保存模型
                save_checkpoint(model, optimizer, epoch, running_loss, f1, acc, save_path.format(epoch=epoch + 1))

    cleanup()  # 销毁分布式进程组，确保每个进程结束时清理
    
def dataparallel_train():
    # <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->
    data = torch.load('datasets/pth/Juliet/CWE_astandcode_mixed_graphast.pth')
    save_path = 'checkpoints/finetune_withcode/mixed_astgraph_batch64_lr1e-6/CWE_mixed_epoch_{epoch}.pt'
    end_epochs = 30
    use_checkpoint = 0 # 1表示使用，0表示不使用
    checkpoint_path = 'checkpoints/finetune_withcode/CWE121_epoch_6.pt'
    # <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->

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

    # 训练和验证数据集
    train_dataset = CodeDataset(train_ast_data, train_code_data, train_labels, tokenizer)
    val_dataset = CodeDataset(val_ast_data, val_code_data, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型，并使用 DataParallel 包裹模型，以利用多个 GPU
    model = GraphCodeBERTFineTuner(graphcodebert).to(device)
    model = DataParallel(model)  # 将模型分布到多个GPU

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 加载 checkpoint
    if use_checkpoint == 0:
        best_f1 = 0
        start_epoch = 0
    elif use_checkpoint == 1:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']

    # 训练过程
    for epoch in range(start_epoch, end_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{end_epochs}") as pbar:
            for ast_input, code_input, label in train_loader:
                # 将数据移到当前 GPU 设备
                ast_input = {key: value.to(device) for key, value in ast_input.items()}
                code_input = {key: value.to(device) for key, value in code_input.items()}
                label = label.to(device)

                optimizer.zero_grad()

                # 前向传播
                outputs = model(ast_input, code_input)
                loss = criterion(outputs, label.unsqueeze(1).float())
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # 更新进度条，显示当前 loss
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)  # 更新进度条

        print(f'Epoch {epoch+1}/{end_epochs}, Loss: {running_loss/len(train_loader)}')

        # 验证模型
        acc, f1, auc = evaluate(model, val_loader)
        print(f'Validation: Accuracy: {acc}, F1: {f1}, AUC: {auc}')

        # 如果F1分数最高，保存当前模型
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, optimizer, epoch, loss.item(), f1, acc, save_path.format(epoch=epoch + 1))

def codebert_train():
    # 加载CodeBERT模型和分词器
    tokenizer = RobertaTokenizer.from_pretrained("pretrained_model/codebert-base")
    codebert = RobertaModel.from_pretrained("pretrained_model/codebert-base")

    # <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->
    data = torch.load('datasets/pth/Juliet/CWE_astandcode_mixed.pth')
    save_path = 'checkpoints/finetune_withcode/mixed_codebert_batch64/CWE_mixed_epoch_{epoch}.pt'
    end_epochs = 30
    use_checkpoint = 0 # 1表示使用，0表示不使用
    checkpoint_path = 'checkpoints/finetune_withcode/CWE121_epoch_6.pt'
    # <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->

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

    # 训练和验证数据集
    train_dataset = CodeDataset(train_ast_data, train_code_data, train_labels, tokenizer)
    val_dataset = CodeDataset(val_ast_data, val_code_data, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GraphCodeBERTFineTuner(codebert).to(device)
    model = DataParallel(model)  # 将模型分布到多个GPU
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if use_checkpoint == 0:
        # 初始化模型、优化器和损失函数(没有checkpoint启用)------------------------------
        best_f1 = 0
        start_epoch = 0
    elif use_checkpoint == 1:
        # 初始化模型、优化器和损失函数(checkpoint启用)-----------------------------
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']

    for epoch in range(start_epoch ,end_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{end_epochs}") as pbar:
            for ast_input, code_input, label in train_loader:
                # 将数据移到GPU
                ast_input = {key: value.to(device) for key, value in ast_input.items()}
                code_input = {key: value.to(device) for key, value in code_input.items()}
                label = label.to(device)

                optimizer.zero_grad()

                # 前向传播
                outputs = model(ast_input, code_input)
                loss = criterion(outputs, label.unsqueeze(1).float())
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # 更新进度条，显示当前loss和accuracy
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)  # 更新进度条

        print(f'Epoch {epoch+1}/{end_epochs}, Loss: {running_loss/len(train_loader)}')

        # 验证模型
        acc, f1, auc = evaluate(model, val_loader)
        print(f'Validation: Accuracy: {acc}, F1: {f1}, AUC: {auc}')

        # 如果F1分数最高，保存当前模型
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, optimizer, epoch, loss.item(), f1, acc, save_path.format(epoch=epoch + 1))

def contrabert_train():
    # 加载CodeBERT模型和分词器
    tokenizer = RobertaTokenizer.from_pretrained("pretrained_model/contrabert-base")
    codebert = RobertaModel.from_pretrained("pretrained_model/contrabert-base")

    # <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->
    data = torch.load('datasets/pth/Juliet/CWE_astandcode_mixed.pth')
    save_path = 'checkpoints/finetune_withcode/mixed_contrabert_batch64/CWE_mixed_epoch_{epoch}.pt'
    end_epochs = 30
    use_checkpoint = 1 # 1表示使用，0表示不使用
    checkpoint_path = 'checkpoints/finetune_withcode/mixed_contrabert_batch64/CWE_mixed_epoch_4.pt'
    # <----------------------------------------加载数据集 .pth 文件, 设置checkpoint保存路径，设置是否使用checkpoint---------------------------------------->

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

    # 训练和验证数据集
    train_dataset = CodeDataset(train_ast_data, train_code_data, train_labels, tokenizer)
    val_dataset = CodeDataset(val_ast_data, val_code_data, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GraphCodeBERTFineTuner(codebert).to(device)
    model = DataParallel(model)  # 将模型分布到多个GPU
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if use_checkpoint == 0:
        # 初始化模型、优化器和损失函数(没有checkpoint启用)------------------------------
        best_f1 = 0
        start_epoch = 0
    elif use_checkpoint == 1:
        # 初始化模型、优化器和损失函数(checkpoint启用)-----------------------------
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']

    for epoch in range(start_epoch ,end_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{end_epochs}") as pbar:
            for ast_input, code_input, label in train_loader:
                # 将数据移到GPU
                ast_input = {key: value.to(device) for key, value in ast_input.items()}
                code_input = {key: value.to(device) for key, value in code_input.items()}
                label = label.to(device)

                optimizer.zero_grad()

                # 前向传播
                outputs = model(ast_input, code_input)
                loss = criterion(outputs, label.unsqueeze(1).float())
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # 更新进度条，显示当前loss和accuracy
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)  # 更新进度条

        print(f'Epoch {epoch+1}/{end_epochs}, Loss: {running_loss/len(train_loader)}')

        # 验证模型
        acc, f1, auc = evaluate(model, val_loader)
        print(f'Validation: Accuracy: {acc}, F1: {f1}, AUC: {auc}')

        # 如果F1分数最高，保存当前模型
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, optimizer, epoch, loss.item(), f1, acc, save_path.format(epoch=epoch + 1))

if __name__== '__main__':
    contrabert_train()
    '''
    gpu_ids = [0, 1, 2, 3]  # 使用 GPU 6 和 GPU 7
    world_size = len(gpu_ids)
    mp.spawn(mutigpu_train, args=(world_size,),nprocs=world_size, join=True)
    '''