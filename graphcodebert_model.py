import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, Dataset

class GraphCodeBERTFineTuner(nn.Module):
    def __init__(self, graphcodebert):
        super(GraphCodeBERTFineTuner, self).__init__()
        self.graphcodebert = graphcodebert
        
        # 分类器，用于将GraphCodeBERT的输出进行处理并得到分类结果
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 128),  # 768 是GraphCodeBERT的输出向量维度
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 假设是二分类任务
        )

    def forward(self, ast_input, code_input):
        # 将输入的形状从 [batch_size, 1, sequence_length] 变为 [batch_size, sequence_length]
        ast_input['input_ids'] = ast_input['input_ids'].squeeze(1)
        ast_input['attention_mask'] = ast_input['attention_mask'].squeeze(1)
        ast_outputs = self.graphcodebert(**ast_input).pooler_output

        code_input['input_ids'] = code_input['input_ids'].squeeze(1)
        code_input['attention_mask'] = code_input['attention_mask'].squeeze(1)
        code_outputs = self.graphcodebert(**code_input).pooler_output
        
        # 合并AST和代码的输出
        combined_output = torch.cat((ast_outputs, code_outputs), dim=1)
        
        # 通过分类器得到预测结果
        output = self.classifier(combined_output)
        
        return output


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
        
        '''
        # 使用 tokenizer 对 ast 进行编码 --------------------------> 适用于graph形式的ast
        ast_json = json.dumps(ast)
        ast_encoding = self.tokenizer.encode_plus(
            ast_json,
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,  # 设置最大长度
            return_tensors='pt'  # 返回 PyTorch 张量
        )
        '''
        '''
        # 这里根据需要进行tokenizer编码，或直接返回编码后的张量 -------------------------> token形式的ast
        ast_encoding2 = self.tokenizer.encode_plus(
            ast, padding='max_length', truncation=True, return_tensors='pt',
        )
        '''
        #  -------------------------> token形式的ast，限制最大长度，适用于contrabert
        ast_encoding3 = self.tokenizer.encode_plus(
            ast, 
            max_length=self.max_length, 
            padding='max_length',  # 补齐输入
            truncation=True,       # 截断到最大长度
            return_tensors='pt'
        )
        '''
        code_encoding = self.tokenizer.encode_plus(
            code, padding='max_length', truncation=True, return_tensors='pt'
        )
        '''
        #  -------------------------> 限制最大长度，适用于contrabert
        code_encoding2 = self.tokenizer.encode_plus(
            code,
            max_length=self.max_length, 
            padding='max_length',  # 补齐输入
            truncation=True,       # 截断到最大长度 return_tensors='pt'
            return_tensors='pt'
        )
        
        return ast_encoding3, code_encoding2, torch.tensor(label)