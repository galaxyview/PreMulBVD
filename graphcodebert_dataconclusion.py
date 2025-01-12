import torch
import os
from pathlib import Path
from tree_sitter import Language, Parser
import re
import os
import networkx as nx

from datahelper import DataHelper

#tokenizer = RobertaTokenizer.from_pretrained("pretrained_model/graphcodebert-base")
#model = RobertaModel.from_pretrained("pretrained_model/graphcodebert-base")


# <------------------------------生成数据集---------------------------------->
class TreeToGraphConverter:
    def __init__(self, tree):
        self.tree = tree
        self.graph_data = {
            'nodes': [],
            'edges': []
        }

    def build_graph(self, node, parent_id=None):
        node_id = id(node)
        
        # 保存节点信息
        self.graph_data['nodes'].append({
            'id': node_id,
            'opname': node.opname,
            'value': node.value
        })
        
        # 如果有父节点，保存边信息
        if parent_id is not None:
            self.graph_data['edges'].append({
                'source': parent_id,
                'target': node_id
            })

        # 递归处理子节点
        for child in node.children:
            self.build_graph(child, node_id)

    def get_graph_data(self):
        # 从 AST 的根节点开始构建图
        self.build_graph(self.tree)
        return self.graph_data

# <---------------------treesitter 归一化功能-------------------------->
# 构建 Tree-sitter 语言库
Language.build_library(
    'build/my-languages.so',  # 输出文件
    ['tree-sitter-c']         # C 语言解析器
)

# 加载 C 语言的 Tree-sitter 解析器
C_LANGUAGE = Language('build/my-languages.so', 'c')
parser = Parser()
parser.set_language(C_LANGUAGE)

# 归一化函数的代码
def normalize_function_code(code):
    # 解析代码
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    # 递归遍历 AST 树
    normalized_code = traverse_node(root_node.children[0], code)
    return normalized_code

def traverse_node(node, code):
    result = []
    
    # 只处理函数定义和内部的元素
    if node.type == 'function_definition' or node.type == 'compound_statement' or node.type == 'declaration' or\
        node.type == 'expression_statement' or node.type == 'assignment_expression' or node.type == 'argument_list' or\
            node.type == 'return_statement' or node.type == 'binary_expression' or node.type == 'array_declarator' or\
                node.type == 'subscript_expression':
        # 获取函数体部分
        for child in node.children:
            if child.type == 'function_declarator':  # 处理函数名
                for childd in child.children:
                    if childd.type == 'identifier':
                        result.append('FUNC_NAME')  # 归一化函数名
                    elif childd.type == 'parameter_list':
                        result.append(childd.text.decode('utf-8'))
            elif child.type == 'return_statement':
                result.append('return '+' '+traverse_node(child, code).split('return')[-1])
            else:
                result.append(traverse_node(child, code))  # 递归处理子节点
        if node.type == 'expression_statement' or node.type == 'declaration':
            result[-1] += '\n'

    elif node.type == 'call_expression':
        temp_nodetype = []
        for child in node.children:
            temp_nodetype.append(child.type)
        if 'identifier' in temp_nodetype and 'argument_list' in temp_nodetype:
            for child in node.children:
                if child.type == 'identifier':  # 处理函数名
                    result.append(child.text.decode('utf-8'))  # 归一化函数名
                else:
                    result.append(traverse_node(child, code))  # 递归处理子节点
    elif node.type == 'identifier':  # 处理变量名
        result.append('VAR_NAME')
    elif node.type == 'string_literal':  # 处理字符串常量
        result.append('"STRING_CONST"')
    elif node.type == 'number_literal':  # 处理数字常量
        result.append('NUM_CONST')
    elif node.type == 'comment':
        result.append('')
    elif node.type == 'sized_type_specifier' or node.type == 'type_identifier':
        result.append(code[node.start_byte:node.end_byte]+' ')
    else:
        # 对于其他节点类型，保留原始代码片段
        result.append(code[node.start_byte:node.end_byte])
    
    return ''.join(result)
# <---------------------treesitter 归一化功能-------------------------->

def get_all_file_paths(directory):
    # 获取目录下的所有文件和子目录
    items = os.listdir(directory)
    file_paths = []

    # 依次处理每个文件和子目录
    for item in items:
        item_path = os.path.join(directory, item)
        
        # 如果是文件，则添加其绝对路径到数组中
        if os.path.isfile(item_path):
            file_paths.append(os.path.abspath(item_path))
    
    return file_paths

def tree_to_string(node):
    if node.num_children == 0:
        return node.opname  # 叶子节点直接返回节点的 opname
    else:
        children_str = " ".join([tree_to_string(child) for child in node.children])
        return f"({node.opname} {children_str})"

# 归一化函数
def normalize_code(code: str) -> str:
    # 1. 删除单行注释和多行注释
    def remove_comments(code):
        # 移除单行注释
        code = re.sub(r'//.*', '', code)
        # 移除多行注释
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        return code

    code = remove_comments(code)  # 删除注释


    # 1. 归一化函数名，适配任意返回类型，并避免拼接
    def normalize_function_name(match):
        return f"{match.group(1)} normalized_func("  # 修复多余的反括号

    # 使用正则匹配函数定义并归一化函数名，匹配任意返回类型和函数名，避免重复归一化
    code = re.sub(r'(\w[\w\s\*\[\]]+)\s+(\w+)\s*\(', normalize_function_name, code)

    # 2. 归一化变量名，适配不同变量类型，包括指针类型
    var_pattern = re.compile(r'\b((?:const\s+)?(?:unsigned\s+)?(?:char|int|__int64|float|double|unsigned __int64|const char)\s*\*?)\s+(\w+)\s*(\[.*\])?')
    var_count = 1
    var_map = {}

    def normalize_variable(match):
        nonlocal var_count
        var_type = match.group(1)
        var_name = match.group(2)
        if var_name not in var_map:
            var_map[var_name] = f"var{var_count}"
            var_count += 1
        return f"{var_type} {var_map[var_name]}{match.group(3) or ''}"

    def normalize_variable_usage(match):
        var_name = match.group(0)
        return var_map.get(var_name, var_name)  # 如果变量已在声明中归一化，则使用归一化后的名称

    # 先归一化声明中的变量名
    code_body = re.split(r'(\{)', code, maxsplit=1)  # 分割函数头和主体
    if len(code_body) > 1:
        # 归一化函数体中的变量声明
        code_body[1] = re.sub(var_pattern, normalize_variable, code_body[1])
        
        # 使用归一化后的变量名替换函数体中的变量使用
        var_names_pattern = re.compile(r'\b(' + '|'.join(re.escape(v) for v in var_map.keys()) + r')\b')
        code_body[1] = re.sub(var_names_pattern, normalize_variable_usage, code_body[1])
    code = ''.join(code_body)

    # 3. 归一化常量和字符串（将硬编码数值和字符串替换为占位符）
    code = re.sub(r'\b\d+\b', 'CONSTANT_VAL', code)  # 数字常量
    code = re.sub(r'"[^"]*"', '"STRING_LITERAL"', code)  # 字符串常量

    # 4. 归一化16进制数值（如 0x28u）
    code = re.sub(r'0x[0-9a-fA-F]+[uUlL]*', 'HEX_VAL', code)

    # 5. 归一化数组索引（如 dest[1] -> var1[INDEX]）
    code = re.sub(r'\[(\d+)\]', '[INDEX]', code)

    # 6. 避免归一化 return 语句中的外部函数调用
    def skip_external_functions_in_return(match):
        return f"return {match.group(1)}"  # 保留外部函数名，不归一化
    
    # 匹配 return 语句的函数调用，并排除归一化
    return_pattern = re.compile(r'\breturn\s+([^\s\(\)]+(?:\s*\([^)]*\))?)')
    code = re.sub(return_pattern, skip_external_functions_in_return, code)

    # 7. 确保赋值和其他代码中的变量也被归一化
    def normalize_assignments(match):
        return f"{var_map.get(match.group(1), match.group(1))} ="

    # 变量赋值中的归一化（包括指针类型变量）
    code = re.sub(r'\b(' + '|'.join(re.escape(v) for v in var_map.keys()) + r')\s*=', normalize_assignments, code)

    return code

def generate_ast2db(feat_path, binary_good_path, binary_bad_path, encode_path=None):
    dh = DataHelper()
    sigle_function_patterns = ['good']  # 单缺陷good函数命名规则
    multi_function_patterns = ['G2B','B2G']    # 多缺陷good函数命名规则
    ft_path = feat_path
    func2embed_info = {}
    file_name = feat_path.split('/')[-1].rsplit('.sqlite', 1)[0]
    feat_path = Path(feat_path)
    if encode_path is None:
        encode_path = feat_path.parent.joinpath('encodings',file_name +'_encodings.pkl')
    else:
        encode_path = Path(encode_path)
    '''
    if encode_path.exists():
        return {
            'errcode': 0,
            'encode_path': str(encode_path)
        }
    '''
    res = {
        'feat_path': str(feat_path),
        'encode_path': str(encode_path)
    }
    try:
        functions = list(dh.get_ast_hash_encode(feat_path))
        for func in functions:
            flag = 0    # 判断该函数是否需要加到数据集中
            if ft_path in binary_good_path:  # 挑选正样本
                if func[0].endswith('_good'):
                    continue
                if ft_path.split('/')[-1].count('CWE') == 2:   # 区分单漏洞和多漏洞的函数名
                    for fp in multi_function_patterns:
                        if fp in func[0]:
                            flag = 1
                            continue
                elif ft_path.split('/')[-1].count('CWE') == 1:
                    for fp in sigle_function_patterns:
                        if fp in func[0]:
                            flag = 1
                            continue
                label = 1
            elif ft_path in binary_bad_path:   # 挑选负样本
                flag = 1
                label = 0

            if flag == 0:
                continue                            

            ast = func[-2]
            ast_str = tree_to_string(ast)
            converter = TreeToGraphConverter(ast)
            graph_ast = converter.get_graph_data()
            #code_normalized = normalize_function_code(func[-3])

            func2embed_info[func[0]] = {
                #'ast': graph_ast,
                'pseudocode': func[-3],
                'label':label
            }
    except EOFError:
        res.update({
            'errcode': 400,
            'errmsg': 'Asteria feature is empty',
        })
    except FileNotFoundError:
        res.update({
            'errcode': 404,
            'errmsg': 'can not find feature path, please generate it first',
        })
    else:
        res.update({
            'errcode': 0,
        })
        # write_pickle(func2embed_info, encode_path)
    return func2embed_info

def generate_db_main():
    main_paths = {
        
        'datasets/C/testcases/CWE843_Type_Confusion':0
    }
    '''
    'datasets/C/testcases/CWE188_Reliance_on_Data_Memory_Layout': 0,
        'datasets/C/testcases/CWE121_Stack_Based_Buffer_Overflow': 9,  
        'datasets/C/testcases/CWE122_Heap_Based_Buffer_Overflow': 11,
    '''
    path_list = []

    for main_path, value in main_paths.items():
        if value == 0:
            # 如果值为0，直接追加路径
            path_list.append(main_path)
        else:
            # 如果值为非0，执行for循环，结束值为value+1
            for i in range(1, value + 1):
                path_list.append(main_path + '/s' + str(i).zfill(2))

    all_data = []
    for path in path_list:
        binary_good_path = get_all_file_paths(path + "/binary_good/sqlite_assembly")
        binary_bad_path = get_all_file_paths(path + "/binary_bad/sqlite_assembly")
        binary_path = binary_good_path + binary_bad_path
        pattern = re.compile(r'(\d+\.o\.sqlite|a+\.o\.sqlite)$')    # 跨多文件调用时只取编号为a的文件
        binary_path = [item for item in binary_path if pattern.search(item)]
        dh = DataHelper()
        # for db in binary_path:
        #     dh.generate_ast_hash_encode(db)
        for temp_path in binary_path: 
            data = generate_ast2db(feat_path=temp_path, binary_good_path=binary_good_path, binary_bad_path=binary_bad_path)
            all_data.extend(data.values())
    torch.save(all_data, 'datasets/pth/Juliet/CWE_assembly.pth')

if __name__ == '__main__':
    generate_db_main()