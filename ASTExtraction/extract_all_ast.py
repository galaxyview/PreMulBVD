import os
import shutil
import subprocess

def execute_commands_in_directory(directory, command_template):
    # 获取目录下的所有文件名
    files = os.listdir(directory)

    # 依次处理每个文件名
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        
        # 确保只处理文件
        if os.path.isfile(file_path):
            # 构建命令
            command = command_template.format(file_path=file_path)
            
            # 打印命令（可选）
            # print(f"Executing: {command}")
            
            # 执行命令
            subprocess.run(command, shell=True)


def move_sqlite_files(source_directory, target_directory):
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 获取源目录下的所有文件
    files = os.listdir(source_directory)

    # 依次处理每个文件
    for file_name in files:
        # 检查文件是否以.sqlite结尾
        if file_name.endswith('.sqlite'):
            source_path = os.path.join(source_directory, file_name)
            target_path = os.path.join(target_directory, file_name)

            # 确保只处理文件
            if os.path.isfile(source_path):
                # 移动文件
                shutil.move(source_path, target_path)
                print(f"Moved: {source_path} to {target_path}")



if __name__ == '__main__':
    # 指定目录和命令模板
    directory = ""
    good_directory = directory + "\\binary_good"
    good_target_directory = good_directory + "\\sqlite_assembly"
    bad_directory = directory + "\\binary_bad"
    bad_target_directory = bad_directory + "\\sqlite_assembly" 
    # 需要修改python文件 API_ast_generator.py 的位置， ida_path需要修改成对应idat64.exe的位置
    command_template = "python ASTExtraction\\API_ast_generator.py --ida_path D:\\IDAPro8.3\\idat64.exe --binary {file_path} --database {file_path}.sqlite"
    # 执行脚本
    execute_commands_in_directory(good_directory, command_template)
    move_sqlite_files(good_directory, good_target_directory)
    execute_commands_in_directory(bad_directory, command_template)
    move_sqlite_files(bad_directory, bad_target_directory)
    