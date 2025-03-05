import os
import shutil
import subprocess

def execute_commands_in_directory(directory, command_template):
    # 获取目录下的所有文件名
    files = [filename for filename in os.listdir(directory) if filename.endswith('.o')]

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

def delete_path(path):
    """删除指定路径的文件或文件夹"""
    if os.path.exists(path):  # 确保路径存在
        if os.path.isfile(path):  # 如果是文件，删除文件
            os.remove(path)
            print(f"文件 '{path}' 已删除")
        elif os.path.isdir(path):  # 如果是文件夹，删除整个文件夹
            shutil.rmtree(path)
            print(f"文件夹 '{path}' 及其内容已删除")
    else:
        print(f"路径 '{path}' 不存在")

def move_pseudo_files(source_directory, target_directory, endwith):
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 获取源目录下的所有文件
    files = os.listdir(source_directory)

    # 依次处理每个文件
    for file_name in files:
        # 检查文件是否以.sqlite结尾
        if file_name.endswith(endwith):
            source_path = os.path.join(source_directory, file_name)
            target_path = os.path.join(target_directory, file_name)

            # 确保只处理文件
            if os.path.isfile(source_path):
                # 移动文件
                shutil.move(source_path, target_path)
                print(f"Moved: {source_path} to {target_path}")

if __name__ == '__main__':
    ghidra_path = "~/workspace/ghidra_11.1_PUBLIC"
    script_path = "/home/ubuntu/workspace/Asteria/Ghidra_parser/ghidra-headless-scripts/"
    assembly_or_pseudo = 1 # 1生成汇编代码，0生成伪代码
    if assembly_or_pseudo == 1:
        post_script = "disassembler.py" # 生成汇编代码用disassembler.py，生成伪代码用decompiler.py
        directory_end = "/ghidra_assembly"
        file_end = ".asm"
    elif assembly_or_pseudo == 0:
        post_script = "decompiler.py"
        directory_end = "/ghidra_pseudo"
        file_end = ".c"
    for testcase_num in range(1,2):
        # 指定目录和命令模板
        directory = "/home/ubuntu/workspace/Asteria/datasets/C/testcases/CWE121_Stack_Based_Buffer_Overflow/s"+str(testcase_num).zfill(2)
        # directory = "/home/ubuntu/workspace/Asteria/datasets/C/testcases/CWE188_Reliance_on_Data_Memory_Layout"
        good_directory = directory + "/binary_good"
        good_target_directory = good_directory + directory_end
        bad_directory = directory + "/binary_bad"
        bad_target_directory = bad_directory + directory_end
        # 需要修改python文件 API_ast_generator.py 的位置
        command_template = ghidra_path + "/support/analyzeHeadless /home/ubuntu/workspace/Asteria/Ghidra_parser/ HeadlessAnalysis -import {file_path} -scriptPath" + script_path + "  -postscript " + post_script + " {file_path}.asm"
        # 执行脚本
        execute_commands_in_directory(good_directory, command_template)
        move_pseudo_files(good_directory, good_target_directory, file_end)
        delete_path("/home/ubuntu/workspace/Asteria/Ghidra_parser/HeadlessAnalysis.rep")
        delete_path("/home/ubuntu/workspace/Asteria/Ghidra_parser/HeadlessAnalysis.gpr")
        execute_commands_in_directory(bad_directory, command_template)
        move_pseudo_files(bad_directory, bad_target_directory, file_end)
    


'''
~/workspace/ghidra_11.1_PUBLIC/support/analyzeHeadless ~/workspace/ HeadlessAnalysis\
    -import /home/ubuntu/workspace/Asteria/datasets/C/testcases/CWE121_Stack_Based_Buffer_Overflow/s09/binary_bad/CWE121_Stack_Based_Buffer_Overflow__src_char_declare_cpy_01.o \
    -scriptPath /home/ubuntu/workspace/Asteria/Ghidra_parser/ghidra-headless-scripts/ \
    -postscript decompiler.py \
    /home/ubuntu/workspace/Asteria/datasets/C/testcases/CWE121_Stack_Based_Buffer_Overflow/s09/binary_bad/Ghidra/CWE121_Stack_Based_Buffer_Overflow__src_char_declare_cpy_01.c
'''