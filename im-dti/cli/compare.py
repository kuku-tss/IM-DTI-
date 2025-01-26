import difflib

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def compare_files(file1_path, file2_path):
    file1_lines = read_file(file1_path)
    file2_lines = read_file(file2_path)

    diff = difflib.unified_diff(
        file1_lines, file2_lines, 
        fromfile='file1', tofile='file2', 
        lineterm=''
    )

    for line in diff:
        print(line)

# 使用绝对路径指定文件
file1_path = '/home/liujin/data/ConPLex-main/sum/so_a_probert_final'
file2_path = '/home/liujin/data/ConPLex-main/sum/soagain_a_probert_final'

compare_files(file1_path, file2_path)