import re

def split_text_blocks(content):
    """切分混合内容为文本块、代码块和 LaTeX 公式块"""
    pattern = r'(```.*?```)|(\$\$.*?\$\$)|(\$.*?\$)|([^`$]+)'
    # pattern = r'(```*?```)|(\$\$.*?\$\$)|(\$.*?\$)|([^`$]+)'
    # pattern = r'(```cpp.*?```)|(\$\$.*?\$\$)|(\$.*?\$)|([^`$]+)'
    blocks = []
    for match in re.finditer(pattern, content, re.DOTALL):
        code_block, latex_block_double, latex_block_single, text_block = match.groups()
        if code_block:
            blocks.append(('code', code_block.strip()))
        elif latex_block_double:
            blocks.append(('latex', latex_block_double.strip()))
        elif latex_block_single:
            blocks.append(('latex', latex_block_single.strip()))
        elif text_block:
            blocks.append(('text', text_block.strip()))
    return blocks

def split_text(text, max_hanzi_per_line=5, max_words_per_line=4):
    """按汉字数量或英文单词数量拆分文本"""
    lines = []
    current_line = []
    char_count = 0
    word_count = 0

    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            # 处理汉字
            char_count += 1
            current_line.append(char)
            if char_count >= max_hanzi_per_line:
                lines.append(''.join(current_line).strip())
                current_line = []
                char_count = 0
        elif char.isspace():
            # 处理空格，判断单词数量
            if current_line:
                word_count += 1
                current_line.append(char)
                if word_count >= max_words_per_line:
                    lines.append(''.join(current_line).strip())
                    current_line = []
                    word_count = 0
            else:
                current_line.append(char)
        else:
            # 处理非汉字和非空格字符
            current_line.append(char)

    # 处理剩余内容
    if current_line:
        lines.append(''.join(current_line).strip())

    return '\n'.join(lines)


def analyze_code_variables(code_block):
    """分析代码变量数量（简化版）"""
    variable_keywords = {'int', 'float', 'double', 'char', 'bool', 'void', 'auto'}
    variables = set()

    # 匹配变量声明模式
    patterns = [
        r'\b([a-zA-Z_]\w*)\s*[,);=]',  # 函数参数和局部变量
        r'(?:\b(?:int|float|double|char|bool|void|auto)\s+)+([a-zA-Z_]\w*)\s*\(',  # 排除函数名
        r'^\s*([a-zA-Z_]\w*)\s+[^(]'   # 变量声明
    ]

    for line in code_block.split('\n'):
        if line.startswith('//') or not line.strip():
            continue

        for pattern in patterns:
            matches = re.findall(pattern, line)
            for var in matches:
                if var.lower() not in variable_keywords and not var[0].isupper():
                    variables.add(var)

    return variables

def process_content(content):
    """主处理函数"""
    blocks = split_text_blocks(content)
    output = []
    code_stats = []

    for block_type, content in blocks:
        if block_type == 'text':
            processed = split_text(content)
            output.append(processed)
        elif block_type == 'code':
            output.append(content)
            variables = analyze_code_variables(content)
            code_stats.append((content, len(variables)))
        elif block_type == 'latex':
            # LaTeX 公式不拆分，直接保留
            output.append(content)

    # 添加代码统计信息
    if code_stats:
        stat_output = ['\n### 代码变量统计']
        for i, (code, count) in enumerate(code_stats, 1):
            stat_output.append(f'{i}. 代码块 {i} 变量数: {count}')
        output.append('\n'.join(stat_output))

    return '\n\n'.join(output)

def read_file(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"读取文件 {file_path} 时出现错误: {e}")
        return None

def write_file(file_path, content):
    """写入文件内容"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        print(f"写入文件 {file_path} 时出现错误: {e}")

def main():
    # 定义文件路径前缀
    pre_ = r"D:\StepByStep"
    # 输入文件路径
    input_file = pre_ + '\\input.txt'
    # 输出文件路径
    output_file = pre_ + '\\output.txt'

    # 读取文件内容
    content = read_file(input_file)
    if content is None:
        return

    # 处理内容
    processed_content = process_content(content)

    # 写入输出文件
    write_file(output_file, processed_content)
    print(f"处理完成，结果已写入 {output_file}")

    # 输入文件路径
    input_file = pre_ + '\\output.txt'

    try:
        # 打开输入文件以读取内容
        with open(input_file, 'r', encoding='utf-8') as infile:
            # 读取输入文件的所有行
            lines = infile.readlines()

        # 处理每一行，在每一行后面添加一个空行
        new_lines = []
        for line in lines:
            new_lines.append(line)
            new_lines.append('\n')

        # 打开输出文件以写入处理后的内容
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # 将处理后的行写入输出文件
            outfile.writelines(new_lines)

        print("处理完成，结果已保存到", output_file)

    except FileNotFoundError:
        print(f"未找到文件: {input_file}，请检查文件路径是否正确。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()