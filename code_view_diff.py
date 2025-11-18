import os
import re
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== vLLM配置（适配4B模型）====================
VLLM_BASE_URL = "http://localhost:8000/v1"
# 注意：替换为你的vLLM启动时指定的模型路径（与--model参数一致）
TARGET_MODEL = "/t9k/mnt/hjy/Qwen/Qwen3-4B-Instruct-2507"
MAX_TOKENS = 81920  # 4B模型建议8192-16384，根据实际模型上下文窗口调整
# ==========================================================
# 路径配置（Windows用\，Linux/Mac用/）
DEFAULT_INPUT_DIR = r"/t9k/mnt/hjy/generated_long"  # 优化文本输出目录
DEFAULT_OUTPUT_DIR = r"/t9k/mnt/hjy/anki_long"  # 优化文本输出目录
# ==========================================================

# 初始化vLLM客户端
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="EMPTY",
    timeout=3000  # 超时时间设为15分钟（格式转换无需20分钟）
)

def read_questions_text(file_path):
    """读取选择题文本，仅返回“”后面的内容"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()
            # 分割并取“”后的内容（仅分割一次，避免多标记干扰）
            if "</think>" in full_text:
                target_text = full_text.split("</think>", 1)[1].strip()
                return target_text if target_text else None
            else:
                print(f"⚠️ {os.path.basename(file_path)} 中未找到“”标记，跳过处理")
                return None
    except Exception as e:
        print(f"❌ 读取选择题 {os.path.basename(file_path)} 失败: {e}")
        return None

def validate_generated_content(content, task_name, file_name):
    """验证生成内容有效性：非空+基础长度+格式校验"""
    if not content:
        print(f"❌ {file_name} - {task_name} 内容为空")
        return False
    if len(content.strip()) < 800:  # 16道题的Anki格式至少需800字符（每道约50字符）
        print(f"❌ {file_name} - {task_name} 内容过短（不足800字符），无效")
        return False
    return True

def format_for_anki(questions_text, file_name):
    print(f"📝 {file_name} - 转换为Anki格式...")
    try:
        # 将包含反斜杠的示例单独定义为字符串（避免f-string解析冲突）
        example_text = """关联参考方向下，线性电容的电压-电流关系（VCR）微分形式为？\tA. $i = C \\frac{\\mathrm{d} u}{\\mathrm{d} t}$<br>B. $u = C \\frac{\\mathrm{d} i}{\\mathrm{d} t}$<br>C. $i = \\frac{1}{C} \\int u \\mathrm{d} t$<br>D. $u = \\frac{1}{C} \\int i \\mathrm{d} t$\tA\t文本推导后给出“关联参考方向下最终VCR公式：$i = C \\frac{\\mathrm{d} u}{\\mathrm{d} t}$”；B是电感的VCR微分形式，C为错误的积分关系（电容积分形式应为$u = \\frac{1}{C} \\int i \\mathrm{d} t$），D是电容VCR的**积分形式**（非题干要求的“微分形式”）。"""
        
        prompt = f"""将每一道选择题严格转换为Anki导入格式（4字段，Tab分隔，无任何额外内容，集中调整到一行里）：
1. 题干：完全保留原始题干，公式用MathJax格式（$包裹），不添加多余说明，如果开头没有题干这两个字符，添加上
2. 选项：必须按"A.<br>B.<br>C.<br>D."格式分行（<br>为换行标记），选项内容与原题一致
3. 答案：仅单个字母（A/B/C/D，大写，无其他字符）
4. 解析：完整保留原题解析，说明正确答案依据及错误选项原因，公式保留MathJax格式
5. 在保留内容的前提下，每一道题目涉及到的题干、选项、答案、解析全都要调整到同一行里
格式示例：
{example_text}

选择题文本：{questions_text}"""

        response = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[
                {"role": "system", "content": "你是Anki格式转换专家，严格按要求输出4字段Tab分隔内容，不添加任何解释、标题或多余字符，每一题调整集中到一行里，一行一题"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.01  # 低随机性确保格式严谨
        )
        
        # 处理空响应异常
        if not response.choices:
            print(f"❌ {file_name} - 未获取到Anki转换结果")
            return None
            
        anki_content = response.choices[0].message.content
        # 先做基础验证，再做格式验证
        if not validate_generated_content(anki_content, "Anki转换", file_name):
            return None
        return anki_content
    except Exception as e:
        print(f"❌ {file_name} - Anki转换失败: {e}")
        return None

def validate_anki_format(content, file_name):
    return True

def save_anki_file(content, output_path):
    try:
        # 确保输出目录存在（处理嵌套目录）
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"💾 Anki格式文件已保存至：{output_path}")
        return True
    except Exception as e:
        print(f"❌ 保存Anki文件失败: {e}")
        return False

def process_single_file(input_path, input_dir, output_dir):
    """处理单个文件：读取→转换→校验→保存→标记已处理"""
    file_name = os.path.basename(input_path)
    print(f"\n===== 开始处理：{file_name} =====")
    
    # 跳过已处理文件（带.processed.txt后缀）
    if input_path.endswith(".processed.txt"):
        print(f"⏭️ {file_name} 已处理完成，跳过")
        return True
        
    # 读取并提取“”后的内容
    target_text = read_questions_text(input_path)
    if not target_text:
        return False
    
    # 转换为Anki格式
    anki_content = format_for_anki(target_text, file_name)
    if not anki_content:
        return False
    
    # 严格校验格式，不通过则不保存
    if not validate_anki_format(anki_content, file_name):
        print(f"❌ {file_name} - 格式校验未通过，不保存")
        return False
    
    # 保留相对路径结构
    relative_path = os.path.relpath(input_path, input_dir)
    relative_dir, orig_file_name = os.path.split(relative_path)
    output_subdir = os.path.join(output_dir, relative_dir)
    os.makedirs(output_subdir, exist_ok=True)
    
    # 构建输出文件名
    base_name = os.path.splitext(orig_file_name)[0].replace(".生成选择题", "")
    output_file_name = f"{base_name}.Anki格式.txt"
    output_path = os.path.join(output_subdir, output_file_name)
    
    # 保存并标记原文件为已处理
    if save_anki_file(anki_content, output_path):
        processed_path = f"{input_path}.processed.txt"
        os.rename(input_path, processed_path)
        print(f"✅ {file_name} - 已标记为处理完成（{os.path.basename(processed_path)}）")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description='vLLM版电路选择题Anki格式转换器')
    parser.add_argument('-n', '--num_workers', type=int, default=32, 
                        help='并行线程数（建议1-4，根据vLLM服务器CPU/GPU资源调整）')
    parser.add_argument('-i', '--input_dir', default=DEFAULT_INPUT_DIR, help='选择题输入目录')
    parser.add_argument('-o', '--output_dir', default=DEFAULT_OUTPUT_DIR, help='Anki格式输出目录')
    
    # 兼容Jupyter等环境的未知参数
    args, unknown = parser.parse_known_args()
    
    # 创建输出根目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 收集待处理文件
    txt_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.生成选择题.txt') and not file.endswith('.processed.txt'):
                file_path = os.path.join(root, file)
                if not file_path.endswith('.processed.txt'):
                    txt_files.append(file_path)
    
    if not txt_files:
        print("📭 未找到待转换的选择题文件（已排除带.processed.txt后缀的文件）")
        return
    
    print(f"📊 发现 {len(txt_files)} 个待转换文件，将使用 {args.num_workers} 线程并行处理")
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_file, f, args.input_dir, args.output_dir) for f in txt_files]
        success_count = sum(1 for future in as_completed(futures) if future.result())
    
    # 输出处理结果
    print(f"\n===== 处理完成 ======")
    print(f"✅ 成功转换并保存：{success_count} 个文件")
    print(f"❌ 转换失败（含格式错误）：{len(txt_files) - success_count} 个文件")
    print(f"📁 Anki格式输出根路径：{args.output_dir}")
    print(f"🔍 已处理文件均添加了 .processed.txt 后缀，下次运行自动跳过")

if __name__ == "__main__":
    main()