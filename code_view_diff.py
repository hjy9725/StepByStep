import os
import re
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== vLLM配置（适配4B模型）====================
VLLM_BASE_URL = "http://localhost:8000/v1"
TARGET_MODEL = "/t9k/mnt/hjy/Qwenq/qwen3-4b-thinking-2507"  # 与启动命令的--model路径一致
MAX_TOKENS = 18000  
# ==========================================================
# 路径配置（保持与你的服务器一致）
DEFAULT_INPUT_DIR = r"F:\同步文件夹\考研pdf\电路大合集"  # 输入原始txt文件目录
DEFAULT_OUTPUT_DIR = r"F:\同步文件夹\Anki输出文件夹\电路大合集\优化文本"  # 优化文本输出目录
# ==========================================================

# 初始化vLLM客户端
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="EMPTY",
    timeout=1800  # 超时时间延长，适应长文本处理
)

def read_raw_text(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip() or None  # 读取完整文本，不做截断
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def validate_generated_content(content, task_name, file_name):
    if not content or len(content.strip()) < 50:
        print(f"❌ {file_name} - {task_name} 内容无效")
        return False
    return True

def optimize_text(raw_text, file_name):
    print(f"📝 {file_name} - 优化文本...")
    try:
        # 移除长度限制，传入完整文本
        prompt = f"""你是电路学科专家，优化以下文本：
1. 补充推导过程，公式用MathJax（如\(U=IR\)）
2. 扩展核心知识点、易错点
3. 段落清晰，公式单独成行

原始文本：{raw_text}"""

        response = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[
                {"role": "system", "content": "优化理工科文本，精准专业"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,  # 使用模型最大支持的token数
            temperature=0.4
        )
        optimized = response.choices[0].message.content
        return optimized if validate_generated_content(optimized, "文本优化", file_name) else None
    except Exception as e:
        print(f"❌ 文本优化失败: {e}")
        return None

def save_optimized_file(content, output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"💾 优化文本保存至：{output_path}")
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False

def process_single_file(input_path, input_dir, output_dir):
    """处理单个文件：读取→优化→保存→标记已处理"""
    file_name = os.path.basename(input_path)
    print(f"\n===== 开始处理：{file_name} =====")
    
    # 跳过已处理文件（后缀为.processed.txt）
    if file_name.endswith(".processed.txt"):
        print(f"⏭️ {file_name} 已优化完成，跳过")
        return True
        
    # 读取原始文本
    raw_text = read_raw_text(input_path)
    if not raw_text:
        return False
    
    optimized = optimize_text(raw_text, file_name)
    if not optimized:
        return False
    
    # 保持原始目录结构
    relative_path = os.path.relpath(input_path, input_dir)
    relative_dir, orig_file_name = os.path.split(relative_path)
    output_subdir = os.path.join(output_dir, relative_dir)
    os.makedirs(output_subdir, exist_ok=True)
    
    base_name = os.path.splitext(orig_file_name)[0]
    output_file_name = f"{base_name}.优化文本.txt"
    output_path = os.path.join(output_subdir, output_file_name)
    
    if save_optimized_file(optimized, output_path):
        # 成功保存后，给原文件添加.processed后缀
        processed_path = f"{input_path}.processed.txt"
        os.rename(input_path, processed_path)
        print(f"🔖 原文件已标记为处理完成：{processed_path}")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description='vLLM版电路文本优化器')
    parser.add_argument('-n', '--num_workers', type=int, default=5, help='并行线程数')
    parser.add_argument('-i', '--input_dir', default=DEFAULT_INPUT_DIR, help='原始文本输入目录')
    parser.add_argument('-o', '--output_dir', default=DEFAULT_OUTPUT_DIR, help='优化文本输出目录')
    
    args, unknown = parser.parse_known_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    txt_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            # 只处理txt文件，且排除已添加.processed后缀的文件
            if file.endswith('.txt') and not file.endswith('.processed.txt'):
                txt_files.append(os.path.join(root, file))
    
    if not txt_files:
        print("📭 未找到待优化的txt文件（已排除.processed.txt后缀的已处理文件）")
        return
    
    print(f"发现 {len(txt_files)} 个文件，使用 {args.num_workers} 线程处理")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_file, f, args.input_dir, args.output_dir) for f in txt_files]
        success_count = sum(1 for future in as_completed(futures) if future.result())
    
    print(f"\n===== 处理完成 =====")
    print(f"✅ 成功优化：{success_count} 个文件")
    print(f"❌ 优化失败：{len(txt_files) - success_count} 个文件")
    print(f"📁 优化文本输出根路径：{args.output_dir}")

if __name__ == "__main__":
    main()