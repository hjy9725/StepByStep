import os
from pdf2image import convert_from_path

def pdf_to_png(pdf_path, dpi=300, first_page=1, last_page=None):
    """
    将PDF文件转换为PNG图片，并保存到与PDF同名的文件夹中
    
    参数:
    pdf_path (str): PDF文件路径
    dpi (int): 图片分辨率，默认300
    first_page (int): 起始页码，默认1
    last_page (int): 结束页码，默认None表示最后一页
    """
    
    # 获取PDF文件的绝对路径
    pdf_abs_path = os.path.abspath(pdf_path)
    
    # 获取PDF文件所在的目录
    pdf_dir = os.path.dirname(pdf_abs_path)
    
    # 获取PDF文件名（不含扩展名）
    pdf_filename = os.path.splitext(os.path.basename(pdf_abs_path))[0]
    
    # 创建完整的输出目录路径：PDF所在目录 + 无后缀的PDF文件名
    output_dir = os.path.join(pdf_dir, pdf_filename)
    
    # 创建输出目录，如果已存在则不报错
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换PDF所有页为PIL Image对象
    print(f"正在转换 {pdf_path} ...")
    pages = convert_from_path(
        pdf_path=pdf_path,
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
        fmt="png"
    )
    
    # 保存所有页为PNG文件到创建的文件夹中
    for i, page in enumerate(pages, start=first_page):
        output_path = os.path.join(output_dir, f"page_{i}.png")
        page.save(output_path, "PNG")
        print(f"已保存: {output_path}")
    
    print(f"转换完成，共 {len(pages)} 页，保存至 {os.path.abspath(output_dir)}")

def process_pdf_directory(directory, dpi=300):
    """
    处理指定目录下的所有PDF文件，只处理未添加processed后缀的文件
    处理完成后给原PDF文件添加processed后缀
    
    参数:
    directory (str): 要处理的目录路径
    dpi (int): 图片分辨率，默认300
    """
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查是否是PDF文件且未被处理过
        if filename.endswith('.pdf') and not filename.endswith('processed.pdf'):
            pdf_path = os.path.join(directory, filename)
            
            # 检查是否是文件而不是目录
            if os.path.isfile(pdf_path):
                # 转换PDF为PNG
                pdf_to_png(pdf_path, dpi=dpi)
                
                # 处理完成后重命名原文件，添加processed后缀
                base_name = os.path.splitext(filename)[0]
                new_filename = f"{base_name}_processed.pdf"
                new_path = os.path.join(directory, new_filename)
                
                # 重命名文件
                os.rename(pdf_path, new_path)
                print(f"已重命名处理过的文件: {new_filename}")
    
    print("目录中所有未处理的PDF文件已处理完毕")

if __name__ == "__main__":
    # 指定要处理的PDF目录
    pdf_directory = r"F:\同步文件夹\考研pdf"  # 替换为你的PDF目录路径
    
    # 处理目录中的所有PDF文件
    process_pdf_directory(pdf_directory, dpi=300)