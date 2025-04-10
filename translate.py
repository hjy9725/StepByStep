import re
import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator
import time
from tqdm import tqdm

import nltk

# 预下载资源（在代码最前面添加）
try:
    sent_tokenize("This is a test.")
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class PDFTranslator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_text = ""
        self.sentences = []
        self.translations = []
        
    def extract_text(self):
        """使用PyMuPDF提取文本并保留布局信息"""
        doc = fitz.open(self.file_path)
        for page in doc:
            self.raw_text += page.get_text("text") + "\n"
        return self.clean_text(self.raw_text)

    def clean_text(self, text):
        """文本预处理"""
        # 合并被错误分割的单词
        text = re.sub(r'-\n(\w+)', r'\1', text)  
        # 替换多余换行
        text = re.sub(r'\n{2,}', '\n\n', text)  
        return text.strip()

    def split_sentences(self):
        """智能分句并保留段落结构"""
        paragraphs = self.raw_text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # 分句时保留段落上下文
                sentences = sent_tokenize(para.replace('\n', ' '))
                self.sentences.extend(sentences)
        return self.sentences

    def translate(self, batch_size=10, timeout=5, max_retries=3):
        """带错误处理和进度显示的翻译功能"""
        self.translations = []
        for i in tqdm(range(0, len(self.sentences), batch_size)):
            batch = self.sentences[i:i+batch_size]
            translated = []
            for sent in batch:
                for attempt in range(max_retries):
                    try:
                        result = GoogleTranslator(
                            source='auto', 
                            target='zh-CN'
                        ).translate(sent[:5000])  # 截断超长文本
                        translated.append(result)
                        break
                    except Exception as e:
                        if attempt == max_retries -1:
                            translated.append("[翻译失败]")
                            print(f"错误：{str(e)} - 原文：{sent[:50]}...")
                        time.sleep(timeout)
            self.translations.extend(translated)
        return self.translations

    def generate_output(self, output_format="md"):
        """生成双语对照文档"""
        output = []
        for idx, (orig, trans) in enumerate(zip(self.sentences, self.translations)):
            output.append(f"[{idx+1:03d}] {orig}\n[{idx+1:03d}] {trans}\n")
            output.append("-"*50 + "\n")
        
        if output_format == "md":
            with open("output.md", "w", encoding="utf-8") as f:
                f.write("\n".join(output))
        elif output_format == "html":
            html = ["<div class='bilingual'>"]
            for item in output:
                # 修改后的代码段
                processed_item = item.replace('\n', '<br>')
                html.append(f"<p>{processed_item}</p>")
            html.append("</div>")
            with open("output.html", "w", encoding="utf-8") as f:
                f.write("\n".join(html))
        return output

# 使用示例
if __name__ == "__main__":
    translator = PDFTranslator("D:\\StepByStep\\input.pdf")
    
    # 步骤1: 提取文本
    print("正在提取文本...")
    raw_text = translator.extract_text()
    
    # 步骤2: 分句处理
    print("正在分句...")
    translator.split_sentences()
    
    # 步骤3: 执行翻译
    print("开始翻译...")
    translator.translate(batch_size=8)  # 小批量处理降低错误率
    
    # 步骤4: 生成输出
    print("生成文档...")
    translator.generate_output(output_format="md")
    
    print("完成！已保存至output.md")