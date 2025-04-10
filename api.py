import re
import fitz
import time
import json
import requests
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

class DoubaoTranslator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://ml.volces.com/api/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.cost_tracker = 0.0  # 成本跟踪器（单位：元）

    def translate(self, text, max_retries=3):
        """智能分批翻译（自动处理长文本）"""
        payload = {
            "model": "doubao-lite-4k",
            "messages": [{
                "role": "user",
                "content": f"将以下英文文本精确翻译为简体中文，保留专业术语：\n{text}"
            }],
            "temperature": 0.1
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=10
                )
                resp_data = response.json()
                
                # 成本计算（输入+输出 tokens）
                usage = resp_data.get('usage', {})
                self.cost_tracker += (usage.get('total_tokens', 0) / 1_000_000) * 0.3
                
                return resp_data['choices'][0]['message']['content'].strip()
            except Exception as e:
                if attempt == max_retries -1:
                    print(f"翻译失败: {str(e)}")
                    return "[翻译失败]"
                time.sleep(2**attempt)  # 指数退避

class PDFTranslator:
    def __init__(self, file_path, api_key):
        self.file_path = file_path
        self.translator = DoubaoTranslator(api_key)
        self.raw_text = ""
        self.sentences = []
        self.translations = []
    
    # [保持原有的 extract_text, clean_text, split_sentences 方法不变]
 
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
    
    def process_translation(self, batch_size=5):
        """带流量控制的翻译处理"""
        total = len(self.sentences)
        with tqdm(total=total, desc="翻译进度") as pbar:
            for i in range(0, total, batch_size):
                batch = self.sentences[i:i+batch_size]
                combined_text = "\n".join(batch)
                
                # 翻译批次文本
                translated = self.translator.translate(combined_text)
                
                # 拆分翻译结果（假设按换行分割）
                translated_sents = translated.split('\n')
                
                # 对齐处理
                min_len = min(len(batch), len(translated_sents))
                self.translations.extend(translated_sents[:min_len])
                
                # 处理长度不匹配情况
                if len(translated_sents) < len(batch):
                    self.translations.extend(["[未翻译]"]*(len(batch)-min_len))
                
                pbar.update(len(batch))
                
                # 遵守速率限制（根据API调整）
                time.sleep(0.5 if batch_size <=5 else 1)

    def generate_report(self):
        """生成成本报告"""
        chars_count = sum(len(s) for s in self.sentences)
        approx_tokens = chars_count // 3  # 粗略估算（1 token ≈ 3字符）
        cost = (approx_tokens / 1_000_000) * 0.3
        return f"""翻译报告：
- 处理句子数：{len(self.sentences)}
- 估算字符数：{chars_count}
- 估算Tokens数：{approx_tokens}
- 预估成本：¥{cost:.4f} 元
- 实际API成本：¥{self.translator.cost_tracker:.4f} 元
"""

# 使用示例
if __name__ == "__main__":
    API_KEY = "XXXXXXX"  # 替换为真实API密钥
    
    translator = PDFTranslator("1input.pdf", API_KEY)
    translator.extract_text()
    translator.split_sentences()
    
    print("开始翻译...")
    translator.process_translation(batch_size=5)  # 小批次更安全
    
    print("生成文档...")
    translator.generate_output()
    
    print(translator.generate_report())