from wxauto_old import WeChat
from openai import OpenAI
import time
import os
import base64
import glob
import pyautogui
import json
import re
from datetime import datetime
from collections import deque
import uiautomation as auto
from PIL import Image
import google.generativeai as genai

# 配置 Gemini
try:
    genai.configure(api_key=config.GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)
    print(f"🧠 已加载 Gemini 大脑 ({config.GEMINI_MODEL})")
except Exception as e:
    print(f"❌ Gemini 配置失败: {e}")

# 导入配置
try:
    import chat_config as config
except ImportError:
    print("❌ 错误：找不到 chat_config.py")
    exit()

# ==================== 🛠️ 初始化 ====================
print(f"🔄 连接豆包 (Doubao)...")
client = OpenAI(api_key=config.VOLC_API_KEY, base_url=config.VOLC_BASE_URL)
wx = WeChat()
chat_memories = deque(maxlen=15)
chat_memories.append({"role": "system", "content": config.SYSTEM_PROMPT})

print(f"🚀 {config.BOT_NAME} v18.0 (视觉Agent版) 已启动")

# ==================== 🧠 新的大脑逻辑 ====================
def generate_reply_with_gemini(text_history, image_descriptions):
    """
    使用 Gemini 1.5 Flash 综合上下文生成像人的回复
    text_history: 列表，最近的几条文字消息
    image_descriptions: 列表，视觉模型提取的图片内容
    """
    
    # 1. 构建一个极其丰富的 Prompt Context
    # 把“视觉情报”转化成“旁白”，让 Gemini 知道发生了什么
    context_str = "【当前状况】\n"
    
    if text_history:
        context_str += f"她刚才发的消息：{' '.join(text_history)}\n"
    else:
        context_str += "她刚才没发文字，直接甩了图片过来。\n"
        
    if image_descriptions:
        context_str += "\n【她发的图片内容（由视觉模块提取）】\n"
        for i, desc in enumerate(image_descriptions):
            context_str += f"---图片{i+1}---\n{desc}\n"
    
    context_str += "\n请根据以上信息，以男朋友的口吻回复。如果图片内容很无聊（比如普通的UI截图），可以敷衍或者只回一个表情。如果有槽点（比如雾霾严重、数学太难），请狠狠吐槽。"

    # 2. 调用 Gemini
    try:
        # 将系统人设 + 当前语境 结合
        # Gemini 的 chat session 能够很好地保持人设
        chat = gemini_model.start_chat(history=[
            {"role": "user", "parts": config.SYSTEM_PROMPT},
            {"role": "model", "parts": "明白了，我是她男朋友，说话直白、带点损、不复读图片内容。"}
        ])
        
        response = chat.send_message(context_str)
        reply = response.text
        
        # 清洗 Gemini 可能带出的多余格式
        reply = reply.replace("\n", "||").replace("回复：", "")
        return reply
        
    except Exception as e:
        print(f"⚠️ Gemini 思考短路: {e}")
        return "..."

# ==================== 🖼️ 视觉与操作模块 ====================

def bring_wechat_to_front():
    """ 强制唤醒微信窗口到最前 """
    try:
        # 使用 uiautomation 查找窗口，比 pywin32 更稳定
        wechat_win = auto.WindowControl(ClassName='WeChatMainWndForPC')
        if wechat_win.Exists(0):
            wechat_win.SetActive()
            wechat_win.SetTopmost(True)
            time.sleep(0.1)
            wechat_win.SetTopmost(False) # 取消置顶，免得挡住操作
            print("🖥️ 微信已唤出")
            return True
        else:
            print("❌ 未找到微信窗口")
            return False
    except Exception as e:
        print(f"⚠️ 唤窗失败: {e}")
        return False

def take_screenshot():
    """ 截取全屏并保存 """
    try:
        img_path = config.TEMP_SCREENSHOT_PATH
        # 截取全屏
        pyautogui.screenshot(img_path)
        return img_path
    except Exception as e:
        print(f"❌ 截图失败: {e}")
        return None

def get_click_coordinates_from_ai(screenshot_path):
    """ 
    🧠 核心逻辑：让豆包VL看截图，返回需要点击的坐标 
    """
    print("🤖 AI正在分析屏幕寻找图片...")
    try:
        with open(screenshot_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # 构造一个非常具体的 Prompt，要求返回 JSON
        prompt = f"""
        这是一张电脑全屏截图(分辨率{config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT})。
        请你的任务是找到微信聊天窗口中，**对方发送的图片缩略图**，并返回它们中心点的点击坐标。

        请严格按照以下步骤进行观察和筛选：
        1.  **定位聊天区域**：首先忽略屏幕最左侧的微信联系人/功能列表栏（深色背景区域）。将注意力集中在右侧的白色聊天消息详情区域。
        2.  **识别目标**：在聊天区域中，寻找对方（显示在左侧）发送的消息气泡。
        3.  **筛选图片**：在这些气泡中，挑出内容是图片缩略图的气泡。它们通常是矩形的照片或截图。
        4.  **排除干扰**：
            * 不要包含我发送的图片（显示在右侧，绿底气泡）。
            * 不要包含小的表情包。
            * **绝对不要**识别最左侧联系人列表里的任何元素。
        5.  **坐标要求**：返回的坐标必须位于聊天内容区域内。这意味着 **x 坐标通常应该大于 650**（跳过左侧列表栏）。

        请直接返回一个JSON格式的坐标列表，包含每个符合条件的图片气泡的中心点坐标 [x, y]。
        顺序从上到下。如果没有找到符合条件的图片，返回空列表 []。

        格式示例（注意 x 坐标的值）：
        [[450, 500], [450, 800]]
        
        只返回纯JSON数据，不要任何解释或废话。
        """

        resp = client.chat.completions.create(
            model=config.VOLC_VL_ENDPOINT_ID,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}]
        )
        
        content = resp.choices[0].message.content
        print(f"🔍 AI返回: {content}")
        
        # 清洗数据，提取JSON
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            coords = json.loads(json_match.group())
            return coords
        return []
    except Exception as e:
        print(f"⚠️ AI视觉分析出错: {e}")
        return []

def smart_click_images(coords):
    """ 执行点击操作 """
    if not coords:
        print("🤷‍♂️ AI没看到需要点击的图片")
        return

    print(f"🖱️ 准备点击 {len(coords)} 张图片...")
    original_pos = pyautogui.position()
    
    for i, (x, y) in enumerate(coords):
        # 安全校验
        if x < 0 or x > config.SCREEN_WIDTH or y < 0 or y > config.SCREEN_HEIGHT:
            continue
            
        print(f"   -> 点击第 {i+1} 张: ({x}, {y})")
        pyautogui.click(x, y, clicks=2, interval=0.1) # 双击查看原图(触发缓存)
        time.sleep(1.5) # 等待大图加载
        pyautogui.press('esc') # 关闭大图查看器
        time.sleep(0.5) # 等待动画
        
    pyautogui.moveTo(original_pos) # 归位

# ==================== 🔐 DAT与回复模块 (复用优化) ====================

def decrypt_dat_file(dat_path):
    """ 解密 DAT 文件 """
    try:
        with open(dat_path, 'rb') as f: content = f.read()
        if not content: return None
        key = content[0] ^ 0xFF
        decrypted = bytearray([b ^ key for b in content])
        
        ext = ".jpg"
        if decrypted[0] == 0x89 and decrypted[1] == 0x50: ext = ".png"
        elif decrypted[0] == 0x47 and decrypted[1] == 0x49: ext = ".gif"
        
        save_dir = os.path.join(os.getcwd(), "temp_decoded")
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        filename = f"dec_{int(time.time())}_{os.path.basename(dat_path)}{ext}"
        save_path = os.path.join(save_dir, filename)
        
        with open(save_path, "wb") as f_out: f_out.write(decrypted)
        return save_path
    except: return None

def find_latest_hd_images(since_time):
    """ 查找时间戳之后生成的大的DAT文件 """
    try:
        current_month = datetime.now().strftime("%Y-%m")
        search_pattern = os.path.join(config.WECHAT_IMAGE_ROOT, "MsgAttach", "**", "Image", current_month, "*.dat")
        files = glob.glob(search_pattern, recursive=True)
        
        valid_files = []
        for f in files:
            mtime = os.path.getmtime(f)
            if mtime > since_time:
                # 过滤掉小于20KB的文件 (通常是缩略图)
                if os.path.getsize(f) > 20 * 1024:
                    valid_files.append((f, mtime))
        
        # 按时间排序
        valid_files.sort(key=lambda x: x[1])
        return [f[0] for f in valid_files]
    except: return []

def get_doubao_vl_description(image_path):
    """ 豆包看图描述 """
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        resp = client.chat.completions.create(
            model=config.VOLC_VL_ENDPOINT_ID,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "简要直白描述图片，如果有文字请提取。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}]
        )
        return resp.choices[0].message.content
    except: return "（图片解析失败）"

def generate_reply(context, is_img=False):
    """ 生成回复 """
    if is_img: prompt = f"[用户发图] 视觉内容：{context}。请根据内容回复。"
    else: prompt = context
    
    chat_memories.append({"role": "user", "content": prompt})
    try:
        resp = client.chat.completions.create(
            model=config.VOLC_TEXT_ENDPOINT_ID,
            messages=list(chat_memories),
            temperature=0.8, max_tokens=200
        )
        reply = resp.choices[0].message.content
        chat_memories.append({"role": "assistant", "content": reply})
        return reply
    except: return None

# ==================== 🔄 主逻辑 (防抖动版) ====================

def main():
    try: wx.ChatWith(config.TARGET_USER)
    except: pass
    
    last_processed_count = len(wx.GetAllMessage())
    last_msg_timestamp = time.time()
    
    # 状态标记：是否有未处理的新消息
    pending_new_msgs = False
    
    # 记录上一次扫描图片的时间，只处理这之后生成的新文件
    last_scan_time = time.time()

    print(f"⏱️ 监控已启动 | 响应延迟: {config.BATCH_WAIT_SECONDS}秒")

    while True:
        try:
            current_msgs = wx.GetAllMessage()
            current_len = len(current_msgs)
            
            # 1. 检测是否有新消息
            if current_len > last_processed_count:
                # 只要有新消息，就重置计时器
                last_msg_timestamp = time.time()
                pending_new_msgs = True
                
                # 获取最新的一条简单打印一下
                new_msg_content = current_msgs[-1].content
                print(f"\r📩 收到新消息 ({datetime.now().strftime('%H:%M:%S')}): {new_msg_content} | 等待发送结束...", end="")
                
                last_processed_count = current_len
            
            # 2. 判断是否满足“静默时间”且有待处理消息
            # 逻辑：(当前时间 - 最后消息时间 > 设定阈值) AND (有未处理消息)
            time_since_last_msg = time.time() - last_msg_timestamp
            
            if pending_new_msgs and time_since_last_msg > config.BATCH_WAIT_SECONDS:
                print(f"\n\n⚡ 对方已{config.BATCH_WAIT_SECONDS}秒未发消息，开始执行批处理...")
                
                # A. 唤起微信
                bring_wechat_to_front()
                time.sleep(0.5)
                
                # B. 视觉全屏识别 & 点击 (这是为了触发高清图下载)
                # 只有当最新几条消息里包含 "[图片]" 时才执行这个昂贵的操作
                recent_msgs = current_msgs[-5:] # 检查最近5条
                has_image = any(msg.content == '[图片]' for msg in recent_msgs)
                
                if has_image:
                    print("📸 检测到聊天记录含图片，启动视觉点击...")
                    screenshot = take_screenshot()
                    if screenshot:
                        coords = get_click_coordinates_from_ai(screenshot)
                        smart_click_images(coords)
                
                # C. 处理已下载的高清图
                # 查找从 last_scan_time 到现在新生成的DAT文件
                new_dat_paths = find_latest_hd_images(last_scan_time)
                image_descriptions = []
                
                if new_dat_paths:
                    print(f"📂 发现 {len(new_dat_paths)} 张高清大图，开始解析...")
                    for dat in new_dat_paths:
                        decrypted = decrypt_dat_file(dat)
                        if decrypted:
                            desc = get_doubao_vl_description(decrypted)
                            print(f"   - 图片内容: {desc}")
                            image_descriptions.append(desc)
                
                # 更新扫描时间锚点
                last_scan_time = time.time()
                
                # # D. 统合回复
                # # 将最后几条纯文本消息和图片描述合并给AI
                # text_context = [m.content for m in recent_msgs if m.content != '[图片]' and m.sender == config.TARGET_USER]
                
                # full_prompt = ""
                # if text_context:
                #     full_prompt += f"她发的文字: {','.join(text_context)}。\n"
                # if image_descriptions:
                #     full_prompt += f"她发的图片内容: {'; '.join(image_descriptions)}。"
                
                # if full_prompt:
                #     print("🧠 生成回复中...")
                #     reply = generate_reply(full_prompt, is_img=bool(image_descriptions))
                    
                #     if reply:
                #         for part in reply.split("||"):
                #             if part.strip():
                #                 wx.SendMsg(part.strip())
                #                 print(f"🗣️ 回复: {part.strip()}")
                #                 time.sleep(1)

                # D. 统合回复
                # 获取最近3条纯文字消息作为背景
                recent_text_msgs = [m.content for m in recent_msgs if m.content != '[图片]' and m.sender == config.TARGET_USER]
                
                # 只有当有图片描述 或者 有文字消息时才回复
                if image_descriptions or recent_text_msgs:
                    print("🧠 Gemini 正在构思骚话...")
                    
                    # 关键修改：传入文字历史 + 图片描述
                    reply = generate_reply_with_gemini(recent_text_msgs, image_descriptions)
                    
                    if reply:
                        for part in reply.split("||"):
                            p = part.strip()
                            if p:
                                wx.SendMsg(p)
                                print(f"🗣️ 回复: {p}")
                                time.sleep(random.uniform(1.0, 2.5)) # 随机延迟，更像真人

                # 重置状态
                pending_new_msgs = False
                print(f"✅ 批处理完成，继续监控...")

            time.sleep(1) # 循环心跳

        except KeyboardInterrupt: break
        except Exception as e:
            print(f"⚠️ 主循环报错: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()