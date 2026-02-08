import re
from aqt import mw
from aqt.qt import *
from aqt.editor import Editor
from aqt import gui_hooks

def replace_math_in_note(editor: Editor):
    """扫描当前笔记的所有字段并替换 $$ 和 $ 为 Anki 格式"""
    note = editor.note
    if not note:
        return

    # 1. 尝试获取光标位置 (适配新旧版本 Anki)
    # Anki 24+ 使用 current_field，旧版使用 currentField
    current_idx = getattr(editor, "current_field", getattr(editor, "currentField", 0))

    # 2. 定义正则替换规则 (注意：这里必须是双反斜杠)
    # 替换 $$...$$ 为 \[...\]
    display_pattern = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
    
    # 替换 $...$ 为 \(...\) 
    # 解释：(?<!\\) 表示前面不能有反斜杠，避免误伤 \$ 转义符
    inline_pattern = re.compile(r"(?<!\\)\$(.*?)(?<!\\)\$")

    changes = False
    for i in range(len(note.fields)):
        content = note.fields[i]
        
        # 先处理块级公式
        new_content = display_pattern.sub(r"\[\1\]", content)
        # 再处理行内公式
        new_content = inline_pattern.sub(r"\(\1\)", new_content)
        
        if new_content != content:
            note.fields[i] = new_content
            changes = True

    if changes:
        # 3. 刷新编辑器 (适配新旧版本 Anki)
        # Anki 25+ 使用 load_note，旧版使用 loadNote
        if hasattr(editor, "load_note"):
            editor.load_note()
        elif hasattr(editor, "loadNote"):
            editor.loadNote()
        
        # 4. 尝试恢复光标焦点
        if current_idx is not None:
             editor.web.eval(f"focusField({current_idx});")

        mw.statusBar().showMessage("✅ 数学符号转换完成！", 2000)
    else:
        mw.statusBar().showMessage("⚠️ 未发现需要转换的内容", 2000)

def on_editor_did_init_buttons(buttons, editor):
    """在新版 Anki 中正确的添加按钮方式"""
    btn = editor.addButton(
        icon=None, 
        cmd="math_fix_click", 
        func=lambda ed=editor: replace_math_in_note(ed),
        tip="一键转换全字段 MathJax (Ctrl+Shift+M)",
        label="∑" 
    )
    buttons.append(btn)
    
    # 绑定快捷键
    editor._math_shortcut = QShortcut(QKeySequence("Ctrl+Shift+M"), editor.widget)
    editor._math_shortcut.activated.connect(lambda: replace_math_in_note(editor))

# 注册钩子
gui_hooks.editor_did_init_buttons.append(on_editor_did_init_buttons)