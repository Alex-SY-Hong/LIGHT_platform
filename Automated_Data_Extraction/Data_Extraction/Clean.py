# 文本清洗

import pylogg
from TextNormalizer import TextNormalizer, normText
from lxml import etree

# 初始化日志（可根据需要注释）
log = pylogg.New('paragraph')

class ParagraphParser(object):
    def __init__(self, name=None, debug=False):
        self.name = name
        self.text = None
        self.body = None
        self.debug = debug
        self.normalizer = TextNormalizer()  # 初始化 TextNormalizer 实例

    def _log(self, message):
        if self.debug:
            print(f"[{self.name or 'Unnamed'}] {message}")

    def _is_reference(self, text: str) -> bool:
        """更精确地判断是否为参考文献（只判断长度较短且包含关键词）"""
        if text is None:
            return False
        text_lower = text.lower()
        
        # 修改：仅对文本中出现 DOI 或 URL 并且长度较短的文本判定为参考文献
        if any(kw in text_lower for kw in ["http", "doi"]) and len(text) < 150:
            if len(text.split()) < 10:  # 进一步细化：假如是一个很短的文本且包含 DOI，则判定为参考文献
                return True
        return False

    def _clean_text(self, text: str) -> str:
        """Clean up a text string using both normText() and TextNormalizer()."""
        if text is None:
            return ""
        
        if self._is_reference(text):
            return ""  # 引用直接跳过
        
        # 第一步：先做结构和语义上的规范处理
        text = self.normalizer.normalize(text)
        # 第二步：做基础字符清洗
        text = normText(text)
    
        return text

    def parse(self, paragraph_text: str):
        """解析纯文本段落"""
        self.body = paragraph_text
        self.text = self._clean_text(paragraph_text)
        if self.debug:
            if not self.text.strip():
                print("Warning: Cleaned text is empty!")

    def is_valid(self) -> bool:
        text = self.text.strip()
        
        # ✅ 英文 + 中文标点一起判断
        contains_punctuation = any(p in text for p in "。！？.?!")
        
        # ✅ 长度和单词数要求稍提高，避免短语误判
        is_long_enough = len(text) > 30
        num_words = len(text.split())
    
        return contains_punctuation and is_long_enough and num_words > 10


    def save(self, outfile):
        with open(outfile, "w+", encoding="utf-8") as fp:
            fp.write(self.text or "")
            fp.write("\n")
        print("Save OK:", outfile)
