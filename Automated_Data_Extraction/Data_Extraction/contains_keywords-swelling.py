import re

def contains_keywords(text, keyword_dict):
    """
    判断文本是否包含关键词，并返回匹配状态和命中类别及关键词信息

    参数:
        text (str): 要检查的文本
        keyword_dict (dict): 关键词字典 {类别名: [同义词列表]}

    返回:
        matched (bool): 是否命中至少一个关键词
        matched_categories (list): 命中的类别名称
        matched_keywords (dict): 每类下命中的具体关键词 {类别: [关键词1, 关键词2]}
    """
    matched_categories = []
    matched_keywords = {}

    for category, synonyms in keyword_dict.items():
        found = []
        for word in synonyms:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, text, flags=re.IGNORECASE):
                found.append(word)
        if found:
            matched_categories.append(category)
            matched_keywords[category] = found

    return bool(matched_categories), matched_categories, matched_keywords
