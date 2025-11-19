import re

# PLA力学性能关键词字典
keyword_dict = {
    "Polymers": [
        # 通用聚合物及共聚物
        "polymer", "copolymer", "blend", "biopolymer",
        # 常见商用或合成聚合物
        "PLA", "polylactic acid", "PCL", "polycaprolactone",
        "PET", "polyethylene terephthalate", "PMMA", "polymethyl methacrylate",
        "PU", "polyurethane", "PA", "polyamide", "nylon",
        "PVC", "polyvinyl chloride", "PVA", "polyvinyl alcohol",
        "PAN", "polyacrylonitrile", "PVP", "polyvinylpyrrolidone",
        "PDMS", "polydimethylsiloxane", "PC", "polycarbonate",
        "PBS", "polybutylene succinate", "PGA", "poly(γ-glutamic acid)",
        "PE", "polyethylene", "PP", "polypropylene", "polyester",
        # 生物基/天然聚合物
        "GelMA", "gelatin methacrylate", "gelatin", "collagen",
        "chitosan", "alginate", "sodium alginate", "cellulose",
        "nanocellulose", "pectin", "lignin", "starch", "hyaluronic acid",
        "silk fibroin", "polyethylene glycol", "polydopamine",
        "polyacrylamide"
    ],
    "Additives or Modifiers": [
        # 基础功能添加剂
        "additive", "modifier", "plasticizer", "compatibilizer", "filler", "blend",
        "hybrid", "nanocomposite", "composite",
        # 无机纳米材料
        "nanoparticle", "nanofiller", "nanoclay", "TiO2", "SiO2", "ZnO", "CaCO3",
        "clay", "montmorillonite", "halloysite", "bentonite",
        # 有机/高分子共混物
        "PBAT", "PEG", "PHA", "PBSA", "PPC", "EVA", "PLA-g-MA",
        # 生物基/天然材料
        "cellulose nanocrystal", "microcrystalline cellulose", "hemicellulose",
        "soy protein", "wheat bran", "rice husk",
        # 碳基材料
        "CNT", "carbon nanotube", "carbon black", "graphene", "graphene oxide",
        "reduced graphene oxide",
        # 纤维及增强材料
        "fiber", "natural fiber", "glass fiber", "bamboo fiber", "hemp fiber",
        "basalt fiber", "jute fiber", "kenaf fiber",
        # 增塑剂
        "glycerol", "triacetin", "citrate", "ATBC", "TEC", "tributyl citrate",
        "polyethylene glycol",
        # 共混及相容化
        "blending", "blended", "copolymerized", "reactive compatibilization",
        "immiscible", "miscible",
        # 其他添加剂
        "antioxidant", "nucleating agent", "chain extender", "crosslinker",
        "UV stabilizer", "thermal stabilizer", "fire retardant", "flame retardant"
    ],
    "Tensile Strength": ["tensile strength", "breaking strength", "tensile properties"],
    "Elongation at Break": ["elongation at break", "breaking elongation"],
    "Young's Modulus": ["young's modulus", "tensile modulus"],
    "Flexural Modulus": ["flexural modulus", "bending modulus", "flexural stiffness"],
    "Impact Strength": ["impact strength", "impact toughness"],
    "Stress-Strain": ["stress-strain", "mechanical behavior"],
    "Hardness": ["hardness", "shore hardness", "rockwell", "durometer"],
    "Glass Transition": ["glass transition", "Tg"],
    "Melting Point": ["melting point", "melting temperature", "Tm"]
}

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
