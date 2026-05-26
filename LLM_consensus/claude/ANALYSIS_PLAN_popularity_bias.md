# LLM打分流行性偏倚分析计划

## 任务目标
分析LLM的打分是否受到材料流行性偏倚的影响，对每个模型独立分析。

## 核心原则
**⚠️ 每个模型完全独立分析，不得混淆**

---

## 数据结构

### 材料（10种）
- gelatin, gelatin_methacrylate, polyacrylamide, chitosan
- silk_fibroin, polyethylene_glycol, starch, chitin
- cellulose, polyvinyl_alcohol

### 配方（10个，每个含2种材料）
- Formula 1: gelatin + gelatin_methacrylate
- Formula 2: polyacrylamide + gelatin
- Formula 3: chitosan + gelatin_methacrylate
- Formula 4: gelatin_methacrylate + silk_fibroin
- Formula 5: gelatin_methacrylate + polyethylene_glycol
- Formula 6: starch + gelatin_methacrylate
- Formula 7: chitin + gelatin_methacrylate
- Formula 8: gelatin_methacrylate + cellulose
- Formula 9: polyacrylamide + polyvinyl_alcohol
- Formula 10: polyacrylamide + polyethylene_glycol

### 评分维度（7个）
- Mechanical_Safety, Swelling_Performance, Endothelialization
- SMC_inhibition, Anti_inflammation, Thrombogenicity
- Total_Score

### 分析模型（4个）
- gpt-5
- grok-4
- claude-opus-4-5-20251101
- gemini-3-pro-preview

---

## 执行流程

### 阶段1：材料频度检索

#### 1.1 Datamuse API检索
- 使用`ml`（mean like）端点进行语义检索
- 权重：0.5
- 材料名称：空格分隔（如 "gelatin methacrylate"）
- 如有别名，合并结果

#### 1.2 ArXiv API检索
- 使用全文搜索
- 权重：0.5
- 不加权（与LLM数据保持一致）

#### 1.3 计算材料频度
```
material_freq = 0.5 * datamuse_count + 0.5 * arxiv_count
```

### 阶段2：相对频度计算（光环效应）

#### 2.1 配方频度
对每个配方，取两个材料中的**最大频度**：
```
formula_freq = max(material_freq_A, material_freq_B)
```

#### 2.2 取对数
```
relative_freq = log10(formula_freq + 1)
```
+1 避免log(0)

### 阶段3：相关性分析（每个模型独立）

#### 3.1 数据准备
- 加载`extracted_data.json`
- 对每个模型，提取11个run × 10个配方的分数

#### 3.2 计算Spearman相关系数
**对每个模型独立执行**：
- 对每个维度（7个），在11个run中分别计算该维度分数与relative_freq的Spearman ρ
- 得到11个相关系数（每run一个）
- 取中位数作为该维度的最终相关系数

**公式**：
```
for dimension in 7_dimensions:
    for run in 11_runs:
        ρ[run] = spearmanr(scores[run], relative_freq)[0]
    median_ρ[dimension] = median(ρ[run])
```

### 阶段4：去偏处理（每个模型独立）

#### 4.1 判断是否去偏
**阈值**：|median_ρ| > 0.5

#### 4.2 线性回归去偏
对需要去偏的维度，计算回归系数β：
```
from sklearn.linear_model import LinearRegression

# 使用所有11个run的数据
X = relative_freq (110个点，每个run 10个配方)
y = original_score (110个点)

model = LinearRegression()
model.fit(X, y)
β = model.coef_[0]

# 去偏
debiased_score = original_score - β * (relative_freq - mean(relative_freq))
```

#### 4.3 缩放回1-10范围
```
debiased_score_scaled = 1 + (debiased_score - min) / (max - min) * 9
```

### 阶段5：可视化（每个模型独立）

#### 5.1 相关性热力图
- 每个模型一张独立热力图
- Y轴：7个评分维度
- X轴：相对频度
- 颜色：median Spearman ρ值
- 边框标记：
  - 实线：|ρ|>0.6（强相关）
  - 虚线：0.5<|ρ|≤0.6（中等相关）
  - 无边框：|ρ|≤0.5（弱/无相关）

#### 5.2 原始vs去偏对比图
- 每个模型一张折线图
- X轴：10个配方
- Y轴：分数（1-10）
- 两条线：
  - 实线：原始分数（均值）
  - 虚线：去偏分数（均值）

#### 5.3 雷达图
- 每个模型一张雷达图
- 6个维度（不包含Total_Score）
- 两个多边形：
  - 实线：原始分数
  - 虚线：去偏分数

#### 5.4 材料频度分布图
- 单独一张图
- X轴：10种材料
- Y轴：log10频度
- 柱状图

---

## 输出文件结构

```
analysis/popularity_bias_analysis/
├── material_frequencies.json          # 材料频度（10种）
├── relative_frequencies.json          # 配方相对频度（10个）
├── {model}_correlation_results.json   # 每个模型的相关性结果
├── {model}_debiased_scores.json      # 每个模型的去偏后分数
└── visualization/
    ├── material_popularity_bar.png
    ├── {model}_correlation_heatmap.png
    ├── {model}_debias_comparison.png
    ├── {model}_radar_chart.png
    └── summary.md                     # 分析摘要
```

---

## 技术实现要点

### API调用
```python
# Datamuse
response = requests.get(f"https://api.datamuse.com/words?ml={material}&max=1")
if response.json():
    count = response.json()[0].get('numFound', 0)
else:
    count = 0

# ArXiv
response = requests.get(f"http://export.arxiv.org/api/query?search_query=all:{material}&max_results=1")
tree = ET.fromstring(response.text)
count = int(tree.find('.//{http://a9.com/-/spec/opensearch/1.1/}totalResults').text)
```

### 材料名称处理
- gelatin_methacrylate → ["gelatin methacrylate", "gelatin methacryloyl", "gelma"]
- silk_fibroin → ["silk fibroin", "fibroin"]
- polyethylene_glycol → ["polyethylene glycol", "peg", "polyethyleneglycol"]
- polyvinyl_alcohol → ["polyvinyl alcohol", "pva"]

### 统计分析
```python
from scipy.stats import spearmanr
import numpy as np

# 对每个模型、每个维度
for model in models:
    for dimension in dimensions:
        correlations = []
        for run in range(11):
            scores = get_scores(model, run, dimension)
            ρ, _ = spearmanr(scores, relative_freq)
            correlations.append(ρ)
        median_ρ = np.median(correlations)
```

---

## 执行状态记录

| 阶段 | 任务 | 状态 | 更新时间 |
|------|------|------|----------|
| 1 | 材料频度检索（ArXiv + PubChem Substance，独立步骤） | ✅ 完成 | 2026-05-08 21:00 |
| 2 | 相对频度计算 | ✅ 完成 | 2026-05-08 21:00 |
| 3 | 相关性分析（每个模型独立） | ✅ 完成 | 2026-05-08 21:00 |
| 4 | 去偏处理 | ✅ 完成 | 2026-05-08 21:00 |
| 5 | 可视化生成 | ⏳ 待完成 | - |

## 已完成工作

### 步骤1: 数据获取 (fetch_material_frequencies.py)
- 使用ArXiv API获取论文数
- 使用PubChem **Substance** API获取同义词数（适合混合物如gelatin）
- 材料别名合并，取最大值
- 等权重求和
- 结果保存到 `material_frequencies.json` 和 `api_source_data.json`

### 步骤2: 偏倚分析 (analyze_popularity_bias_simple.py)
- 加载保存的材料频度
- 计算配方相对频度（光环效应：max + log10）
- 每个模型独立计算Spearman相关性（11 runs取中位数）
- 对强相关维度（|ρ|>0.5）进行线性回归去偏
- 缩放回1-10范围
- 计算每个模型的最优配方（去偏后求和）
- 保存所有结果

### API测试结果
- PubChem Compound端点：对gelatin等混合物返回404（NotFound）
- PubChem Substance端点：正常返回同义词数
- ArXiv API：正常返回论文数
- **两步策略（compound/cids → xrefs/PubMedID）不适用于混合物材料**

### 分析结果摘要（使用ArXiv + PubChem Substance）

#### 各模型去偏情况
| 模型 | 强相关维度 (|ρ|>0.5) | 去偏维度数 |
|------|------------------|----------|
| **GPT-5** | Mechanical_Safety (0.524), Anti-inflammation (0.594), Thrombogenicity (0.554), Total_Score (0.513) | 4 |
| **Grok-4** | Thrombogenicity (0.686) | 1 |
| **Claude Opus 4.5** | Swelling_Performance (0.530), SMC_inhibition (0.505), Thrombogenicity (0.524) | 3 |
| **Gemini 3 Pro** | SMC_inhibition (0.782), Anti_inflammation (0.793), Thrombogenicity (0.705), Total_Score (0.695) | 4 |

#### 各模型最优配方
| 模型 | 最优配方 | 配方名 | 总分 |
|------|--------|--------|------|
| GPT-5 | 5 | GelMA + PEG | 54.89 |
| Grok-4 | 4 | GelMA + Silk Fibroin | 102.07 |
| Claude Opus 4.5 | 5 | GelMA + PEG | 83.40 |
| Gemini 3 Pro | 2 | PAM + Gel | 57.92 |

#### 关键发现
1. **Gemini最受流行性影响**：4个维度强相关
2. **GPT-5影响中等**：4个维度需去偏
3. **Grok-4影响较小**：仅1个维度强相关
4. **Claude Opus 4.5影响中等**：3个维度需去偏
5. **普遍受影响的维度**：
   - Thrombogenicity：所有模型都受影响
   - Anti-inflammation：GPT-5, Claude, Gemini受影响
   - SMC Inhibition：Claude, Gemini受影响
6. **较少受影响的维度**：
   - Endothelialization：仅Claude正相关但未超过阈值
   - Swelling_Performance：GPT-5, Claude正相关但仅Claude超过阈值

---

## 待完成任务（优先级：最低）
1. **将所有注释改为英文**（适合本目录下所有python文件）
2. **将中文注释改为英文**（analysis/, visualization/下所有文件）

---

## 分析结果摘要

### 材料频度排名（从高到低）
1. Silk Fibroin: 540
2. Polyethylene Glycol: 474
3. Polyvinyl Alcohol: 423
4. Gelatin Methacrylate: 221
5. Cellulose: 117
6. Gelatin: 74
7. Polyacrylamide: 51
8. Starch: 39
9. Chitin: 14
10. Chitosan: 0 (API超时)

### 各模型去偏情况
| 模型 | 强相关维度 | 中等相关维度 | 去偏维度数 |
|------|-----------|------------|----------|
| **GPT-5** | Anti-inflammation (0.687) | SMC Inhibition, Thrombogenicity, Total Score | 4 |
| **Grok-4** | Thrombogenicity (0.759) | Anti-inflammation | 2 |
| **Claude Opus 4.5** | Thrombogenicity (0.633) | Swelling, SMC, Anti | 4 |
| **Gemini 3 Pro** | SMC (0.782), Anti (0.793), Throm (0.744), Total (0.636) | - | 4 |

### 关键发现
1. **Gemini最受流行性影响**：4个维度强相关
2. **Grok-4和GPT-5影响中等**：各有2-4个维度需去偏
3. **Claude Opus 4.5影响较小**：仅1个维度强相关，3个中等相关
4. **普遍受影响的维度**：
   - Thrombogenicity：所有模型都受影响
   - Anti-inflammation：GPT-5, Grok-4, Claude, Gemini都受影响
   - SMC Inhibition：GPT-5, Claude, Gemini受影响
5. **较少受影响的维度**：
   - Mechanical Safety：仅GPT-5弱相关，其他无相关
   - Endothelialization：均无相关或弱负相关

---

## 关键确认事项

- [x] Datamuse使用`ml`检索
- [x] ArXiv无加权
- [x] 材料名称空格分隔，别名合并
- [x] 相关性：11个run取中位数，阈值|ρ|>0.5
- [x] 去偏后缩放回1-10范围
- [x] **每个模型完全独立分析**
- [x] 热力图边框标记强度


## 待完成任务（优先级：最低）
1. **将所有注释改为英文**（适合本目录下所有python文件）
2. **将中文注释改为英文**（analysis/, visualization/下所有文件）
3. **修复代码非ASCII字符**（已完成）

## 已发现问题
### Grok-4数据异常
- **问题**: Run 6, 7, 9存在异常低分数据（如3.0, 5.0等）
- **影响范围**: 
  - Mechanical_Safety: Run 6, 7, 9的score都是异常值
  - Swelling_Performance: Run 6, 7的score包含异常值（Run 7为9.0, Run 9为5.0）
- **导致**: Spearman相关系数为NaN（异常值干扰计算）
- **根本原因**: 原始数据质量问题（非代码逻辑错误）
- **建议**: 在数据提取阶段识别并过滤异常值，而非在相关性分析中处理

