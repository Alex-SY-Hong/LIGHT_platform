# Claude 项目配置文件

本目录包含 Claude Code (AI 编程助手) 的项目上下文和配置文件。

## 文件说明

### CLAUDE_CONTEXT.json
- **用途**: 存储项目的完整上下文信息
- **内容**:
  - 项目概述和目的
  - 项目结构（数据文件、Python 模块、报告文件）
  - 工作流程和标准管道
  - 最近完成的工作
  - 技术细节（ICC 计算方法、统计指标）
  - 快速开始指南

### 关键指令.md
- **用途**: 项目特定的关键指令和工作流程
- **内容**:
  - Python 虚拟环境配置
  - 可视化包使用说明
  - 项目上下文恢复方法
  - 忽略规则

### README.md (本文件)
- **用途**: Claude 配置文件目录说明

## 使用方法

### 首次使用或恢复项目上下文
1. 阅读 `claude/CLAUDE_CONTEXT.json` 了解项目全貌
2. 阅读 `claude/关键指令.md` 了解关键配置
3. 根据需要更新 `CLAUDE_CONTEXT.json` 中的 "recent_work" 部分

### 更新上下文
完成重要工作后，更新 `CLAUDE_CONTEXT.json` 中的相关部分：
- `recent_work`: 记录本次会话完成的工作
- `status`: 更新项目状态
- `next_steps`: 添加后续改进建议

## 目录结构

```
LLM_consensus/
├── claude/                          # Claude 配置文件目录
│   ├── CLAUDE_CONTEXT.json          # 项目上下文
│   ├── 关键指令.md                   # 关键指令
│   └── README.md                     # 本文件
├── visualization/                   # 可视化模块
├── extracted_csv/                   # 提取的 CSV 数据
├── visualizations/                  # 生成的图表
└── *.py                             # Python 脚本
```

## 注意事项

- 本目录下的文件专用于 Claude Code 理解项目上下文
- 定期更新 `CLAUDE_CONTEXT.json` 以保持上下文最新
- 新增关键工作流程时更新 `关键指令.md`
