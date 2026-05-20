# AGFC

AGFC（Atom-Graph Figure Closure）是一个面向复杂文档视觉对象恢复的正式研发项目。当前主验证场景是 PDF figure recovery，但方法论目标是从异构页面元素中恢复逻辑完整对象，而不是停留在单一脚本级“PDF 抽图”。

## 当前状态

- 单一正式主线：`AGFC`
- 主运行入口：`scripts/run_agfc.py`
- 批量语料入口：`scripts/run_corpus.py`
- 评测入口：`scripts/run_evaluation.py`
- 未来 MinerU 对接预留：`src/agfc/integrations/mineru/`

## 目录结构

```text
AGFC/
├── src/agfc/             # 正式源码包
├── scripts/              # 对外运行入口
├── tests/                # 自动化测试
├── docs/                 # 架构 / 研究 / 开发文档
├── data/                 # 样本语料与 benchmark 定义
└── artifacts/            # 运行产物、评测结果、历史归档
```

## 快速开始

运行单文档：

```bash
python3 scripts/run_agfc.py /absolute/path/to/file.pdf
```

运行样本语料：

```bash
python3 scripts/run_corpus.py
```

运行评测汇总：

```bash
python3 scripts/run_evaluation.py --run-a <path> --run-b <path> --output <path>
```

## 说明

- 所有新的开发应围绕 `src/agfc/` 中的正式模块展开。
- `docs/architecture/` 和 `docs/research/` 保存架构与论文相关文档。
