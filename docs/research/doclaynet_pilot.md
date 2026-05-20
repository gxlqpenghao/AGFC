# DocLayNet Pilot

## 1. 目标

为 AGFC 建立一个小空间、可复现、可对比的公开 benchmark pilot。

当前只做第一轮公开 benchmark 对齐评测：

- IoU
- F1
- recall

暂不扩展到完整 mAP 体系。

## 2. 固定目录

- 缓存：
  `/Users/paul/Coding/AGFC/data/public/doclaynet_pilot/`
- 结果：
  `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/`
- 运行入口：
  `/Users/paul/Coding/AGFC/scripts/run_doclaynet_pilot.py`

## 3. 数据来源与获取方式

- 数据集：`docling-project/DocLayNet-v1.2`
- split：`test`
- 获取方式：Hugging Face datasets-server `rows` API

选择这个路径的原因：

1. 不需要下载 39GB 级全量 parquet
2. 单行样本已经包含：
   - `pdf`
   - `bboxes`
   - `category_id`
   - `metadata`
3. 可以按需分页抓取并缓存固定 64 页 pilot 子集

## 4. 当前 pilot 子集规则

当前首轮 pilot 采用以下确定性规则：

1. 从 `test` split 按 `offset` 升序扫描
2. 只保留包含 `Picture` 类别（`category_id == 7`）的页面
3. 计算 `Picture` bbox 缩放回原始 PDF 页面后的面积占比
4. 保留 `max_picture_area_ratio >= 0.002` 的页面
5. 取前 `64` 页

`manifest.json` 至少记录：

- `split`
- `row_id`
- `offset`
- `source_pdf_name`
- `page_no`
- `page_hash`
- `selected_reason`

## 5. 缓存结构

```text
data/public/doclaynet_pilot/
├── manifest.json
├── pdfs/
│   └── <row_id>.pdf
├── rows/
│   └── <row_id>.json
└── gt/
    └── <row_id>.json
```

说明：

- `pdfs/` 存单页 PDF
- `rows/` 存去掉二进制 `pdf` 字段后的源行 JSON
- `gt/` 存转换后的 AGFC 对齐 GT

## 6. 指标定义

### IoU

对 GT `Picture` bbox 与 AGFC 预测 bbox 做一对一 greedy matching，计算匹配对的平均 IoU。

### recall

`match_count / total_gt_count`

### F1

基于 IoU 阈值 `0.5` 的 detection-style matching 计算：

- precision = `match_count / total_prediction_count`
- recall = `match_count / total_gt_count`
- F1 = `2 * precision * recall / (precision + recall)`

## 7. 运行命令

```bash
python3 /Users/paul/Coding/AGFC/scripts/run_doclaynet_pilot.py \
  --split test \
  --limit 64 \
  --cache-dir /Users/paul/Coding/AGFC/data/public/doclaynet_pilot \
  --output-dir /Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot
```

限制 `Picture` 数的重跑示例：

```bash
python3 /Users/paul/Coding/AGFC/scripts/run_doclaynet_pilot.py \
  --split test \
  --limit 64 \
  --max-picture-count 1 \
  --cache-dir /Users/paul/Coding/AGFC/data/public/doclaynet_pilot/single_picture_64 \
  --output-dir /Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/single_picture_64
```

```bash
python3 /Users/paul/Coding/AGFC/scripts/run_doclaynet_pilot.py \
  --split test \
  --limit 64 \
  --max-picture-count 2 \
  --cache-dir /Users/paul/Coding/AGFC/data/public/doclaynet_pilot/picture_le2_64 \
  --output-dir /Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/picture_le2_64
```

DataProxy MinerU baseline 示例：

```bash
python3 /Users/paul/Coding/AGFC/scripts/run_doclaynet_mineru_baseline.py \
  --cache-dir /Users/paul/Coding/AGFC/data/public/doclaynet_pilot/single_picture_64 \
  --output-dir /Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_mineru_baseline/single_picture_64 \
  --dataproxy-root /Users/paul/Coding/DataProxy \
  --limit 64
```

## 8. 首轮 64 页结果

首轮结果文件：

- `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/manifest.json`
- `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/results.json`

首轮 aggregate：

- `page_count = 64`
- `total_gt_count = 151`
- `total_prediction_count = 58`
- `total_match_count = 6`
- `IoU = 0.7562`
- `recall = 0.0397`
- `F1 = 0.0574`

## 8.1 限制 Picture 数后的重跑结果

结果文件：

- `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/single_picture_64/results.json`
- `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/picture_le2_64/results.json`

对比结果：

| 子集 | 页数 | GT 数 | 预测数 | match 数 | IoU | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_64 | 64 | 151 | 58 | 6 | 0.7562 | 0.0397 | 0.0574 |
| single_picture_64 | 64 | 64 | 50 | 7 | 0.7518 | 0.1094 | 0.1228 |
| picture_le2_64 | 64 | 75 | 54 | 8 | 0.7677 | 0.1067 | 0.1240 |

## 8.2 这两次重跑说明什么

限制 `Picture` 数之后：

- `IoU` 基本稳定，仍在 `0.75-0.77` 附近
- `recall` 从 `0.0397` 提升到约 `0.107-0.109`
- `F1` 从 `0.0574` 提升到约 `0.123-0.124`

这说明：

1. 多 `Picture` 页面确实在显著压低 detection-style `recall/F1`
2. 但即使限制到 `picture_count == 1` 或 `<= 2`，AGFC 当前公开 benchmark 的 detection-style 结果仍然不高
3. 因此当前问题不是单一的“多图页面统计偏置”，而是：
   - 一部分是输出粒度与 DocLayNet detection 标注不一致
   - 另一部分是 AGFC 当前在这些公开页面上的命中率本身也偏低

## 8.3 DataProxy MinerU baseline 对比

AGFC 仓当前没有直接可运行的 MinerU 主入口，实际可运行链路在 DataProxy。

本轮在 AGFC 中新增了正式 bridge：

- `/Users/paul/Coding/AGFC/src/agfc/integrations/mineru/dataproxy_adapter.py`
- `/Users/paul/Coding/AGFC/src/agfc/doclaynet_mineru_baseline.py`
- `/Users/paul/Coding/AGFC/scripts/run_doclaynet_mineru_baseline.py`

它的做法是：

1. 复用 AGFC 固定 pilot cache
2. 调 DataProxy `runtime_pilot.py`
3. 如果 `runtime_pilot` 在 Milvus reindex 阶段失败，继续读取已经生成的 MinerU `extracted/layout.json`
4. 从 `image` block 提取 bbox，按同一套 DocLayNet 指标评分

`single_picture_64` 同 manifest 的 MinerU 结果文件：

- `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_mineru_baseline/single_picture_64/results.json`
- `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_mineru_baseline/single_picture_64/taxonomy.json`

对比结果：

| 子集 | 方法 | GT 数 | 预测数 | match 数 | IoU | recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `single_picture_64` | AGFC | 64 | 50 | 7 | 0.7518 | 0.1094 | 0.1228 |
| `single_picture_64` | MinerU | 64 | 49 | 38 | 0.9341 | 0.5938 | 0.6726 |

当前结论：

- 在 `single_picture_64` 这个更可比的公开 benchmark pilot 上，**AGFC 明显弱于 MinerU**
- 因此当前还不能用 DocLayNet 这条线支持“AGFC 并不差于 MinerU”
- DocLayNet 当前更适合承担：
  - 相对对比
  - 误差分析
  - gap 定位

## 9. 当前结论

这轮结果说明两件事：

1. **公开 benchmark 最小链路已经打通**
   - DocLayNet 获取/缓存
   - PDF/GT 适配
   - AGFC 预测对齐
   - IoU/F1/recall 输出
2. **当前这版 fixed 64-page 子集还不能支持“AGFC 在公开 benchmark 上不差”这个结论**

首轮结果偏低的直接原因很明确：

- 64 页里有 `27` 页的 GT `Picture` 数量大于 `1`
- 平均每页 GT 数量是 `2.3594`
- 有些页面 GT `Picture` 数达到 `7`、`9`、`10`、`11`
- AGFC 当前在这些页面上通常只输出 `0` 或 `1` 个逻辑 figure

因此：

- **匹配上的框 IoU 并不差**：目前三组子集都在 `0.75+`
- **但 detection-style recall / F1 仍低**
- 这既有 **pilot 子集与 AGFC“逻辑完整图”输出粒度不一致** 的因素，也有当前公开页命中率不足的问题

## 9.1 当前 taxonomy 观察

AGFC taxonomy：

- `baseline_64`
  - `zero_prediction = 6`
  - `one_prediction_no_match = 52`
  - `multi_gt_collapsed_to_one = 2`
  - `matched = 4`
- `single_picture_64`
  - `zero_prediction = 14`
  - `one_prediction_no_match = 43`
  - `matched = 7`

MinerU taxonomy：

- `single_picture_64`
  - `matched = 32`
  - `zero_prediction = 23`
  - `one_prediction_no_match = 3`
  - `hit_but_granularity_mismatch = 6`

这说明在 `single_picture_64` 上：

- AGFC 的主要失败类型是 `one_prediction_no_match`
- MinerU 的主要优势是显著提高了有效命中数
- AGFC 当前更值得优先拆解的不是 mAP，而是：
  - 为什么大量页面会落到 `one_prediction_no_match`
  - 这些页面里有多少是 seed recall 不足
  - 有多少是 bbox 偏移
  - 有多少是 suppression 或 closure 把边界切坏

## 10. 下一步最小可执行路径

如果下一轮目标是尽快验证“公开 benchmark 上并不差”，最小路径建议是：

1. 保持当前正式链路不变
2. 在当前两组更可比子集基础上继续做误差分解，而不是再回到全量混合子集
3. 优先抽检：
   - `single_picture_64` 中 `0 prediction` 页
   - `single_picture_64` 中 `1 prediction but no match` 页
   - `single_picture_64` 中 MinerU 命中而 AGFC 未命中的页
4. 区分两类失败：
   - 完全没出图
   - 出了逻辑图但与 DocLayNet `Picture` 粒度不一致
5. 只在误差类型清楚后，再决定是：
   - 调整 pilot 选择规则
   - 还是针对公开 benchmark 专门补 recall

这样可以在不改 AGFC 主算法的前提下，更快判断“公开 benchmark 上不差”有没有现实空间，以及差距主要来自哪里。
