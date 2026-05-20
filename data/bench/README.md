# AGFC-Bench

本目录存放 `AGFC-v2` 的评测样本与标注说明。

当前目标不是一次性构建完整 benchmark，而是先冻结标注接口，避免后续：

- 评测脚本猜测字段含义
- 不同人按不同格式标注
- 指标实现与标注结构反复返工

## 标注单元

建议以“单页 JSON 记录”为基础单元，每页一个标注对象。

## 最小字段

- `page_idx`
- `page_label`
- `figures`

每个 `figure` 最小字段应包含：

- `figure_id`
- `bbox`
- `logical_group_id`
- `panel_bboxes`
- `caption_bbox`
- `body_exclusion_bboxes`

## 设计原则

1. `bbox` 表示最终希望导出的完整图区域
2. `panel_bboxes` 表示图内部子面板结构
3. `caption_bbox` 可为空
4. `body_exclusion_bboxes` 用于显式标注不应被混入图的正文区域

## 使用建议

- 初期先对当前中建科技进步奖 PDF 做高价值页标注
- 优先覆盖：
  - 海报式组图
  - 无边框组合图
  - 图下说明条
  - 图与正文紧贴页面

schema 见：

- [gt_schema.json](./gt_schema.json)

## Private Journal-Paper Benchmark

Private journal-paper figure evaluation (AGFC-JournalMix-v1) lives in
`data/private/journalmix_v1/`.  That set uses the same GT schema but
targets AGFC-specific stress cases across four figure-type buckets.
See its own README for details.
