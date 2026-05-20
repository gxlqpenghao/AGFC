# AGFC 正式架构方案

## 1. 结论先行

本方案完全接受以下判断：

1. `AGFC` 的核心创新点成立：
   - 将 PDF 图片抽取从 `bbox 回归` 重定义为 `页面原子图上的语义闭包`
   - 这是方法论层面的创新，而不是单纯的工程调参

2. 当前 `AGFC v1` 仍然是**验证原型**，还不是顶刊级方法：
   - 闭包尚未形式化
   - 评测指标不足
   - 阈值硬编码明显
   - 闭包传播仍主要依赖 `same_row`

3. 特别是 “大量 magic number / 阈值硬编码，通用性存疑” 这一点，必须严肃对待。
   - 这不是普通工程小问题
   - 它已经触及方法能否被证明具有通用性的核心

因此，后续目标不是继续在 `v1` 上增补更多特殊候选器，而是升级为：

> **AGFC: Self-Calibrated Bipolar Hierarchical Closure**

---

## 2. 当前 v1 的主要问题

当前实现的优点是：

- 已经证明 `AGFC` 路线在真实 PDF 上可工作
- 已经有稳定的原子提取、候选生成、图构建、闭包和批次归档能力

当前实现的主要问题是：

### 2.1 候选生成仍是“按类型补规则”

当前 `panel` 候选层已经演化出多类分支：

- `border`
- `layout_column`
- `image_seed`
- `image_cluster`
- `captioned_image_seed`

这些分支虽然在工程上有效，但会带来两个问题：

1. 新版式一来，就要继续新增第 6、第 7 类种子
2. 论文叙事会退化为“我们为不同图片类型写了不同规则”

### 2.2 阈值仍然是绝对值

当前系统中存在大量绝对阈值，例如：

- 面积阈值
- 宽高阈值
- gap 阈值
- 对齐容差
- 附件检测距离

这会导致：

- 同样方法对不同版式 PDF 失稳
- 方法难以论证“文档自适应”
- reviewer 很容易质疑通用性

### 2.3 闭包还不是严格闭包算子

当前闭包更接近：

- 在 `same_row` 图上取连通分量
- 再并入 repeated attachments

这在工程原型中是合理的，但在理论上还不够：

- 不是完整的 `E+ / E-` 传播体系
- 没有最小固定点定义
- 缺少可证明性

### 2.4 图关系的表达能力强于图求解能力

当前关系层已经开始积累：

- `contains`
- `same_row`
- `same_column`
- `near_body_text`
- `attachment_candidate`

但闭包层还没有把这些关系全部统一进一个严格的求解框架。

---

## 3. AGFC 的正式目标

`AGFC` 的目标不是继续补更多候选规则，而是完成以下三个升级：

1. 从“硬编码阈值系统”升级到“文档自校准系统”
2. 从“按类型产生 panel”升级到“统一证据驱动的候选发现系统”
3. 从“same_row 联通分量”升级到“可形式化证明的双极闭包系统”

---

## 4. 总体架构

`AGFC` 分为 4 个主层。

### Layer A: Self-Calibrated Atomization

目标：

- 替代当前绝对阈值原子提取
- 让原子分类与尺度判定相对于文档自身统计量进行

核心模块：

1. `Text DNA Fingerprinting`
2. `Layout Fingerprint Calibration`

输出：

- `PageAtom`
- `TypographyDNA`
- `LayoutFingerprint`

### Layer B: Unified Seed Evidence

目标：

- 不再保留并列的“图片类型专属候选器”
- 所有候选来源统一收束到同一个接口

统一数据结构：

```python
@dataclass
class SeedCandidate:
    id: str
    bbox: tuple[float, float, float, float]
    source_atoms: list[str]
    evidence_tags: list[str]
    score: float
    provenance: str
```

说明：

- `border panel`、`layout column`、`image seed`、`image cluster`、`captioned image`
- 以及未来的 `seed-free community`

都不再是主逻辑分叉，而只是：

- 不同的 `evidence provider`
- 统一输出 `SeedCandidate`

### Layer C: Bipolar Closure

目标：

- 建立可形式化定义的语义闭包
- 用统一图传播替代当前 `same_row` 主导逻辑

核心定义：

给定页面原子图：

`G = (V, E+, E-)`

其中：

- `V`: 页面原子或高层候选节点
- `E+`: 吸引边
- `E-`: 排斥边

推荐边集合：

- `E+`
  - `contains`
  - `same_row`
  - `same_column`
  - `attachment`
  - `layout_repetition`
  - `community_cohesion`

- `E-`
  - `body_text_barrier`
  - `whitespace_barrier`
  - `font_discontinuity`
  - `semantic_discontinuity`
  - `over_gap_barrier`

闭包定义建议：

> 从候选种子集合 `S0` 出发，沿 `E+` 做单调扩张；若某条传播路径被 `E-` 截断，则停止传播。最终得到的最小不再变化集合 `Cl(S0)` 为该候选图的语义闭包。

这一定义应作为论文中的正式 Definition。

### Layer D: Hierarchical Closure + Export

目标：

- 让系统既能输出“整张逻辑完整图”
- 也能保留子层次结构

三级结构建议：

1. `L0 Element Closure`
   - 原子 -> 微结构块
2. `L1 Panel Closure`
   - 微结构块 -> 面板组
3. `L2 Figure Closure`
   - 面板组 -> 逻辑完整图

输出能力：

- 整图导出
- 中间层 debug
- 子面板结构表示

---

## 5. 对你提出的 4 个关键 Gap 的正式回应

## 5.1 P0: “闭包”缺形式化

完全认可。

升级要求：

1. 必须给出 `Bipolar Closure` 的正式定义
2. 必须把当前 `same_row` 连通分量替换为更一般的 `E+ / E-` 固定点传播
3. 必须补一组理论命题：
   - 单调性
   - 终止性
   - 最小闭包存在性

建议写法：

- `Definition 1: Atom Graph`
- `Definition 2: Bipolar Closure`
- `Proposition 1: Monotonic Expansion`
- `Theorem 1: Existence of Least Fixed Point`

## 5.2 P1: 需要自定义评估指标

完全认可，而且这是 AGFC 能否讲出“独特优势”的关键。

建议主指标：

1. `Fragmentation Rate`
   - 一张 GT 图被系统拆成多少个结果图

2. `Contamination Rate`
   - 结果图中混入的正文面积占比

建议辅助指标：

3. `Overmerge Rate`
   - 不相关图被错误合并的比例

4. `Structural Completeness`
   - 图题/图注/共享说明条/共享标签是否被保留

为什么重要：

- `IoU` 不能反映 AGFC 的真正优势
- AGFC 解决的不是单纯框精度，而是：
  - 不拆碎
  - 少污染
  - 结构完整

## 5.3 P1: 大量 magic number

完全认可，而且我建议把它视为 `P0.5`。

因为它不是“代码看起来丑”，而是：

- 会直接削弱通用性主张
- 会让 reviewer 怀疑这只是“对当前文档调出来的系统”

这里需要区分两类参数：

### 必须消灭的参数

- 与页面物理尺度直接绑定的绝对值
- 例如：
  - `80 pt`
  - `40 pt`
  - `180 pt`
  - `36 pt`

这些参数必须迁移为：

- `LayoutFingerprint` 驱动的相对尺度函数

### 可以保留的参数

- 归一化比例超参数
- 例如：
  - `0.3`
  - `1.5`
  - `3.0`

这些在论文中可以作为：

- 可解释的比例超参数
- 而不是“magic number”

## 5.4 P2: 只有 same_row 传播

完全认可。

当前闭包主链路仍然过于依赖：

- `same_row`

即使 `same_column` 已经存在于关系层，它还没有真正成为闭包求解主逻辑的一部分。

升级要求：

1. `same_column` 必须与 `same_row` 同级进入 `E+`
2. `attachment` 不能只是附属修补，而应成为一级吸引边
3. 纵向 panel group 必须被纳入 `L1 panel closure`

---

## 6. 与当前代码的映射关系

当前实现中最需要升级的部分如下：

### 6.1 `pdf_agfc_atoms.py`

当前问题：

- 原子分类里仍有绝对阈值
- 还没有 `Text DNA`

升级方向：

- 提取 typography feature vector
- 输出 `TypographyDNA`
- 用 `LayoutFingerprint` 代替硬编码阈值

### 6.2 `pdf_agfc_panels.py`

当前问题：

- `panel_kind` 分叉越来越多
- 候选发现还是 archetype-driven

升级方向：

- 改造成 `seed evidence providers`
- 所有 provider 统一输出 `SeedCandidate`

### 6.3 `pdf_agfc_graph.py`

当前问题：

- 已有较好的关系表达基础
- 但 `E-` 仍然非常弱

升级方向：

- 加入 `body_text_barrier`
- 加入 `whitespace_barrier`
- 加入 `semantic_discontinuity`

### 6.4 `pdf_agfc_closure.py`

当前问题：

- 目前更像“连通分量 + 附件并入”
- 不是正式闭包求解器

升级方向：

- 升级为 `Bipolar Fixed-Point Closure`
- 再升级为 `Hierarchical Closure`

---

## 7. 正式升级顺序

我建议严格按下面顺序升级，而不是并行乱做。

### 阶段 1：先消灭通用性瓶颈

1. `Text DNA Fingerprinting`
2. `Layout Fingerprint Calibration`

原因：

- 这是当前最紧迫的架构短板
- 实现难度低
- 对论文和工程都最有价值

### 阶段 2：统一候选接口

将所有 `panel_kind` 收束为统一的：

- `SeedCandidate`

原因：

- 这是从“规则枚举器”走向“证据框架”的关键

### 阶段 3：闭包形式化

引入：

- `Bipolar Closure`
- `E+ / E-`
- 固定点传播

原因：

- 这是方法论层面的主贡献

### 阶段 4：补 recall 的上限

引入：

- `Seed-Free Figure Discovery`

原因：

- 这是 recall 的理论解，不是继续补第 6、第 7 类 seed

### 阶段 5：层次输出与边界验证

引入：

- `Whitespace Topology`
- `Hierarchical Closure`

原因：

- 让系统从“可工作”升级为“可发表、可解释、可扩展”

---

## 8. 论文层面的正式 contribution 建议

不要强调 6 个创新点。

正式论文建议只保留 4 个 contribution：

1. 我们提出 `AGFC`，将 PDF 图片抽取重新定义为页面原子图上的语义闭包问题。
2. 我们提出 `Text DNA + Layout Fingerprint`，实现无训练、自校准的正文/图元素分离与尺度统一。
3. 我们提出 `Bipolar Closure`，用吸引边与排斥边定义图闭包固定点。
4. 我们提出统一的 `Seed Evidence + Seed-Free Discovery` 框架，并支持层次化图恢复导出。

---

## 9. 数据集与评测策略

### 9.1 JournalMix-v1 作为主 benchmark

AGFC 当前的主评测线应收敛到 `JournalMix-v1`，而不是再并列多条公开 benchmark 支线。

原因：

- `JournalMix-v1` 直接面向 AGFC 关注的 figure extraction 场景
- 页面选择、GT、review 和审计工件已经冻结
- 可以自然承接 AGFC 与 MinerU 的对比

### 9.2 自建 AGFC-Bench

为了评测 AGFC 的独特优势，必须构建自有 benchmark：

**规模建议**：100–200 页即可支撑论文（不需要 PubLayNet 级规模）。

**数据来源建议**：

| 子集 | 来源 | 页数 | 用途 |
|---|---|---|---|
| `AGFC-Sci` | arXiv 开放论文（英文+中文） | ≥80 页 | 学术论文场景验证 |
| `AGFC-Eng` | 工程报告 / 技术标书 | ≥60 页 | 中文工程场景验证 |
| `AGFC-Poster` | 海报式组图、宽幅图表文档 | ≥40 页 | 极端复杂场景验证 |

**标注规范**（每页需标注）：

1. **逻辑完整图 GT bbox**：组图的外边界（不是子面板边界）
2. **子面板组成列表**：哪些 PDF 元素属于同一张逻辑图
3. **body_text vs figure_text 标签**：每个 text block 的角色
4. **图题/图注/说明条标签**：是否属于图的一部分

### 9.3 双轨评测策略

第一轨（主 benchmark + baseline 对比）：
  JournalMix-v1 冻结页集
  → AGFC fresh benchmark
  → Local MinerU raw-page audit
  → 汇报 IoU / Precision / Recall / F1
  → 证明 AGFC 在目标场景中的基线优势

第二轨（核心贡献 + 差异化）：
  AGFC-Bench（自建 100-200 页）
  → 评测 FCOS 四指标：
    - Fragmentation Rate
    - Contamination Rate
    - Overmerge Rate
    - Structural Completeness
  → 按图片复杂度分组报告（Simple / Captioned / Compound / Hybrid）
  → 证明"在难例上有质的飞跃"

第一轨解决"可信度"，第二轨解决"独特性"。

### 9.4 关于输入格式的通用性说明

AGFC 选择在 PDF 层操作，这不是局限而是优势：

- **PDF 是所有文档格式的通用渲染终点**
- DOCX / PPTX / HTML / LaTeX 均可通过标准工具（LibreOffice、Chrome print-to-PDF）无损转为 PDF
- 转换后 AGFC 以完全相同的方式处理

因此 AGFC 不需要为不同文档格式编写独立解析器。论文中建议正式表述为：

> "AGFC operates on the PDF layer, which serves as the universal rendering target for virtually all document formats. This single-pathway design eliminates the need for format-specific parsers."

在实验中，建议至少包含一组 `DOCX → PDF → AGFC` 的跨格式验证，以实证支撑该主张。

---

## 10. 方法论通用性与论文定位

### 10.1 问题："PDF 抽图"是不是太小了？

如果论文的定位是"一个从 PDF 里提取图片的工具"，确实太窄，不足以支撑顶刊。

但 AGFC 的真正贡献不是"抽图"这个应用，而是"图闭包"这个方法论：

```
表面上做的事情：从 PDF 里把图提出来
本质上提出的方法：从异构页面元素中恢复逻辑复合对象的通用图闭包框架
```

论文卖的不是应用，而是方法论。PDF figure extraction 只是第一个也是最有说服力的验证场景。

### 10.2 框架的可迁移场景

bipolar graph closure 框架可以直接迁移到以下问题，因为它们的本质结构完全同构：

| 问题 | 异构元素 | 逻辑复合对象 | 与 AGFC 的关系 |
|---|---|---|---|
| **PDF 抽图** | text + image + vector + band | 逻辑完整图 | 当前主验证场景 |
| **PDF 表格恢复** | cell + border + header + merged cell | 逻辑完整表 | 表格也是多元素拼接的复合对象 |
| **PPT 版面理解** | shape + textbox + connector + group | 逻辑幻灯片区块 | PPT 元素同样异构且经常丢失分组信息 |
| **网页视觉分块** | DOM element + CSS box + image | 视觉信息块 (VIPS) | 经典问题，当前用启发式规则 |
| **扫描文档 OCR 分区** | text line + separator + logo | 逻辑文档区域 | 同样是 bottom-up grouping |
| **工程图纸理解** | line + dimension + annotation + symbol | 逻辑标注组 | CAD 领域经典痛点 |

这些问题的本质都是一样的：**很多小零件，需要搞清楚哪些属于同一个逻辑整体。**

### 10.3 论文标题与定位建议

标题不要绑定"PDF"或"抽图"：

- ❌ 太窄：`A Method for Extracting Figures from PDF Documents`
- ❌ 稍窄：`AGFC: Graph Closure for PDF Figure Recovery`
- ✅ 合适：`Beyond Bounding Boxes: Bipolar Graph Closure for Logical Object Recovery from Heterogeneous Document Elements`

在这个定位下：

- **方法论是通用的**（bipolar graph closure on heterogeneous elements）
- **主实验在 PDF figure 上做**（最成熟、最有对比空间的场景）
- **Discussion 中展示迁移性**（表格恢复的 pilot 实验）

### 10.4 论文结构中如何支撑通用性

```
Section 1 Introduction:
  提出"异构文档元素的逻辑对象恢复"这个一般性问题
  → 不以"抽图"开头，以"文档理解的核心挑战"开头

Section 3 Problem Formulation:
  在一般化的异构元素集合上定义 Atom Graph 和 Bipolar Closure
  → 理论框架不绑定 PDF，不绑定 figure

Section 4 Method (AGFC):
  作为框架的具体实例化
  → "As a concrete instantiation, we apply the framework to 
     PDF figure extraction, the most challenging case of 
     logical object recovery in document understanding."

Section 6 Experiments:
  6.1-6.4 主实验：PDF figure extraction（完整对比 + 消融）
  6.5 迁移实验：在 PDF 表格上跑 pilot → 初步证明框架通用性

Section 7 Discussion:
  → 列出全部可迁移场景
  → 如果有表格 pilot 数据，展示初步结果
  → 明确提出"AGFC 不限于 figure extraction"
```

### 10.5 这个做法在顶刊中是标准模式

很多经典论文的 contribution 是通用方法论，但实验主要在一个领域做：

| 论文 | 通用方法 | 主验证场景 |
|---|---|---|
| **Transformer** | Self-attention mechanism | 机器翻译 |
| **RAG** | Retrieval-augmented generation | 开放域问答 |
| **PageRank** | 图上的随机游走收敛 | 网页排序 |
| **MapReduce** | 分布式函数式计算 | 文本索引 |

AGFC 的模式完全一致：**contribution 是 bipolar graph closure for logical object recovery，PDF figure extraction 是最强验证场景。**

### 10.6 建议补充的迁移验证

为了在论文中实际支撑通用性主张，建议在主实验之外补一组轻量 pilot：

- 选 10-20 页含复杂表格的 PDF
- 将 AGFC 的原子化 + 闭包直接应用于表格边界恢复
- 报告 table IoU 对比 baseline
- 即使结果不如专用表格工具，只要闭包框架"能跑起来"就足以证明通用性

这个 pilot 不需要做到主实验的深度，但它的存在会让 reviewer 无法质疑"这个方法是不是只对 figure 有效"。

---

## 11. 最终结论

我对你的 Gap 评估结论是：

- 整体上完全认可
- 尤其认同：
  - `闭包缺形式化`
  - `大量 magic number`
  - `same_row 传播过窄`

我对 `AGFC` 的正式建议是：

> 不要继续在 `v1` 上增加新的特殊候选器，而应将系统升级为  
> **Self-Calibrated Atomization + Unified Seed Evidence + Bipolar Fixed-Point Closure + Hierarchical Export**

这条升级路线同时兼顾：

- 方法论创新性
- 工程可实现性
- 论文可证明性
- 通用性论证

如果后续继续沿这条路线推进，`AGFC` 就能从“一个有效的原型系统”升级为“一个理论上可阐述、实验上可验证、工程上可落地的通用方法框架”。
