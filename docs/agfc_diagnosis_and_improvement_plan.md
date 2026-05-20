# AGFC Edgefix4 诊断与下一阶段开发指导

## 0. 现状快照

| 指标 | AGFC v1 原始 | AGFC edgefix4 | MinerU baseline |
|---|---|---|---|
| match_count / 64 | 7 | **33** | 38 |
| precision | 0.14 | **0.7021** | 0.7755 |
| recall | 0.1094 | **0.5156** | 0.5938 |
| f1 | 0.1228 | **0.5946** | 0.6726 |
| mean IoU (matched) | 0.7518 | **0.8364** | 0.9341 |
| prediction_count | 50 | 47 | 49 |

edgefix4 已经证明方向正确：它把 `match_count` 从 `7 -> 33`，把 `recall` 从 `0.1094 -> 0.5156`。  
当前与 MinerU 的差距缩小到 `5` 个 match，但剩余问题已经不再是单一模式，而是分裂成少数几类高价值缺口。

本文件的目标不是复述历史，而是给出一份可直接指导下一阶段开发的、经过核实的文档。

---

## 0.1 战略定位修订

AGFC 的正确定位不是“MinerU 之后的补丁式后处理”，也不是“只有和神经方法融合才有意义”的附属系统。

AGFC 应定位为：

> 一个独立的、非神经的、基于 PDF 结构图闭包的 figure 抽取方案。

它的关键特征是：

- 不依赖训练数据
- 不依赖 GPU
- 不依赖外部检测模型
- 直接从 PDF 内部结构化原子出发，构建空间关系图并做双极闭包

MinerU 在这里的角色是：

- **竞争对比基线**
- **异质方法参照系**
- **可选的后续融合对象**

而不是：

- AGFC 的前置依赖
- AGFC 存在价值的前提
- AGFC 必须服务的主系统

因此，下一阶段叙事应统一成：

1. **方法定位**
   - AGFC 是独立的 figure extraction 方法，不是谁的结构化补丁
2. **研究突破口**
   - AGFC 的最强差异化方向在 `vector figure recovery`
   - 但它首先仍是一个完整的、可单独运行的 figure 抽取系统
3. **实验关系**
   - 与 MinerU 的关系应先写成 **竞争对比**
   - 融合实验是后续可选增值路线，而不是核心主线

这不意味着放弃公开 benchmark。

公开 benchmark 在下一阶段的作用是：

- 证明 AGFC 作为独立方法具有可比性能
- 量化它与神经方法的优劣边界
- 为后续 `vector figure recovery` 专项评测提供对照背景

更准确的目标表述应是：

> 在当前 PDF figure 抽取高度依赖神经网络的背景下，提出基于 PDF 原子图语义闭包的非神经方法 AGFC。该方法直接解析 PDF 结构化数据，通过空间关系图构建和双极闭包算法，实现对包括矢量绘制图表在内的复合 figure 的自动恢复。在公开 benchmark 上，AGFC 以零训练数据达到与神经方法可竞争的性能，并在特定的 vector-dominant 场景上展现独特优势。

---

## 0.2 机制准入标准

后续开发必须优先补强 **核心机制**，严肃拒绝“看起来抽象、实则仍是对样本特征做规则”的 **伪通用机制**。

### 允许进入主线的改动

只有满足下列至少一类的改动，才应进入 AGFC 主线：

1. **原子完整性补强**
   - 目标是让页面里的真实视觉原子更完整地进入图系统
   - 例：
     - xref 级图片恢复
     - 细长但显著的 drawing primitive 保留
     - bbox 几何归一化到 page rect

2. **尺度统一与校准**
   - 目标是把散落的绝对阈值收口到统一入口，并逐步向相对尺度或 fingerprint 校准过渡
   - 例：
     - `thresholds.py`
     - 相对页面尺寸门槛
     - 未来 `LayoutFingerprint` 驱动阈值

3. **图关系统一化**
   - 目标是让不同 seed / panel / atom 在同一套图推理机制下工作，而不是为个别类型开旁路
   - 例：
     - `seed_free` 候选通过 `seed_to_atom` 正式接入闭包图
     - 统一 coverage / ranking 约束
     - attachment 规则共享实现

4. **统一目标优化**
   - 目标是让评分、抑制、闭包收敛遵循统一原则，而不是按图种分裂
   - 例：
     - ranking penalty 统一化
     - panel / seed / figure 之间的目标一致性

### 不允许直接进入主线的改动

以下模式默认视为 **伪通用机制**，除非能给出强理论理由，否则不应进入主线：

1. **按 figure 外观类型命名的新 panel/seed 规则**
   - 例如：
     - `annotated_visual_seed`
     - “chart_title_seed”
     - “legend_seed”
   - 原因：
     - 这类命名通常意味着我们已经在按样本类型建模板

2. **把附近标题/标签文本并入 figure 的语义规则**
   - 原因：
     - 这会把 AGFC 从结构闭包方法推向图表模板规则
     - 不同文档域的“标题/标签”语义差异极大

3. **只对少数剩余页明显有利的评分补丁**
   - 例如：
     - 只为某一类 seed 再抬 cap
     - 只对某一页型再加惩罚或豁免
   - 原因：
     - 这类规则往往只是把样本现象包装成抽象词汇

4. **缺乏统一解释的新分支**
   - 如果一个机制不能解释为：
     - 原子完整性
     - 尺度统一
     - 图关系统一
     - 统一目标优化
   - 那么它就不应轻易进入主线

### 代码评审的硬门

以后每一条算法改动，在合并前都必须回答 4 个问题：

1. 它补强的是哪一个核心层？
   - atom
   - calibration
   - graph
   - closure / ranking

2. 它是否能在不提及具体样本页的情况下解释？

3. 它是否引入了新的 figure 类型语义？
   - 如果答案是“是”，默认拒绝

4. 它是否能通过“去掉页号和样本名后仍然成立”的表述来描述？
   - 如果不能，说明它大概率是伪通用机制

### 当前主线允许保留的补强

截至目前，以下改动仍属于核心机制补强，可以保留在主线：

- xref 图片恢复
- bbox 裁到 page rect
- 细长 drawing primitive 保留
- 阈值统一入口
- `seed_free` 作为通用小视觉候选发现
- `seed_free` 通过 `seed_to_atom` 接入闭包图

### 当前需要持续警惕的机制

以下机制虽然还未全部删除，但它们都应被视为“需要持续审查”的对象：

- `page_as_image`
- `_is_overexpanded_single_raster_community`
- `_coverage_penalty` 只限 `visual_community`
- `clipped_to_page` xref raster 的弱证据晋升限制
- `_is_hero_visual_atom` 的多语义耦合

这些机制可以暂时存在，但必须被理解为 **过渡性工程启发式**，而不是 AGFC 的理论内核。

---

## 1. 已核实的事实修正

### 1.1 `page_as_image` 不是当前 3 个全页失败页的直接根因，但它是潜伏地雷

代码位置：

- `src/agfc/pipeline.py`
- `_is_boilerplate_seed()`
- `page_as_image = isolated_raster and len(seed.source_atoms) == 1 and area_ratio >= 0.85`

核实结果：

- `test_000150 / test_000170 / test_000195` 在 edgefix4 调试包里 **没有 `raster_image` atom**
- 它们的 `atoms.json` 只有离屏 `vector_cluster`
- 这 3 页 `seeds.json` 为空，说明 seed 从未生成，`page_as_image` 根本没有被触发

因此：

- 当前真实根因是 **原子提取层漏掉了 xref 级嵌入图片**
- 但一旦补上 xref 图片提取，这条 `page_as_image` 分支就会立即开始阻断这些页

结论：

- `page_as_image` 不是当前 blocker
- 它是 **Phase 1 上线后立即会爆炸的 latent mine**

### 1.2 AGFC-only 优势页精确为 5 个

edgefix4 相对于 MinerU 的精确矩阵是：

- `Both hit = 28`
- `AGFC-only hit = 5`
- `MinerU-only hit = 10`
- `Both miss = 21`

AGFC-only 命中的 5 页是：

- `test_000108`
- `test_000123`
- `test_000131`
- `test_000134`
- `test_000157`

初版文档列出的很多页面实际上是 `both-miss`，不能作为 AGFC 独特优势的证据。

### 1.3 剩余 `bbox_drift` 不是一种机制，而是 3 种独立机制

剩余 drift 页需要拆开看：

- `test_000105`
  - 只有一个 `image_cluster` seed
  - 没有 `same_row` / `same_column` 扩张
  - 问题是 **选中了错误视觉区域**，找到的是底部缩略图簇而不是主体图

- `test_000217`
  - 只有一个 `visual_community` seed
  - 没有 panel 间 BFS 扩张
  - 问题是 **seed 本身 bbox 就过大**

- `test_000188`
  - 同时存在 `image_seed` 和 `visual_community`
  - 这是唯一明确符合 `same_row` / `same_column` 扩张路径的案例

因此，不能把剩余 drift 统一归因为“闭包 BFS 膨胀”。

### 1.4 零-seed 页必须拆成 3 类，而不是 1 类

#### A. xref 图片存在，但 atom 层没恢复出来

- `test_000150`
- `test_000170`
- `test_000195`
- `test_000214`

特点：

- `page.get_images()` 能枚举到图片
- edgefix4 的 `atoms.json` 中却没有对应 `raster_image`

这类页应优先由 **xref 图片提取补全** 解决。

#### B. 图片已经被抽成 raster atom，但没升成 seed

- `test_000205`

特点：

- 已经有 `raster_image` atom
- 但没有 panel / seed
- 所以它不是纯 extraction miss，而是 **小图 seed 提升不足**

#### C. 纯 vector 弱原子页

- `test_000122`

特点：

- `page.get_images()` 返回 `0`
- 现有 atom 只有 `color_band / vector_cluster / text_block`
- 需要的是 **vector chart / weak visual seed**，不是 xref 图片补全

---

## 2. 下一阶段需要正视的 6 个有损通用性机制

下面这 6 项不是都同一优先级，但它们确实定义了下一阶段的技术债地图。

| ID | 严重度 | 代码位置 | 结论 | 说明 |
|---|---|---|---|---|
| `F2` | `P0` latent mine | `src/agfc/pipeline.py` `page_as_image` | 认同 | 当前没触发，但 xref 图片补全后会立即阻断全页图 |
| `F1` | `P1` | `src/agfc/visual_community.py` `_is_overexpanded_single_raster_community` | 认同方向，暂不升 P0 | 当前帮我们挡掉了坏社区，但它确实可能误杀“单图 + 大 legend / 大图注”的合法布局 |
| `F5` | `P2` | `src/agfc/visual_community.py` `_has_page_background_anchor` | 部分认同 | 设计上是传染式丢弃，但当前前置 `_is_visual_candidate()` 已先排掉大部分 page background，现实杀伤小于表面 |
| `F4` | `P1` | `src/agfc/pipeline.py` `_coverage_penalty` | 认同 | 只惩罚 `visual_community`，设计不对称 |
| `F3` | `P1` | `src/agfc/graph.py` / `src/agfc/bipolar_graph.py` | 认同 | 附着规则重复实现，维护风险真实存在；阈值是否还要继续收紧需数据支撑 |
| `F6` | `P1` | `src/agfc/visual_community.py` `_is_hero_visual_atom` | 认同 | 一个函数承担两种语义，调一个阈值会同时改两个行为面 |

### 2.1 必须立即处理

#### `F2` `page_as_image`

问题：

- 这是一个“未来一定会踩到”的硬杀分支
- 一旦 `collect_page_atoms()` 补上 xref 级图片，`test_000150/170/195` 这类全页图会立即被它过滤掉

原则：

- Phase 1 里必须和 xref 提取一起处理
- 不允许先补图片提取、后处理 `page_as_image`

### 2.2 应在下一阶段一并治理

#### `F1` 单 raster 社区硬杀

问题：

- `community_area / raster_area >= 4.0` 的社区被整体否掉
- 对当前数据它是有效修补
- 对更通用的学术布局，它有误杀风险

建议：

- 不改回“完全放开”
- 改成软化机制：降分、裁剪、或只排除明显装饰成员

#### `F4` coverage penalty 不对称

问题：

- 只有 `visual_community` 受罚
- 其他 seed 类型闭包即使异常大也不受约束

建议：

- 把 coverage 约束统一到 closure 层或 score 层
- 不要让 seed type 决定是否被约束

#### `F3` attachment 规则重复

问题：

- `graph.py` 与 `bipolar_graph.py` 里有重复的 attachment 合理性逻辑
- 未来继续调阈值时极易不同步

建议：

- 抽成共享 helper
- 阈值是否再收紧，基于新 benchmark 再判断

#### `F6` hero 语义耦合

问题：

- `_is_hero_visual_atom()` 同时承担：
  - “这个原子是不是 hero visual”
  - “小社区是否值得豁免晋升”

建议：

- 拆成两个语义明确的函数
- 例如：
  - `is_hero_visual_atom()`
  - `can_promote_small_visual_community()`

### 2.3 可以放到次级重构

#### `F5` page background 传染式丢弃

问题：

- 社区中只要有一个背景成员，整个社区就被放弃

现状判断：

- 这是不够优雅的设计
- 但当前 `_is_visual_candidate()` 已经把大部分 page background vector 在聚类前过滤掉
- 所以它不是当前第一优先级的 recall blocker

建议：

- 放到 visual community 统一重构时处理
- 做法是“剔除背景成员后重算社区”，而不是直接整团丢弃

---

## 3. 剩余 10 页失败的开发导向拆解

| row_id | root_cause | 验证后机制 | 下一步最可能有效的修复 |
|---|---|---|---|
| `test_000150` | `seed_recall_gap` | xref 图片未进入 atom 层 | xref 图片提取补全 + 移除 `page_as_image` 地雷 |
| `test_000170` | `seed_recall_gap` | 同上 | 同上 |
| `test_000195` | `seed_recall_gap` | 同上 | 同上 |
| `test_000214` | `seed_recall_gap` | xref 图片存在，但 atom 层没恢复 raster | xref 图片提取补全 |
| `test_000205` | `seed_recall_gap` | 已有 raster atom，但没有 seed | 小图 seed 提升策略 |
| `test_000122` | `seed_recall_gap` | 纯 vector 弱原子页 | vector chart / weak visual seed |
| `test_000105` | `bbox_drift` | 找到了错误视觉区域 | seed 选择 / panel proposal 精度 |
| `test_000188` | `bbox_drift` | 唯一明显 BFS/VC 扩张型 | closure / ranking 收紧 |
| `test_000217` | `bbox_drift` | visual_community seed 本身过大 | community proposal 裁剪 |
| `test_000058` | `wrong_target` | 有 11 个 seed 但排名选错 | ranking 信号重权重 |

结论：

- **批量提分的唯一高置信入口** 仍是 xref 图片提取补全
- 纯 vector 页、wrong target、错误区域选择，都是后续的小而硬问题

---

## 4. 下一阶段优先级

### Phase 1: xref 图片提取补全 + 解除 `page_as_image` 地雷

这是下一阶段最高优先级任务。

目标：

- 恢复 `test_000150 / 000170 / 000195 / 000214`
- 为 `test_000205` 提供更完整图片几何信息
- 避免全页图在 atom 恢复后又被 `page_as_image` 误杀

涉及文件：

- `src/agfc/atoms.py`
- `src/agfc/pipeline.py`
- `tests/test_pdf_agfc_atoms.py`
- `tests/test_pdf_agfc_pipeline.py`

必须同时满足：

1. `collect_page_atoms()` 能从 xref 级图片枚举恢复缺失的 `raster_image`
2. `_is_boilerplate_seed()` 不再把“全页主体图”视为 boilerplate

### Phase 2: 通用性治理

目标：

- 把 edgefix4 的有效经验修补，整理成更稳的通用机制

重点：

- 软化 `F1`
- 统一 `F4`
- 去重 `F3`
- 解耦 `F6`
- 顺带重写 `F5` 的背景成员处理方式

涉及文件：

- `src/agfc/visual_community.py`
- `src/agfc/pipeline.py`
- `src/agfc/graph.py`
- `src/agfc/bipolar_graph.py`
- `tests/test_pdf_agfc_visual_community.py`
- `tests/test_pdf_agfc_pipeline.py`
- `tests/test_pdf_agfc_graph.py`
- `tests/test_pdf_agfc_bipolar_graph.py`

### Phase 3: 目标化修补剩余 singleton failures

仅在 Phase 1 和 Phase 2 合并、重跑 benchmark 后再开始。

对象：

- `test_000122`
- `test_000205`
- `test_000105`
- `test_000188`
- `test_000217`
- `test_000058`

---

## 5. 子代理并行执行计划

下一步不要串行硬啃。应该按 **子代理并行、写集互斥、控制器统一验收** 的方式推进。

### 5.1 并行轨道划分

#### Track A: xref 图片提取轨

**目标**

- 在 atom 层补上 `page.get_images()` / xref 级图片恢复

**文件责任**

- 修改：`src/agfc/atoms.py`
- 测试：`tests/test_pdf_agfc_atoms.py`

**成功标准**

- 当 `text dict` 未报告 `type == 1` block，但页面存在 xref 图片时，仍能产出 `raster_image` atom
- 不重复产出与现有 text-dict image block 完全重叠的 image atom

**注意事项**

- 需要处理 bbox 来源与去重策略
- 必须避免对已存在的 raster atom 产生双份重复

#### Track B: 全页图语义轨

**目标**

- 解除 `page_as_image` latent mine
- 明确“全页主体图不是 boilerplate”

**文件责任**

- 修改：`src/agfc/pipeline.py`
- 测试：`tests/test_pdf_agfc_pipeline.py`

**成功标准**

- `area_ratio >= 0.85` 的单独主体 raster seed 不再被无条件过滤
- 小页眉 logo 过滤能力不回退

**依赖**

- 可与 Track A 并行开发
- 但合并时必须一起进，不能只合 A 不合 B

#### Track C: visual community 通用性轨

**目标**

- 把 edgefix4 的 `visual_community` 修补整理成更通用、可解释的规则

**文件责任**

- 修改：`src/agfc/visual_community.py`
- 测试：`tests/test_pdf_agfc_visual_community.py`

**涵盖问题**

- `F1` 单 raster 社区硬杀软化
- `F5` 背景成员先剔除再重评估
- `F6` hero 语义拆分

**成功标准**

- 现有 edgefix4 已修复页面不回退
- 函数语义边界更清晰，后续调参不再耦合

#### Track D: 图连接规则轨

**目标**

- 去重 attachment 逻辑
- 统一图构建和 bipolar 图构建的空间规则

**文件责任**

- 修改：`src/agfc/graph.py`
- 修改：`src/agfc/bipolar_graph.py`
- 可选新增：共享 helper 模块
- 测试：`tests/test_pdf_agfc_graph.py`
- 测试：`tests/test_pdf_agfc_bipolar_graph.py`

**成功标准**

- 两套实现不再手工保持一致
- 不丢失 edgefix4 修复过的 `000082/086/092` 类页面

### 5.2 控制器顺序

主控代理不应自己实现上述 4 轨，而应做这几件事：

1. 派发 4 个实现子代理，写集保持互斥
2. 每个子代理完成后，先做目标测试验证
3. 汇总后跑全量：
   - `python3 -m pytest -q`
4. 再重跑 benchmark：
   - `JournalMix-v1`
5. 更新：
   - `results.json`
   - `comparison.json`
   - `error_review.json`
   - `error_review.md`

### 5.3 推荐派发顺序

第一轮并行：

- Worker A: Track A `atoms.py`
- Worker B: Track B `pipeline.py`
- Worker C: Track C `visual_community.py`
- Worker D: Track D `graph.py` + `bipolar_graph.py`

第二轮串行验收：

- Controller 合并 A + B，先验证全页图与 xref 恢复
- Controller 合并 C + D，验证 edgefix4 已修页面不回退
- Controller 跑完整 benchmark

第三轮再决定是否进入 singleton 修补：

- `test_000122`
- `test_000205`
- `test_000105`
- `test_000188`
- `test_000217`
- `test_000058`

---

## 6. 下一步建议

如果只做一轮开发，就做：

1. **Track A**
2. **Track B**
3. **全量验证 + benchmark 重跑**

原因：

- 这是当前唯一高置信、批量提分、且理论上属于“修复框架遗漏”而不是“再加特例”的方向
- 它能最快回答一个最关键的问题：
  - AGFC 距离 MinerU 的剩余差距，究竟主要是 **atom recall**，还是 **closure / ranking 精度**

在这轮完成前，不建议把时间主要花在 `000122` 或 `000058` 这类 singleton 上。
