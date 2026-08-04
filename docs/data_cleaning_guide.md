# SensorSync 数据清洗指南

本文基于当前 **SensorSync DataOps Studio** 实现，说明从质量检测到时间戳对齐、再到问题修复的完整数据清洗流程。适用于演示讲解、面试叙事与后续工程扩展对照。

---

## 1. 目标与原则

### 1.1 要解决什么问题

机器人多模态采集（RGB / Depth / Joint / Force / TCP / Gripper）常见三类问题：

1. **时间不同步**：相机与关节/力传感器存在几十毫秒级时钟偏差  
2. **质量缺陷**：丢帧、模糊、深度空洞、力尖峰、TCP 跳变  
3. **标签不可用**：技能段边界不准或置信度偏低  

清洗目标是：把原始 Episode 变成**可对齐、可审计、可导出**的训练数据候选，并给出 Pass / Review / Reject 决策。

### 1.2 当前实现的关键原则

| 原则 | 说明 |
|------|------|
| 原始数据只读 | Align / Clean **不改写**磁盘上的 parquet、mp4、metadata 传感器内容 |
| 内存视图层 | 对齐结果、清理状态保存在后端进程内存中；重启后需重做 |
| 可对比 | Align 后仍可用 Raw / Aligned 时间轴对比 |
| 策略标签 + 演示落地 | Interpolate / Repair 等是**治理策略名**；demo 中更新 issue 状态与分数，并非全部完成像素级修复 |

---

## 2. 推荐操作流程

```text
选择 Episode
    ↓
① 质量检测（加载时自动 + Overview）
    ↓
② 时间戳对齐（Sync：Estimate → Apply）
    ↓
③ 问题修复（Quality：Detect / Clean）
    ↓
④ 技能标签微调（Labels）
    ↓
⑤ 导出 Dataset Card（Export）
```

也可使用顶栏 **Run Pipeline**：等价于对当前 Episode 依次执行 **Apply Alignment → Detect / Clean Issues**，然后跳到 Labels。

### 2.1 启动

```bash
docker compose up --build
```

- UI：http://localhost:3000  
- API：http://localhost:8000  

左侧选择合成 Episode（如 `EP_0042`）或导入的 `HF_*` Episode。

---

## 3. 质量检测（Quality Analysis）

### 3.1 何时触发

- **打开 Episode 时自动运行**（`analyze_episode`）  
- 左侧列表展示状态与分数  
- 右侧 **Overview** 展示决策原因、四维权重、传感器诊断、模糊统计  
- **Quality** 页可再次点击 **Detect / Clean Issues** 强制重检并清理  

实现：`backend/app/services/quality.py`

### 3.2 检测项与阈值

| 检测项 | 输入 | 判定逻辑（当前） | 产出 |
|--------|------|------------------|------|
| Sync offset | RGB/Depth/FT vs Joint 时间戳 | 最近邻中位偏差；合成数据优先用 injected offset | `sync_error_ms` |
| Drop rate | RGB 时间间隔 | 相对期望帧率的缺口比例 | `dropped_frames_pct` |
| Blur | RGB `blur_score`（Laplacian） | `< 40` 视为模糊 | issue `blur` |
| Depth hole | Depth `valid_ratio` | `< 0.65` 记缺失 | issue `depth_missing` |
| Force spike | Force `fz` | `\|Fz\| > 40 N` | issue `force_spike` |
| TCP jump | TCP xyz 差分 | 位移 `> 5 cm` | issue `tcp_jump` |
| Label confidence | 技能段 `confidence` 平均 | `< 0.7` 扣分；`< 0.9` 易进 Review | 分数维度 |

### 3.3 综合评分（0–100）

分数从 **100** 起扣，四维权重（最大扣分占比）：

| 维度 | 权重 | 公式（摘要） |
|------|------|----------------|
| Sync Alignment | 35% | `min(35, \|sync_ms\| × 0.35)` |
| Frame Continuity | 20% | `min(20, drop% × 2.5)` |
| Sensor Faults | 37% | `high×6 + medium×3 + (失败任务×15)`，封顶 37 |
| Label Confidence | 8% | 置信度 `< 0.7` 则 −8 |

Overview 中的 **Quality score · 4 dimensions & weights** 会显示每维实际扣分与公式。

### 3.4 自动决策：Pass / Review / Reject

硬规则示例（详见 `THRESHOLDS`）：

- Sync `> 50 ms` → Reject  
- Drop `> 5%` → Reject  
- Sync `> 10 ms` 或存在可操作 issue → 至少 Review  
- 分数 `< 40` → Reject  
- 分数 `< 85` 且有问题 → 多为 Review  

Overview 的 **Auto Decision** 与 **Why this status** 列出具体原因。

---

## 4. 时间戳对齐（Sync）

对齐解决的是：**多传感器时钟不一致 + 频率不一致**，使训练样本能在统一时间栅格上取值。

实现：`backend/app/services/pipeline.py`

### 4.1 两步操作

#### Step A — Estimate Offset

1. 选定 **Reference Clock**（Joint / RGB / Depth / FT / PTP）  
2. 选定 **Target Rate Hz**（如 10 / 20 / 30 / 50，或手输）  
3. 点击 **Estimate Offset**

偏移来源：

- **合成数据**：`metadata.injected.offsets_ms`（如 RGB +48 ms, Depth +55 ms, FT −8 ms）  
- **HF 导入**：对各传感器时间戳相对 Joint 做最近邻中位估计  

再按参考时钟做坐标变换（例如以 RGB 为参考时，RGB offset 变为 0）。

平均 sync error（演示）：

```text
0.85 × |rgb_ms| + 0.15 × |depth_ms|
```

#### Step B — Apply Alignment

点击 **Apply Alignment** 后（**不改磁盘**）：

1. 保存当前 `reference_clock`、`target_rate_hz`  
2. 生成 `alignment_report`（前后 sync error、各通道 offset、变更列表）  
3. 标记 `alignment_applied = true`  
4. 对齐后残差在演示中约为 **~5 ms**  

可随时用时间轴 **Raw Timeline / Aligned Timeline** 对比未对齐与已对齐。

### 4.2 对齐内部顺序（重要）

对 Aligned 视图：

```text
① 对各传感器做时间平移（apply offset）
      t' = t_raw − offset_sensor

② 按 Target Rate 生成统一时间栅格
      t = 0, 1/rate, 2/rate, …

③ 在每个 t 上，对每个 sensor 最近邻取样
      → playback[] 供四宫格同步回放
```

因此高频 Joint（~500 Hz）、Force（~1 kHz）会被**降采样到目标频率**（如 20 Hz），与 RGB 同一套时间戳。

### 4.3 Raw vs Aligned

| 模式 | 行为 |
|------|------|
| Raw | 保留各通道 offset，时间轴上可见错位；黄区提示 offset |
| Aligned | offset 按 0 处理（时间已扳齐），playback 按目标 Hz 取样 |

Sync 面板在 Apply 后会显示 before/after 对照与 **What changed**。

### 4.4 关于重采样方法说明

界面标注：RGB Nearest · Joint Linear · TCP SLERP · Force Lowpass · Event ZOH。

- 这些方法名会记入 alignment report（设计意图 / 面试叙事）  
- **当前回放主路径**主要是：时间平移 + **最近邻取样** + 目标频率栅格  
- `_linear` / `_slerp` / `_lowpass` 已在代码中预留，完整按模态导出时可接上  

---

## 5. 问题修复（Clean / Repair）

### 5.1 入口

右侧 **Quality** → **Detect / Clean Issues**

实现：`clean_episode`（`pipeline.py`）+ 检测逻辑（`quality.py`）

### 5.2 流程

```text
重新检测全部 issues
    ↓
按 action 分流
    ├─ 可自动修：Interpolate / Repair / Trim / Keep → resolved
    └─ 需人工：Reject / Needs Review → remaining
    ↓
用剩余 issues 重算质量分与 Pass/Review/Reject
    ↓
更新 Overview / Quality 列表 / 时间轴异常色块
```

### 5.3 Action 何时使用

| Action | 触发场景 | 判定条件（当前） |
|--------|----------|------------------|
| **Interpolate** | RGB 可修丢帧 | 帧间隔 &gt; 1.8×期望，且 gap ≤ max(6×期望, 0.25s) |
| **Interpolate** | Depth 较重缺失 | `valid_ratio < 0.65` 且非 low severity |
| **Repair** | RGB 模糊 | 存在 `blur_score < 40` |
| **Trim** | TCP 跳变 | 相邻位姿位移 &gt; 5 cm |
| **Keep** | Depth 轻微空洞 | severity = low（缺失较轻） |
| **Needs Review** | 中等力尖峰 | 40 N ≤ \|Fz\| &lt; 100 N |
| **Reject** | 严重丢帧或大力尖峰 | gap 过大，或 \|Fz\| ≥ 100 N |

可自动修的四类会在 Clean 后进入 **Resolved** 列表；Reject / Needs Review 留在 **Remaining**。

### 5.4 修复在当前软件中的真实含义

| Action | Demo 实际效果 | 真实产线可扩展为 |
|--------|---------------|------------------|
| Interpolate | issue 出库、drop 对分数影响降低 | 缺口处插时间戳；RGB 光流/复制帧；Depth 线性/最近邻填洞；或仅在导出栅格上 nearest |
| Repair | blur issue 出库，分数上升 | 换清晰邻帧 / 裁糊段 / 学习式去模糊（需评估训练伪影） |
| Trim | TCP jump issue 出库 | 裁掉跳变邻域时间窗，缩短 episode |
| Keep | 轻症直接接受 | 保留并记录，不阻塞 Pass |

**再次强调**：当前 Clean **不修改**原始图像与力/位姿文件；模糊统计等诊断仍反映原始测量。这是「策略级修复 + 质量门禁」，不是完整的信号重建流水线。

### 5.5 Clean 后的可见变化

- Quality：清理报告（resolved/remaining、分数前后）  
- Overview：质量分与决策可能改善  
- 时间轴：异常色块减少（只渲染剩余 issues）  
- 状态栏：如 `Cleaned 4/5 · score 63 → 87 · 1 left`  

---

## 6. 技能标签与导出（清洗闭环后）

### 6.1 Labels

- 展示技能段（Approach / Grasp / Insert …）及 confidence  
- 可微调 start/end；Label Confidence = 各段 confidence 平均  
- 影响质量分中的 Label 维度与 Review 规则  

### 6.2 Export

- 选择格式（LeRobot / RLDS / HDF5 / Parquet）  
- **Create Dataset v1.2** 生成 Dataset Card（接受/拒绝/人工复核统计、平均 sync error、血缘 lineage）  
- 演示导出侧重卡片与可追溯叙事；完整训练格式写出可按同样对齐参数扩展  

---

## 7. 端到端演示脚本（约 3 分钟）

1. 打开 `EP_0042` → Overview 见 Auto Decision（多为 Review）与四维扣分  
2. Sync → Estimate Offset（RGB ~+48 ms）→ 设 Target Rate（如 20 Hz）→ Apply Alignment  
3. 切换 Raw / Aligned，说明错位消失、高频被抽到统一栅格  
4. Quality → Detect / Clean Issues → 展示 Resolved vs Remaining  
5. Labels 微调一段边界  
6. Export → 展示 Dataset Card  

或：顶栏 **Run Pipeline** 自动完成 2+4，再口述 Labels / Export。

---

## 8. 数据与 API 索引

### 8.1 Episode 落盘结构（示意）

```text
sample_data/episodes/<EP_ID>/
  metadata.json          # 时长、offsets、labels、injected 缺陷等
  camera_rgb.parquet
  camera_depth.parquet
  joint_state.parquet
  force_torque.parquet
  tcp_pose.parquet
  gripper_state.parquet
  rgb.mp4 / depth ...
```

### 8.2 关键 API

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/episodes/{id}` | 详情 + 质量报告 + sync/clean report |
| GET | `/episodes/{id}/timeline?mode=raw\|aligned` | 时间轴与 playback |
| POST | `/episodes/{id}/estimate_offset` | 估计偏移 |
| POST | `/episodes/{id}/align` | 应用对齐（内存） |
| POST | `/episodes/{id}/clean` | 检测并清理（内存） |
| POST | `/episodes/{id}/analyze` | 强制重分析 |
| POST | `/datasets/lerobot/import` | 导入 HF LeRobot |

---

## 9. 面试一句话总结

> SensorSync 把数据清洗做成可操作闭环：先自动质量检测并打分决策，再按参考时钟估计偏移并对齐到统一采样率，再按策略自动修复可恢复缺陷、拦截需人工或拒绝的样本，最后带血缘导出供技能微调——全程保留原始数据，对齐与清理以可对比、可审计的视图层落地。

---

## 10. 已知边界（与扩展方向）

1. Align / Clean 状态在进程内存中，重启需重跑  
2. Interpolate / Repair 尚未做像素/信号级写回  
3. 回放取样以 nearest + 目标 Hz 为主；SLERP/Lowpass 为扩展点  
4. HF 导入若缺力/深度，可能由 state/action 代理，诊断语义需注意  

扩展建议：Clean 时写出 `cleaned/` 旁路副本；导出时真正按模态重采样；模糊段支持邻帧替换预览。
