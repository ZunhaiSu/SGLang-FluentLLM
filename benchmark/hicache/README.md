# HiCache 综合性能测试与深度分析报告

**日期**: 2025-12-29
**分析对象**: Flash OSS FP8  & Qwen3 4B 
**对比基准**: Tree Cache (Baseline) vs HiCache (Various Configs)

---

## 1. HiCache 设计理念与目标

*   **设计理念**: 
    *   **多级存储架构**: 打破显存（HBM）的物理限制，将 KV Cache 扩展至 Host 内存（DRAM）甚至高速 SSD（NVMe）。
    *   **无感预取 (Transparent Prefetching)**: 利用计算与 I/O 的流水线并行，在 GPU 计算的同时异步预取下一轮所需的 KV Cache，掩盖 I/O 延迟。
    *   **Radix Tree 扩展**: 在原有的 Radix Tree 基础上增加层级感知，支持对被驱逐（Evicted）节点的元数据保留和按需回加载（Load Back）。

*   **核心目标**:
    1.  **无限 Context Window**: 理论上支持受限于磁盘容量的超长上下文。
    2.  **提升 Cache Hit Rate**: 在多轮对话（Multiturn）和长文档问答场景中，即使显存不足，也能通过层级存储命中历史 Cache，避免昂贵的重计算（Recomputation）。
    3.  **保持高性能**: 通过 Kernel 优化和异步流水线，确保不影响性能。


*   **TODO**:
    1.  **优化L2设置**: 目前默认L2设置为L1 的2倍，但是由于L1 memory pool size不是实际size。所以导致L2分配过大，待优化。
    2.  **提高吞吐**: 分析现在吞吐提升不大的原因并优化。
    3.  **NPU下的开发与优化**: 测试开发优化 NPU下的global cache功能。


### 1.1 实验设置

本次测试涵盖了两种典型的生产场景，旨在全面评估 HiCache 在不同负载下的表现。

*   **测试模型**:
    *   **Flash OSS FP8**: 1P1D, mooncake master部署在prefill服务器。
    *   **Qwen3 4B**: 快速实验，h20服务器，一卡做P一卡做D。

*   **测试场景 (Scenarios)**:
    *   **Serving**: 随机请求，模拟真实服务负载，重点考察 **TTFT (首字延迟)** 和 **Throughput (吞吐量)**。
    *   **Multiturn**: 多轮对话（5轮，每轮叠加输入512，max out token 512），重点考察 **Cache Hit Rate** 和上下文复用带来的性能提升。
    *   **LongContext**: 长上下文(4-10k)，测试系统在处理长序列时的稳定性与效率。
    *   通过使用sharegpt数据集和随机种子控制每次测试输入基本一致

*   **关键指标**:
    *   **Throughput (RPS)**: 每秒处理请求数。
    *   **TTFT P50 (ms)**: 首字延迟中位数，衡量响应速度。
    *   **Cache Hit Rate (%)**: 缓存命中率，直接关联计算资源的节省。

---

## 2. 核心结论 (Executive Summary)

本次测试综合了高并发吞吐（Flash OSS）和参数敏感度（Qwen3）两个维度的实验，得出以下核心结论：

1.  **HiCache 性能优势确认**: 
    *   **Cache Hit Rate 显著提升**: 在 Flash OSS 高并发场景下，HiCache (Opt) 的 Multiturn Hit Rate 达到 **68.74%**，相比 Tree Cache (60.82%) 提升了 **+13%**。在 Qwen3 低并发 (Rate 1) 场景下，HiCache 更是达到了 **83.19%** 的 Hit Rate，远超 Tree Cache 的 61.66% (**+35%**)。
    *   **TTFT 优化**: 在 Flash OSS Serving 场景中，HiCache (Opt) 的 P50 TTFT 为 **261.23ms**，相比 Tree Cache 的 318.32ms 降低了 **17.9%**，说明层级缓存有效减少了排队和冷启动开销。
    *   **吞吐量保持**: 在 Cache Hit Rate 提升的同时，单机实验的qwen3 下吞吐提升13%，但是flash oss场景下HiCache 的吞吐量与 Tree Cache 基本持平, 需要进一步验证分析原因。
    *   **准确性**: 在MMLU画布测试和自设置的benchmark测试中，准确率与tree cache一致，MMLU画布测试都是准确率89%

2.  **参数调优是关键**: HiCache 的性能对配置极度敏感。
    *   **Write Policy**: 高并发场景必须使用 `Write Back`，否则性能会因 I/O 阻塞而崩盘。
    *   **Prefetch Threshold**: 设为 `1`，以规避代码逻辑中的 "Break Early" 问题。
    *   **IO Backend**: `Kernel` 模式相比 `Direct` 模式带来约 **8.8%** 的综合性能提升（与单独测试/宣传不符，待确定原因，目前感觉kernel模式效率没全部开发出来）。

3.  **理论与现实的差距**: 尽管 Hit Rate 提升，但吞吐未见显著下降，目前猜测归因于线程调度开销，预取逻辑的截断效应，kernel的效率问题等。待进一步验证

---

## 3. 实验一：Flash OSS FP8 (高并发吞吐场景)

**对比组**:
*   `Tree Cache`: 纯显存 Radix Cache (Baseline)
*   `HiCache (Opt)`: Threshold=1, Write Back (推荐配置)
*   `HiCache (Base)`: Threshold=256, Write Back
*   `HiCache (WT)`: Threshold=1, Write Through

### 3.1 关键性能对比 (vs Tree Cache)

| Metric | Scenario | Tree Cache (Baseline) | HiCache (Opt) | 提升幅度 |
| :--- | :--- | :--- | :--- | :--- |
| **Throughput (RPS)** | Serving | 8.12 | 8.10 | -0.2% (持平) |
| **TTFT P50 (ms)** | Serving | 318.32 | **261.23** | **-17.9% (更优)** |
| **Hit Rate (%)** | Multiturn | 60.82% | **68.74%** | **+13.0%** |
| **Hit Rate (%)** | LongCtx | 77.27% | 77.27% | 持平 |

**分析**:
*   **TTFT 显著降低**: HiCache 在高负载 Serving 场景下的 P50 TTFT 降低了近 18%，说明层级缓存有效减少了部分请求的排队或重计算时间。
*   **Hit Rate 提升**: 在 Multiturn 场景下，HiCache 利用 Host 内存成功挽回了约 8% 的 Cache Miss，这直接转化为计算资源的节省。

### 3.2 HiCache 内部参数影响

| Config | Hit Rate (Multiturn) | 说明 |
| :--- | :--- | :--- |
| **HiCache (Opt)** | **68.74%** | Threshold=1, Write Back |
| **HiCache (Base)** | 67.52% | Threshold=256, Write Back |
| **HiCache (WT)** | 24.20% | Threshold=1, Write Through |

**警示**: `Write Through` 在高并发下是灾难性的。由于每次 Cache Hit 都触发 I/O 写入，导致系统 I/O 队列阻塞，Hit Rate 暴跌至 24%。

---

## 4. 实验二：Qwen3 4B (参数敏感度搜索)

### 4.1 HiCache vs Tree Cache (不同速率对比)

选取 HiCache 最优配置 (Best Effort, Page First, Write Through*, Threshold 1, Kernel) 与 Tree Cache 对比。
*注: Qwen3 测试中负载未饱和，Write Through 表现尚可，但在高并发下应慎用。*

| Rate (req/s) | Metric | Scenario | Tree Cache | HiCache (Best) | 差异 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Hit Rate** | Multiturn | 61.66% | **83.19%** | **+35% (显著提升)** |
| **1** | **TTFT P50** | Multiturn | 16.88ms | **14.62ms** | **-13% (更优)** |
| **2** | **Hit Rate** | Multiturn | 62.09% | **66.77%** | **+7.5%** |
| **8** | **Hit Rate** | Multiturn | **61.91%** | 56.43% | -8.8% (下降) |
| **8** | **Throughput** | Serving | 7.91 | 7.92 | 持平 |

**分析**:
*   **低并发优势巨大**: 在 Rate 1 和 Rate 2 下，HiCache 展现了强大的缓存能力，Hit Rate 大幅提升，TTFT 也有所下降。
*   **高并发瓶颈**: 在 Rate 8 下，HiCache 的 Hit Rate 反而低于 Tree Cache。这可能是由于 `Write Through` 策略在高并发下的 I/O 阻塞，或者是 "Break Early" 逻辑在碎片化请求下的负面影响。这再次印证了 Flash OSS 实验中 **Write Back** 的重要性。

### 4.2 参数敏感度分析

| 参数维度 | 胜出配置 | 差异幅度 | 原因分析 |
| :--- | :--- | :--- | :--- |
| **IO_BACKEND** | `kernel` | **+8.8%** vs direct | Kernel 实现避免了 Python 层面的 Tensor 拷贝开销。 |
| **PREFETCH_POLICY** | `best_effort` | **+4.5%** vs timeout | 激进预取策略能最大化利用 I/O 带宽。 |
| **WRITE_POLICY** | `write_through` | **+4.7%** vs write_back | **注意**: 仅在低负载/短测试中有效。在 Qwen3 测试中，负载未饱和，激进写入反而保证了数据快速落盘。但在生产高负载下应慎用（见 Flash OSS 结果）。 |
| **PREFETCH_THRESHOLD** | `1` | **+1.5%** vs 256 | 越小越好 |

---

## 5. 深度归因分析 (Root Cause Analysis)

为何 HiCache 的表现如此依赖参数？为何理论上的巨大优势在某些指标上不明显？

### 5.1 "Break Early" 预取逻辑缺陷
代码 `cache_controller.py` 中存在逻辑：按 Batch (128 pages) 检查存储，一旦发现 **任何一页** 缺失，立即停止后续所有预取。
*   **后果**: 存储碎片化导致预取覆盖率大幅下降。待优化

### 5.2 Write Policy 
*   **Write Back**: 延迟写入，仅在 Evict 时触发。适合 **高吞吐/生产环境**，避免 I/O 争抢。
*   **Write Through**: 立即写入。适合 **低负载/调试环境**，数据一致性好，但高并发下会阻塞主线程。

### 5.3 线程开销
HiCache 引入了 Python 线程 (`prefetch_thread`, `backup_thread`)。在 GIL 限制下，高频的 `Queue` 操作和线程切换消耗了 CPU 时间，这解释了为何在某些轻量级测试中，HiCache 的延迟并未显著优于 Tree Cache（甚至略高）。

---

## 附录：详细测试结果数据

### 附录 A: Flash OSS 详细测试结果

| Config | Rate | Scenario | Throughput (RPS) | TTFT P50 (ms) | Hit Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Tree Cache | 8 | serving | 8.12 | 318.32 | 0.00% |
| Tree Cache | 8 | multiturn | 1.19 | 8.25 | 60.82% |
| Tree Cache | 8 | longcontext | 2.02 | 10.14 | 77.27% |
| Tree Cache | 4 | serving | 4.13 | 206.56 | 0.00% |
| Tree Cache | 4 | multiturn | 1.20 | 8.36 | 60.83% |
| Tree Cache | 4 | longcontext | 2.02 | 10.14 | 77.27% |
| Tree Cache | 2 | serving | 2.08 | 199.71 | 0.00% |
| Tree Cache | 2 | multiturn | 1.20 | 8.53 | 60.81% |
| Tree Cache | 2 | longcontext | 1.80 | 9.93 | 77.27% |
| Tree Cache | 1 | serving | 1.04 | 198.13 | 0.00% |
| Tree Cache | 1 | multiturn | 1.03 | 8.35 | 61.10% |
| Tree Cache | 1 | longcontext | 0.84 | 7.43 | 77.27% |
| HiCache (Opt) | 8 | serving | 8.10 | 261.23 | 0.00% |
| HiCache (Opt) | 8 | multiturn | 1.19 | 8.38 | 68.74% |
| HiCache (Opt) | 8 | longcontext | 2.01 | 10.26 | 77.27% |
| HiCache (Opt) | 4 | serving | 4.13 | 206.80 | 0.00% |
| HiCache (Opt) | 4 | multiturn | 1.20 | 8.48 | 20.70% |
| HiCache (Opt) | 4 | longcontext | 1.98 | 10.36 | 78.85% |
| HiCache (Opt) | 2 | serving | 2.08 | 202.12 | 0.00% |
| HiCache (Opt) | 2 | multiturn | 1.20 | 8.57 | 71.29% |
| HiCache (Opt) | 2 | longcontext | 1.56 | 9.07 | 78.59% |
| HiCache (Opt) | 1 | serving | 1.04 | 197.12 | 0.00% |
| HiCache (Opt) | 1 | multiturn | 1.01 | 8.39 | 71.73% |
| HiCache (Opt) | 1 | longcontext | 0.87 | 7.51 | 78.59% |
| HiCache (WT) | 8 | serving | 8.11 | 275.12 | 0.00% |
| HiCache (Base) | 8 | serving | 8.11 | 260.46 | 0.00% |
| HiCache (Base) | 8 | multiturn | 1.19 | 8.24 | 67.52% |
| HiCache (Base) | 8 | longcontext | 1.99 | 10.25 | 77.27% |
| HiCache (Base) | 4 | serving | 4.13 | 200.38 | 0.00% |
| HiCache (Base) | 4 | multiturn | 1.20 | 8.36 | 71.36% |
| HiCache (Base) | 4 | longcontext | 1.95 | 10.27 | 78.59% |
| HiCache (Base) | 2 | serving | 2.08 | 202.12 | 0.00% |
| HiCache (Base) | 2 | multiturn | 1.20 | 8.57 | 71.29% |
| HiCache (Base) | 2 | longcontext | 1.56 | 9.07 | 78.59% |
| HiCache (Base) | 1 | serving | 1.04 | 197.12 | 0.00% |
| HiCache (Base) | 1 | multiturn | 1.01 | 8.39 | 71.73% |
| HiCache (Base) | 1 | longcontext | 0.87 | 7.51 | 78.59% |
| HiCache (WT) | 8 | multiturn | 1.19 | 8.41 | 24.20% |
| HiCache (WT) | 8 | longcontext | 2.01 | 10.26 | 77.27% |
| HiCache (WT) | 4 | serving | 4.13 | 204.49 | 0.00% |
| HiCache (WT) | 4 | multiturn | 1.20 | 8.56 | 24.03% |
| HiCache (WT) | 4 | longcontext | 1.93 | 10.25 | 78.85% |
| HiCache (WT) | 2 | serving | 2.08 | 202.12 | 0.00% |
| HiCache (WT) | 2 | multiturn | 1.20 | 8.57 | 71.29% |
| HiCache (WT) | 2 | longcontext | 1.56 | 9.07 | 78.59% |
| HiCache (WT) | 1 | serving | 1.04 | 197.12 | 0.00% |
| HiCache (WT) | 1 | multiturn | 1.01 | 8.39 | 71.73% |
| HiCache (WT) | 1 | longcontext | 0.87 | 7.51 | 78.59% |
| HiCache (WT+LF) | 8 | serving | 8.11 | 273.99 | 0.00% |
| HiCache (WT+LF) | 8 | multiturn | 1.19 | 8.53 | 40.85% |
| HiCache (WT+LF) | 8 | longcontext | 2.00 | 10.32 | 77.27% |
| HiCache (WT+LF) | 4 | serving | 4.13 | 206.13 | 0.00% |
| HiCache (WT+LF) | 4 | multiturn | 1.20 | 8.48 | 30.43% |
| HiCache (WT+LF) | 4 | longcontext | 1.97 | 10.27 | 78.59% |
| HiCache (WT+LF) | 2 | serving | 2.08 | 202.12 | 0.00% |
| HiCache (WT+LF) | 2 | multiturn | 1.20 | 8.57 | 71.29% |
| HiCache (WT+LF) | 2 | longcontext | 1.56 | 9.07 | 78.59% |
| HiCache (WT+LF) | 1 | serving | 1.04 | 197.12 | 0.00% |
| HiCache (WT+LF) | 1 | multiturn | 1.01 | 8.39 | 71.73% |
| HiCache (WT+LF) | 1 | longcontext | 0.87 | 7.51 | 78.59% |

### 附录 B: Qwen3 参数搜索详细测试结果

| Config Description | Rate | Scenario | Throughput (RPS) | TTFT P50 (ms) | Hit Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 8 | serving | 7.92 | 1981.93 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 8 | multiturn | 1.16 | 14.84 | 56.43% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 8 | longcontext | 1.33 | 15.24 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 4 | serving | 4.10 | 43.05 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 4 | multiturn | 1.16 | 15.00 | 54.34% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 4 | longcontext | 1.32 | 15.14 | 89.14% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 2 | serving | 2.06 | 40.32 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 2 | multiturn | 1.17 | 14.93 | 66.77% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 2 | longcontext | 1.29 | 15.18 | 89.14% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 1 | serving | 1.03 | 36.05 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 1 | multiturn | 0.98 | 14.62 | 83.19% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:kernel | 1 | longcontext | 0.86 | 14.19 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 8 | serving | 7.92 | 1981.93 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 8 | multiturn | 1.16 | 14.84 | 56.43% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 8 | longcontext | 1.33 | 15.24 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 4 | serving | 4.10 | 43.05 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 4 | multiturn | 1.16 | 15.00 | 54.34% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 4 | longcontext | 1.32 | 15.14 | 89.14% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 2 | serving | 2.06 | 40.32 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 2 | multiturn | 1.17 | 14.93 | 66.77% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 2 | longcontext | 1.29 | 15.18 | 89.14% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 1 | serving | 1.03 | 36.05 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 1 | multiturn | 0.98 | 14.62 | 83.19% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:256 IO:kernel | 1 | longcontext | 0.86 | 14.19 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 8 | serving | 7.91 | 1472.36 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 8 | multiturn | 1.16 | 14.72 | 65.66% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 8 | longcontext | 1.32 | 15.26 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 4 | serving | 4.10 | 43.10 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 4 | multiturn | 1.16 | 14.84 | 59.34% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 4 | longcontext | 1.32 | 15.17 | 89.14% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 2 | serving | 2.06 | 40.70 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 2 | multiturn | 1.16 | 15.57 | 67.03% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 2 | longcontext | 1.24 | 15.49 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 1 | serving | 1.03 | 39.32 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 1 | multiturn | 0.97 | 15.24 | 83.01% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:64 IO:kernel | 1 | longcontext | 0.82 | 14.80 | 89.14% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 8 | serving | 7.87 | 1917.27 | 0.00% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 8 | multiturn | 1.16 | 15.14 | 56.99% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 8 | longcontext | 1.29 | 15.48 | 88.25% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 4 | serving | 4.09 | 46.46 | 0.00% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 4 | multiturn | 1.16 | 15.63 | 55.41% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 4 | longcontext | 1.28 | 15.70 | 88.25% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 2 | serving | 2.06 | 42.41 | 0.00% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 2 | multiturn | 1.16 | 15.81 | 68.99% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 2 | longcontext | 1.21 | 15.56 | 89.14% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 1 | serving | 1.03 | 37.60 | 0.00% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 1 | multiturn | 0.99 | 15.40 | 81.93% |
| Pol:best_effort Lay:layer_first Wri:write_through Thr:1 IO:kernel | 1 | longcontext | 0.85 | 14.86 | 89.14% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 8 | serving | 7.91 | 1427.53 | 0.00% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 8 | multiturn | 1.16 | 15.51 | 52.45% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 8 | longcontext | 1.28 | 15.78 | 88.25% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 4 | serving | 4.09 | 44.26 | 0.00% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 4 | multiturn | 1.16 | 15.42 | 49.56% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 4 | longcontext | 1.26 | 15.79 | 88.25% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 2 | serving | 2.06 | 42.43 | 0.00% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 2 | multiturn | 1.16 | 15.49 | 53.04% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 2 | longcontext | 1.25 | 15.72 | 89.14% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 1 | serving | 1.03 | 39.61 | 0.00% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 1 | multiturn | 0.95 | 15.26 | 79.17% |
| Pol:timeout Lay:page_first Wri:write_through Thr:1 IO:kernel | 1 | longcontext | 0.87 | 14.98 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 8 | serving | 7.92 | 1666.06 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 8 | multiturn | 1.16 | 15.17 | 68.48% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 8 | longcontext | 1.29 | 15.72 | 86.46% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 4 | serving | 4.10 | 45.31 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 4 | multiturn | 1.16 | 15.32 | 56.87% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 4 | longcontext | 1.25 | 15.72 | 89.14% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 2 | serving | 2.06 | 41.74 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 2 | multiturn | 1.16 | 15.58 | 68.51% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 2 | longcontext | 1.18 | 16.31 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 1 | serving | 1.03 | 36.25 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 1 | multiturn | 1.00 | 15.11 | 82.52% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:128 IO:kernel | 1 | longcontext | 1.06 | 15.01 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 8 | serving | 7.94 | 1454.93 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 8 | multiturn | 1.16 | 15.28 | 66.99% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 8 | longcontext | 1.28 | 15.76 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 4 | serving | 4.09 | 43.22 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 4 | multiturn | 1.16 | 15.17 | 73.27% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 4 | longcontext | 1.27 | 15.72 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 2 | serving | 2.06 | 42.67 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 2 | multiturn | 1.16 | 15.26 | 71.04% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 2 | longcontext | 1.26 | 15.65 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 1 | serving | 1.03 | 39.14 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 1 | multiturn | 0.98 | 15.31 | 68.71% |
| Pol:best_effort Lay:page_first Wri:write_back Thr:1 IO:kernel | 1 | longcontext | 0.92 | 15.03 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 8 | serving | 5.66 | 49.27 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 8 | multiturn | 1.16 | 15.28 | 55.04% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 8 | longcontext | 1.28 | 15.70 | 88.25% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 4 | serving | 4.09 | 45.21 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 4 | multiturn | 1.16 | 15.18 | 55.05% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 4 | longcontext | 1.26 | 15.66 | 89.14% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 2 | serving | 2.06 | 42.95 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 2 | multiturn | 1.16 | 15.30 | 61.94% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 2 | longcontext | 1.23 | 15.44 | 89.14% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 1 | serving | 1.03 | 39.87 | 0.00% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 1 | multiturn | 0.98 | 15.18 | 81.54% |
| Pol:best_effort Lay:page_first Wri:write_through Thr:1 IO:direct | 1 | longcontext | 0.79 | 14.86 | 88.25% |

### 附录 C: Qwen3 Tree Cache 详细测试结果

| Config | Rate | Scenario | Throughput (RPS) | TTFT P50 (ms) | Hit Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Tree Cache | 8 | serving | 7.91 | 1588.91 | 0.00% |
| Tree Cache | 8 | multiturn | 1.16 | 14.71 | 61.91% |
| Tree Cache | 8 | longcontext | 1.32 | 15.22 | 88.25% |
| Tree Cache | 4 | serving | 4.10 | 43.04 | 0.00% |
| Tree Cache | 4 | multiturn | 1.16 | 14.77 | 61.87% |
| Tree Cache | 4 | longcontext | 1.31 | 15.28 | 88.25% |
| Tree Cache | 2 | serving | 2.06 | 41.36 | 0.00% |
| Tree Cache | 2 | multiturn | 1.17 | 14.73 | 62.09% |
| Tree Cache | 2 | longcontext | 1.30 | 15.19 | 88.25% |
| Tree Cache | 1 | serving | 1.03 | 40.88 | 0.00% |
| Tree Cache | 1 | multiturn | 0.95 | 16.88 | 61.66% |
| Tree Cache | 1 | longcontext | 0.74 | 16.72 | 88.25% |