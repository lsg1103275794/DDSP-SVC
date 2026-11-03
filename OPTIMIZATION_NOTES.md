# DDSP-SVC 第一批优化说明（分支 `opt/perf-batch1`）

> 写给醒来后的你。先看「一页结论」，细节和证据在后面各节。

## 一页结论

- **基线**：`b9d4e6a`（上游 6.3）。**工作目录**：独立 git worktree（分支 `opt/perf-batch1`）。
- **主检出未动**：原有检出仍在 `exp/mert-encoder`，未提交的 MERT 改动都在，没有 checkout / stash / reset。
- **没有合并进任何主线，Studio 也没有接入这个分支**：Studio 的配置仍指向原 DDSP-SVC 检出。要不要接入、怎么合并见第 10 节。
- **候选来源**：一次静态代码调研（结果不在本仓库），按收益、风险和证据强度选出前五项 C01–C05。

| 项 | 提交 | 改了什么 | 实测结论 | 默认行为是否改变 |
|---|---|---|---|---|
| C04 | `575113a` | Hann 窗口缓存按真实窗口长度/设备/dtype 做 64 项 LRU | **证实**：1 万次调用后缓存 10,008 → 64 条，驻留约 79 MiB → 0.5 MiB；Hann/mel 逐位相同 | 否（数值不变） |
| C01 | `aa71e01` | DDSP 几何常量改用 Python 整数做形状控制，保留 state_dict 键 | **同步次数证实下降**（每次前向 18 → 1）；输出/梯度逐位相同；**时延无稳健结论**（两轮一慢 7.5%、一快 19%） | 否；几何与配置不一致的 checkpoint 现在会明确报错 |
| C03 | `1855377` | checkpoint 先在 CPU 加载再一次性搬上 GPU | **证实**：冷加载显存峰值 827.5 MB → 413.8 MB（减半）；202 个参数哈希相同，转换结果逐位相同；**加载时间没有变快**（首轮慢 18.7%） | 否 |
| C02 | `1ee2448` | 验证拆成「仅 loss / 预览子集 / 完整审计」三种模式和独立频率 | 仅 loss 模式验证计算耗时约 **−93%**；受控 RNG 下保留的预览逐位相同 | **否**：默认仍是完整审计、fp32、旧 RNG，需要你手动启用 |
| C05 | `9b7c4b0` | 只有 fp16 才用 GradScaler；非有限梯度不更新；LR 和步数只在真实更新后推进 | 保护逻辑全部通过注入测试；训练吞吐两轮 **+13.8% / +1.3%**，波动大，不能归因到单项 | **是**：`global_step` 变成「成功更新次数」；非有限 reflow loss 从致命退出改为跳过 |

**最值得要的是 C03 和 C04**：8 GB 显卡上冷加载少占约 395 MiB 显存，长时间预处理不再因缓存无界增长吃显存，而且数值完全不变。
**C01、C05 的速度收益目前没有可靠证据**，不要据此宣传「变快了」。

**需要你决定的事**：
1. 是否接入 Studio（建议先只接 C03 + C04 + C01，这三项输出逐位相同）。
2. C05 的「步数 = 成功更新次数」新语义是否接受，它会影响 Studio 的进度显示、`max_steps` 和断点续训的说明。
3. C02 是否改用「仅 loss + 隔离 RNG」做日常验证。这样训练轨迹会和旧版不同。

**这批没做到的**：完整 B0–B5 验收矩阵中的 2000 步训练曲线、batch 4、B1 的 3 次预热 + 10 次转换统计、盲听、长期语料稳定性。这个分支应视为「已做冒烟验证和部分性能取证的候选」，不是可以直接上线的结论。

---

## 1. 范围、测量条件与限制

- 基线由 `git archive b9d4e6a` 导出到单独目录，与优化分支交替测量。
- 硬件：Windows WDDM，RTX 4060 Ti 8 GB，驱动 617.42。
- 解释器：原有 DDSP-SVC 虚拟环境的 `python.exe`，均带 `-I -X utf8 -B`。**没有安装任何依赖**。
- 版本：torch `2.9.1+cu130`、torchaudio `2.9.1+cu130`、transformers `5.22.1`、torchfcpe `0.0.4`、numpy `1.26.4`、soundfile `0.14.0`。
- 随机性与精度：Python、NumPy、torch CPU/CUDA 种子固定为 42；CPU torch 8 线程；matmul TF32 关闭，cuDNN TF32 保持开启。
- 输入：
  - 音频 A = 一段 24 秒的测试歌曲（`test_song_24s.wav`）；
  - 模型 M = 一个自训练的 `model_16000.pt`，只读加载，架构为 10 层/1024 通道辅助 DDSP + 6 层/1024 通道 reflow；
  - ContentVec 用 Studio 的回退资产 `DDSP-SVC\pretrain\contentvec\pytorch_model.bin`。
- 所有数据、特征、输出、日志、脚本都放在本仓库之外的工作区，没有写进 DDSP-SVC 主检出或模型目录。
- GPU 互斥：每次 GPU 操作都先取锁，硬限时 1100 秒，用 trap 释放。测量时没有其他 Python 计算进程，但桌面图形进程仍在用 GPU，所以计时波动不能当成严格的独占硬件实验。
- 测试：17 个 CPU unittest 全部通过（4.522 s）。没有 pytest / coverage，改用标准库 tracer + AST 统计**新增语句**覆盖 186/232 = 80.17%。这不是全仓覆盖，也不是分支覆盖。

## 2. C01：DDSP 几何常量不再从 CUDA 读回 Python

**改了什么**（`ddsp/vocoder.py`）：
- 保留 `sampling_rate`、`block_size`、`win_length`、`window` 等原 state_dict 键，旧 checkpoint 照常加载。
- 新增只读的 Python 整数几何，供 arange、unfold、STFT/ISTFT 参数和 pad 分支使用。涉及 sampling_rate、block_size、f0 的浮点运算不变。
- 加载时一次性检查保存的几何与构造配置是否一致，不一致直接报错（`strict=False` 时也报）。

**为什么**：这些是配置常量。原实现把它们当 CUDA 标量 buffer 用于形状和 Python 分支，每次前向都触发 GPU→主机同步，在 Windows 和小 batch 下代价更明显。

**实测**（真实 M，A 派生的冻结 batch，batch 2，172 帧）：
- fp32 DDSP 输出、hidden 逐位相同；122 个参数梯度逐位相同；整模型 202 个 state tensor 的形状、dtype、SHA256 全部相同。
- 单次前向 profiler：`aten::item` 18 → 1，`aten::_local_scalar_dense` 18 → 1。
- DDSP 前向 + 反向时延（每轮 5 次预热、50 次计时，不含 optimizer）：

| 轮次 | 基线中位数 | 优化中位数 | 基线 p95 | 优化 p95 |
|---|---:|---:|---:|---:|
| 1 | 32.714 ms | 35.183 ms | 42.839 ms | 41.011 ms |
| 2 | 34.028 ms | 27.546 ms | 45.152 ms | 31.422 ms |

第一轮慢约 7.5%，第二轮快约 19.0%。**同步次数下降已证实，时延收益没有稳健结论。**

**风险与回退**：
- 原来能悄悄容忍的「checkpoint 与 config 几何不一致」现在会明确失败，这是有意收紧。
- 不要在构造后手动改几何 buffer。
- 回退：撤销该提交，不涉及权重迁移。

证据：`runtime\ddsp-opt\evidence\{baseline,optimized}-stage-{1,2}.json`、`*-stage-1-ddsp.pt`、`*-stage-1-ddsp-trace.json`

## 3. C02：日常 loss 验证与合成预览、完整审计分开

**改了什么**（`reflow/solver.py`、`configs/reflow.yaml`）：
- `validation_mode`：
  - `loss`：全验证集只算 loss；
  - `preview`：全集 loss，外加验证集开头 N 个 clip 的合成、音频和图；
  - `full_audit`：全部 clip 合成，即旧行为。
- 独立的 `interval_preview`、`interval_audit`，保存频率不变。
- 显式 `validation_precision`，不再默默跟随训练 AMP。
- `validation_rng: legacy | isolated`。
- 预览指标写到 `validation_preview/*`，避免和完整审计的 `validation/*` 历史曲线混淆。
- 非预览 clip 不再重读原始音频；预览用的原始音频直接给 CPU tensor，不再上传 GPU。

**兼容默认**：`full_audit` + `fp32` + `legacy`。不改配置，行为就和旧版一样。

**实测**（A 派生的两个 4 秒验证 clip，真实 M，50 步 Euler，fp32，每种模式 3 次）：

| 模式 | 三次耗时 | 中位数 |
|---|---|---:|
| 旧验证 | 1.028 / 0.568 / 0.579 s | 0.579 s |
| 新完整审计 | 1.000 / 0.646 / 0.560 s | 0.646 s |
| 新仅 loss | 0.0355 / 0.0465 / 0.0388 s | 0.0388 s |
| 新 1-clip 预览 | 0.2967 / 0.2866 / 0.2999 s | 0.2967 s |

- 计时**不含** matplotlib 绘图和 TensorBoard 写盘，只是验证计算、音频读取和 CPU 快照的耗时，不等于完整生产验证的端到端耗时。
- 仅 loss 模式比旧验证约 −93.3%，1-clip 预览约 −48.8%；完整审计没有变快（这轮中位数还略慢）。
- 受控 RNG 下，保留的首个预览的 mel、原始音频、合成音频与旧验证逐位相同。`isolated` 模式验证后各 RNG 状态都恢复。
- `legacy` 下的 loss / preview 模式少消耗随机数，之后的训练 RNG 序列会改变。两种模式的 loss 数值不同，原因是随机 loss 采样顺序变了，**不代表训练质量变好或变差**。

**风险与回退**：
- 用 isolated RNG 或减少合成调用都会改变旧的训练轨迹，不保证与旧训练逐步同权重。
- 很长的单个验证 clip 仍可能吃内存，这批没有加预览裁窗。
- 历史 SNR / PSNR / SI-SNR 公式故意保留，便于和旧审计比较。
- 回退：配置改回 `full_audit` / `fp32` / `legacy`，两个新 interval 设 0；或撤销提交。

证据：`evidence\baseline-validation.json`、`optimized-validation-fixed.json`（**以这个为准**，早期 `optimized-validation.json` 的设置因 DotDict 复制问题没写回）、`*-previews.pt`、`comparisons.json`、`c02-metric-label-green.log`

## 4. C03：checkpoint 先在 CPU 加载，降低冷加载显存峰值

**改了什么**（`logger/checkpoint.py`、`reflow/vocoder.py`、`nsf_hifigan/models.py`、`ddsp/vocoder.py`、`logger/utils.py`）：
- 加载顺序改为：checkpoint 加载到 CPU → CPU 上的模型 strict 加载 → 删除 snapshot → 整体一次性搬到目标设备。
- ContentVec / Hubert 本地 encoder checkpoint 和训练续训 checkpoint 同样改为 CPU 加载。
- 新增显式反序列化策略：
  - `trusted`（默认）：显式 `weights_only=False`，兼容旧的本地 checkpoint，只用于可信文件；
  - `weights_only`：严格只读张量，遇到不支持的对象直接失败，不做不安全回退。
- HiFiGAN 仍然延迟加载；去 weight norm 仍在目标设备上执行，避免改变数值。

**实测**（真实 M 冷加载）：

| 指标 | 基线 | 优化 |
|---|---:|---:|
| `max_memory_allocated` | 827,521,024 B | 413,760,512 B |
| `max_memory_reserved` | 849,346,560 B | 436,207,616 B |
| 加载结束 live allocated | 413,760,512 B | 413,760,512 B |

- allocated 峰值减少约 394.6 MiB（50%），reserved 减少约 394 MiB。
- 冷加载时间两轮：基线 1.008 / 0.968 s，优化 1.196 / 0.936 s。首轮慢 18.7%，第二轮略快。**这一项的收益是显存峰值，不是速度。**
- 202 个 state tensor 哈希全部相同，Studio 等价转换波形与基线逐位相同。

**风险与回退**：
- `weights_only` 可能拒绝含自定义对象的老模型，要调用方主动选择。
- CPU 内存峰值没有单独测；ContentVec 与 HiFiGAN 的单独显存峰值也没有分开统计。
- 其他 loader 会导入这里新增的 helper；单独撤销时要连 import 一起撤。

证据：`evidence\*-stage-{1,2}-state-hashes.json`、`*-convert.json`、`c03-green.log`

## 5. C04：Hann 窗口缓存按真实几何设上限

**改了什么**（`nsf_hifigan/nvSTFT.py`）：
- 缓存键从「随机浮点 keyshift 的字符串」改为「(实际窗口长度, 规范化设备, 默认 dtype)」。
- 默认 64 项 LRU，可通过构造参数调整。
- 不量化 keyshift；FFT 长度、缩放、resize 都不变；仍在 CPU 构造 periodic Hann 再搬到目标设备。

**为什么**：预处理的音高增强每个 clip 都抽一个连续随机 keyshift，旧缓存几乎每次都新增一项，GPU 上的窗口永不释放，语料越大占用越多。

**实测**：
- 7 组确定性 shift（含 `1.0` 与 `1.00001`）：Hann 窗口与原构造逐位相同，mel 7/7 逐位相同；取整后长度相同的不同浮点 shift 复用同一个窗口。
- 10,001 次 CUDA `get_mel` 调用：

| 指标 | 基线 | 优化 |
|---|---:|---:|
| 缓存条目 | 10,008 | 64 |
| 驻留 Hann tensor | 82,845,104 B | 533,064 B |
| 测量段 live allocated 增量 | 85,391,360 B | 495,616 B |
| 合成调用总耗时 | 23.489 s | 16.275 s |

驻留减少约 99.4%。耗时只测了一轮短音频压力测试，**约 30.7% 的耗时下降不能外推成真实预处理吞吐提升**。

**风险与回退**：LRU 会让部分窗口重建，用一点计算换固定内存上限。没做 10 万次压力测试和完整六种预处理特征的 A/B。回退：撤销该提交。

证据：`evidence\*-cache.json`、`*-cache-mel.pt`、`c04-green.log`

## 6. C05：只给 fp16 用 GradScaler，并正确统计真实更新

**改了什么**（`reflow/solver.py`）：
- bf16 / fp32 直接 backward，经有限梯度检查后再 step；只有 fp16 用 GradScaler。
- 梯度检查覆盖 Inf / NaN，用分组 infinity norm，避免有限大值做 L2 求和时溢出。
- fp16 通过 scale 是否更新判断是否跳过，不依赖 `optimizer.step()` 的返回值。
- scheduler、成功计数和 saver 的 `global_step` 只在真实更新后推进；attempted / skipped / successful 分开记录。
- autocast 按设备类型取值，兼容 `cuda:0`；组件 loss 和总 loss 都做有限性保护。
- 新增带版本的 `OptimizationStep.state_dict()`，但原 Saver 和 Studio 的 ManagedSaver **还没有接上它的持久化**。

**训练基准 B3**：
- 数据：一次性小数据集，train 取 A 的 0–4 / 4–8 / 8–12 / 12–16 秒，val 取 16–20 / 20–24 秒。
- 设置：真实 M 初始化，bf16，batch 2，2 秒 crop。
- 计时：每轮 50 次预热 + 300 次计时，关闭验证、保存、日志，基线与优化交替跑两轮。

| 轮次 | 基线 updates/s | 优化 updates/s | 基线步中位数 | 优化步中位数 |
|---|---:|---:|---:|---:|
| 1 | 4.543 | 5.171 | 219.100 ms | 191.339 ms |
| 2 | 4.934 | 5.001 | 199.825 ms | 198.047 ms |

- 提升分别为 13.8% 和 1.3%，波动很大。两侧每轮都是 350 次成功更新、0 次跳过。
- 显存峰值约 allocated 1.911 GB、reserved 2.164 GB，前后基本相同。
- 这是 C01–C05 合起来的分支对比，不能把收益归给 C05；也没做 2000 步训练曲线和质量验证。

**冻结更新与噪声底线**：

| 比较 | 梯度最大绝对差 | 梯度相对 L2 | 参数最大绝对差 | 参数相对 L2 |
|---|---:|---:|---:|---:|
| bf16 基线 → 优化 | 0.00229735 | 0.00434580 | 0.00456029 | 0.000264353 |
| bf16 基线自身重复 | 0.00150912 | 0.00432842 | 0.00528293 | 0.000264885 |
| fp16 基线 → 优化（scale 128） | 0.0000687651 | 0.000345807 | 0.00245313 | 0.000115836 |
| fp16 基线自身重复（scale 128） | 0.0000610352 | 0.000345670 | 0.00317094 | 0.000116053 |

AMP 下梯度和参数本来就不逐位相同；优化前后的差异与同一版本自身重复的噪声同量级。这只是数值冒烟检查，不能证明长期训练质量等价。

**非有限更新与计数**：
- fp16 初始 scale 65536 在真实冻结 batch 上自然溢出：两侧都跳过更新，优化侧不推进 LR。
- 用 scale 128 完成一次真实更新后注入 Inf；bf16 也在成功更新后注入 Inf。两种精度结果相同：参数、LR、scheduler 都不变，attempted=2、successful=1、skipped=1；fp16 的 scale 从 128 降到 64。
- CPU 上双子 ChainedOptimizer 在 fp32 / bf16 / fp16 下遇到 NaN 都不会只更新一部分；3 次尝试 1 次跳过后，`global_step` = 2、scheduler step = 2。

**风险与回退**：
- 旧 checkpoint 的 `global_step` 仍能读，但新语义是「成功更新次数」，旧的跳过次数无法还原。
- attempted / skipped 只是本次运行内的计数，不跨续训持久化。
- 原 Saver 和 Studio 都没保存 scheduler、scaler、RNG、采样器的完整状态，不能承诺位级续训。
- 原来非有限 reflow loss 会致命退出，现在改为记录并跳过。如果所有 batch 都持续非有限，成功步数不会前进；Studio 的停止按钮仍有效，但之后应加「连续跳过上限」。
- 有限梯度扫描本身有开销，可能抵消去掉 scaler 的收益。
- 回退：撤销该提交。回退后不要把新旧两种步数语义混在同一条训练曲线上。

证据：`evidence\{baseline,optimized}-train-{1,2}.json`、`comparisons.json`、`repeat-noise.json`、`*-finite-frozen-step.pt`、`optimized-{bf16,fp16}-finite.json`、`c05-control-green.log`

## 7. Studio 等价集成检查

- 做法：直接调用 Studio 现有 `workers\svc_worker.py` 的加载与转换函数，DDSP 根目录分别指向基线副本和优化 worktree。没有启动或提交任务到 8765，也没改 Studio 的代码和配置。
- 参数：FCPE，key / formant / register shift 0，阈值 −60 dB，speaker 1，50 步 Euler，t_start = 0，15 秒窗口 + 0.1 秒重叠，实际 1 段、2 窗。
- 结果：输出 1,058,400 样本，44100 Hz，24.000 秒，RMS 0.07259597，非静音。两轮基线与优化的浮点波形逐位相同，各自重复也逐位相同。
- 计时不可比：首次转换基线 8.321 s、优化 4.378 s，但基线包含 FCPE 提取，优化复用了同一份 f0 缓存。warm 数据每侧只有一次，不作为收益。

证据：`evidence\{baseline,optimized}-convert.log`、`*-convert-audio-{1,2}.npy`、`optimized-convert-converted-{0,1}.wav`

## 8. 与 MERT 分支的冲突评估

- **C01**：与 MERT 同在 `ddsp/vocoder.py`，但改的是 CombSubSuperFast，不碰 MERT encoder 构造、Audio2MERT 和归一化。
- **C03**：改了 encoder 的旧加载位置和 reflow loader，与 MERT 的插入和 DotDict 修正多为不同 hunk。合并时保留 MERT 已有的 CPU 加载和 24000/320 几何约束，不要重复引入 helper。
- **C02 / C05**：solver 和配置新字段与 MERT 基本独立；配置里要保留 MERT 的注释。
- **C04**：`nvSTFT.py` 独立，没有已知冲突。
- 这批没改 DotDict，不含 C17。合并后要另外验证 MERT 的几何和逐样本归一化，ContentVec 模型的测试结果不能代表 MERT 模型。

## 9. 下一批建议顺序

1. **C08**：F0 提取器的设备与生命周期。全局 FCPE / RMVPE 缓存在 unload 后仍会留着，先解决显存归属，保留 warm 复用。
2. **C07**：units 对齐改 `index_select`。小范围省内存，同时修 batch > 1 的对齐问题，是安全批量预处理的前提。
3. **C06 + C18**：合并有限性同步、复用标量日志；增加结构化的 attempted / skipped / successful 事件、连续跳过上限，以及完整的 optimizer / scheduler / scaler / RNG 续训合同。不要替换 Studio 已有的原子保存和停止机制。
4. **C17**：路径与 DotDict 兼容。先复用 MERT 已有修正，再在构造前解析资产路径。
5. **C19**：指标和数据鲁棒性。修正指标要带版本标识；验证集为空、过长、过短，以及 t_start、设备参数要尽早报错。
6. **C09**：Windows 数据加载矩阵。实测 CPU 缓存 workers 0/1/2 和 streaming 0/2/4，不要盲目加 worker。
7. **C11**：推理时不变的 conditioning 投影只算一次。保留训练梯度，先测实际调用次数和净显存。
8. **C10 / C20 / C21 放到后面**：减少采样步数、推理 AMP、`torch.compile` 都会影响音质，要先由你做 A/B 批准，不能因为这批透明优化成功就默认开启。

## 10. 如何接入 Studio（由你决定，未代为操作）

1. 保持 `DDSP-SVC-opt` 与 MERT 主检出互不干扰。
2. 先用验证脚本的「根目录 + 绝对资产路径覆盖」方式复测，不要直接改正式配置（`<workspace>` 是存放验证脚本和证据的工作区，不在本仓库）：
   ```text
   <DDSP-SVC venv>\Scripts\python.exe -I -X utf8 -B <workspace>\ddsp-opt\scripts\verify_batch1.py --root <本 worktree> --output-dir <workspace>\ddsp-opt --operation convert --tag user-recheck
   ```
   运行前请确认没有其他 GPU 任务（Studio 的任务也一样）。
3. 注意 worktree 里**没有** `pretrain/` 等原始资产。只把 Studio 的 `svc.root` 指过来、不处理旧的相对资产路径，加载可能失败。
4. 正式切换后，先做一次 ContentVec 模型转换和训练的开始 / 停止 / 最大步数 / 续训冒烟。
5. C02 先用兼容默认；愿意改变验证 RNG 和训练轨迹时，再启用 `loss` + `isolated`，并把 preview / audit 频率调低。
6. C05 的「成功更新」语义要和 Studio 的进度显示、`max_steps`、续训说明一起确认。
7. 回退：把 Studio 的 root 指回原检出即可，不要移动或覆盖模型。

## 证据索引

证据都在本仓库之外的工作区 `<workspace>\ddsp-opt\` 下：

- 总索引：`REPORT.md`
- 输入指纹与原始证据清单：`evidence\provenance.json`（A、M、ContentVec、HiFiGAN 的 SHA256，以及 164 个原始证据文件，约 7.15 GiB）
- 对比汇总：`evidence\comparisons.json`、`evidence\repeat-noise.json`
- 测试：`evidence\final-unittest.log`、`evidence\cpu-changed-statement-coverage.json`
- 脚本：`scripts\{verify_batch1.py, gpu_run.sh, compare_evidence.py, cpu_coverage.py}`
- 一次性数据与冻结 batch：`dataset\{config.yaml, train, val, frozen_batch.pt}`

失败阶段的日志都保留了，便于复核测试先行的过程。`prepare-a-dataset` 第一次失败是因为验证脚本用 `/` 作字典键，而 Windows loader 返回 `\`；只修了 runtime 下的夹具。
