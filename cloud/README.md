# 阿里云 PAI DLC 训练

## 官方 GR00T N1.7 训练入口

使用单节点四张同型号的完整 A100 或 A800 GPU。DLC worker 安装固定版本的 NVIDIA Isaac-GR00T `708617f5716c63fc0a327391492ea2b91977ae90`，使用其 Python 3.12、PyTorch 2.9/CUDA 12.8 和 Transformers 4.57.3 依赖。原来的 CPU DSW 可用于校验 NAS 文件和提交 DLC。

在 `/mnt/workspace/robottt/assets` 准备官方 `GR00T-N1.7-3B` 的 `373d34e2723d928870157d07dc216a19a624b5bb` 快照到 `gr00t-n1.7-3b-373d34e`，并准备公开的 Qwen3-VL-2B-Instruct 处理器文件到 `qwen3-vl-2b-processor-8964489`。脚本核验官方权重的 SHA-256 及处理器文件与 Cosmos 相同的 Git blob 哈希。DLC 端以离线方式加载已验的模型文件。 [官方权重](https://huggingface.co/nvidia/GR00T-N1.7-3B/tree/373d34e2723d928870157d07dc216a19a624b5bb) · [公开处理器快照](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203)

在 CPU DSW 的源码目录执行以下准备步骤。下载支持断点恢复。`--endpoint` 可指定可达的镜像，所有文件仍由固定摘要校验。DLC 使用第二个脚本输出的绝对 `ROBOTTT_PYTHON` 路径。

```bash
python3 cloud/stage_gr00t_assets.py stage /mnt/workspace/robottt/assets
bash cloud/prepare_gr00t_runtime.sh
```

四卡验证作业使用单独的 `ROBOTTT_RUN_DIR`，取四条最长的真实示范，每卡一条。本地数据的四条序列为 169 至 192 次完整策略调用。每 8 次调用截断梯度，完成一个 BF16 优化器步，并保存验证用 checkpoint。确认四个 rank 都完成反向传播、损失和梯度有限，且 XPolicyLab 的闭环验证通过，再提交正式作业。

```bash
ROBOTTT_RUN_DIR=/mnt/workspace/robottt/runs/gr00t-n1.7-smoke bash cloud/run_gr00t_dlc.sh smoke
ROBOTTT_RUN_DIR=/mnt/workspace/robottt/runs/gr00t-n1.7-50k bash cloud/run_gr00t_dlc.sh train
```

正式作业读取 `/mnt/workspace/raw` 的 300 条示范，每项 50 条，先缓存冻结视觉语言模型的真实特征，再进入训练。四卡各累积 16 条轨迹，等效全局 batch 64。固定 50,000 个优化器步，每 10,000 步产出 `step_NNNNNN.pt`，`latest.pt` 指向最近一次完整保存。恢复时使用原 `ROBOTTT_RUN_DIR` 并设置 `ROBOTTT_RESUME` 指向命名 checkpoint。模型、数据、调度和 DDP 配置须和原次运行一致。

仓库的 [论文对照](../REPRODUCTION.md#公开权重在-insert-benchmark-的训练设置)列出上下文定义、稳定化及优化范围。DLC 日志中的 `vlm_cache_episode` 记录真实特征处理进度，`train` 记录损失、梯度、上下文长度、耗时和显存峰值，`checkpoint` 记录保存位置。

## 早期小模型训练入口

用同一份 NAS 的 `raw` 目录。程序直接读取 Insert Benchmark HDF5，不需要生成 LeRobot 副本。正式训练使用 300 条示范，六个任务各 50 条。benchmark 自带的测试种子用于独立闭环评估。

## 在 CPU DSW 做数据预检

把要运行的仓库版本解包到 NAS，例如 `/mnt/workspace/robottt/source/<commit>`。容器需要 Python 3.10+、CUDA 版 PyTorch 2.5+、NumPy 1.24+、h5py 3.8+。预检也可使用 CPU 版 PyTorch。

```bash
commit=替换为已推送的提交哈希
cd "/mnt/workspace/robottt/source/$commit"
bash cloud/run_dlc.sh check
```

`dataset_preflight` 应报告 300 条轨迹及每项 50 条。同步过程中如果只读到五项旧数据，脚本会退出。镜像的 Python 环境不同时，用 `ROBOTTT_PYTHON` 设置解释器路径。若镜像仅缺 h5py，可安装到单独目录并通过 `PYTHONPATH` 引用。

## DLC 单节点四卡启动

选择有四张同型号卡的 A100 或 A800 专用配额，Worker 数 1，GPU 4。异构资源配额需使用已核实 GPU 型号的指定节点。只使用已核实为 A100 或 A800 的资源，避免闲时跨配额借用带入其他 GPU。挂载同一个 NAS 到 `/mnt/workspace`。程序在启动四个训练进程之前另行检查 CUDA 实际可见卡的数量及型号。

在 DLC 的启动命令设置仓库和输出位置：

```bash
commit=替换为已推送的提交哈希
cd "/mnt/workspace/robottt/source/$commit"
export ROBOTTT_RUN_DIR=/mnt/workspace/robottt/runs/robottt-4gpu-50k
bash cloud/run_dlc.sh train
```

`ROBOTTT_NAS_ROOT` 可改为 DLC 的实际挂载点，例如 `/mnt/data`。脚本默认从该挂载点下的 `raw` 读取；如原始数据位于其他绝对路径，用 `ROBOTTT_DATA_ROOT`。输出目录须在该 NAS 下。`ROBOTTT_CACHE_DIR` 默认 `/tmp/robottt-rgb-cache`，用于工作容器内的图像特征缓存。

固定全局 batch 64、50,000 次优化器更新。四个数据并行进程默认各处理 16 条序列。可用 ROBOTTT_MICRO_BATCH 调低每卡微批次，启动脚本会保持全局 batch 不变。每 10,000 步保存命名 checkpoint，同时更新 `latest.pt` 供断点续训。重启使用同一版本的代码、数据和输出目录，并设置：

```bash
export ROBOTTT_RESUME="$ROBOTTT_RUN_DIR/latest.pt"
bash cloud/run_dlc.sh train
```

DLC 日志的 `configuration` 会打印模型和 batch 参数，`train` 带 optimizer step、loss 和吞吐，`checkpoint` 记录 NAS 保存路径。完成后可从输出目录取 `step_050000.pt` 评测。

六任务训练对内循环启用 Q/K/V 范数约束及快权重投影，内循环仍在 FP32 中计算。外层 AdamW 峰值学习率为 `1e-4`。第一次未加约束的四卡训练在约 1,250 步后出现非有限梯度。现在探测到异常时会记录失败步及受影响的参数，退出供检查和恢复。

[阿里云 DLC 任务创建与资源配置](https://help.aliyun.com/zh/pai/create-a-training-task)
