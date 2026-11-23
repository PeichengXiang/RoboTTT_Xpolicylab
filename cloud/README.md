# 阿里云 PAI DLC 训练

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

固定全局 batch 64、50,000 次优化器更新。四个数据并行进程默认各处理 16 条序列。可用 ROBOTTT_MICRO_BATCH 调低每卡微批次，启动脚本会保持全局 batch 不变。每 500 步更新 `latest.pt`，每 5,000 步保存命名 checkpoint。重启使用同一版本的代码、数据和输出目录，并设置：

```bash
export ROBOTTT_RESUME="$ROBOTTT_RUN_DIR/latest.pt"
bash cloud/run_dlc.sh train
```

DLC 日志的 `configuration` 会打印模型和 batch 参数，`train` 带 optimizer step、loss 和吞吐，`checkpoint` 记录 NAS 保存路径。完成后可从输出目录取 `step_050000.pt` 评测。

[阿里云 DLC 任务创建与资源配置](https://help.aliyun.com/zh/pai/create-a-training-task)
