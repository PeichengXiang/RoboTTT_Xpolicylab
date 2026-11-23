# RoboTTT XPolicyLab

本项目实现 RoboTTT 的快权重动作头、序列 flow matching、TBPTT 及仅作为上下文的轨迹片段。附带小型闭环任务、NPZ 训练输入、光模块插拔仿真演示转换器，以及接入公开 GR00T N1.7 动作头的接口。论文中未给出的配置及公开模型与论文骨干的差异见 [论文对照与复现假设](REPRODUCTION.md)。

## XPolicyLab 与六任务训练

仓库包含 [XPolicyLab RoboTTT 适配器](policy/RoboTTT/README.md)，支持天机仿真的 `marvin_dual_gripper` 和 `joint` 控制。通过标准 `ModelTemplate` 和 WebSocket 加载训练 checkpoint，三路 RGB 与 16 维关节/夹爪状态由各环境独立推进快权重。

```bash
bash setup_xpolicylab.sh /path/to/XPolicyLab
ROBOTTT_DATA_ROOT=/path/to/benchmark_data bash cloud/run_dlc.sh check
```

[阿里云 DLC 训练入口](cloud/README.md)使用一个节点上的 4 张 A100 或 4 张 A800。有效全局 batch 为 64 条轨迹序列，每卡 micro-batch 4、累积 4 次，总计 50,000 个优化器步。训练读取 NAS 的 `/mnt/workspace/raw`，要求六个任务各有 50 条 HDF5 示范。官方测试种子独立于这 300 条训练轨迹。

这次训练使用本项目的小型 RoboTTT 动作头，保留 TTT-MLP、序列 flow matching 与 TBPTT。三相机图像用 RGB patch token，训练中逐步将上下文上限从 32 增至 192 个策略调用，每次预测 8 个 30 Hz 动作。任务文本用六个固定任务 ID 编码。论文中的 Eagle/GR00T 初始化与预训练数据仍不可用。

## 本机结果

| 验证 | 结果 |
| --- | --- |
| 合成双目标闭环，24 个固定测试种子 | 正确前缀 **23/24**，反转与去除线索均 **0/24**。同一 checkpoint 每次预测前重置快权重 **0/24**。单帧基线 **10/24**，对三种前缀的输出完全相同。 |
| 8192 个上下文更新 | 快权重张量总量始终 **8832 bytes**，单次 4 步去噪平均约 **3.93 → 3.86 ms**。对段长 128 的 TBPTT，1024 与 8192 步的 PyTorch CUDA 峰值分配均为 **40.16 MiB**。 |
| Isaac Sim 插入并拔出，`timed_single` seed 25001 | 使用 12 条示范训练、4 条独立示范做离线验证。TTT 和单帧基线分别完成 **480/480** 个控制步，接口均无报错，插拔得分均为 **0/1**。 |
| 公开 GR00T N1.7 | 以公开的 32 层配置和随机初始化参数通过 BF16 推理、快权重跨步传递。移除 register token 并关闭 TTT 门控后，四步去噪与原生动作头一致。官方权重索引的 537 个头部键与当前实现匹配。已验证小型本地 safetensors 装载，公开大权重尚未运行。 |

本机工作区保留了早期实验的合成与插拔 checkpoint 及指标。Git 仓库保存源码与文档。8192 步的测量使用小模型和随机输入，反映状态存储与训练显存的变化。YAM 真机结果和论文的闭环上下文缩放曲线仍待原版实验数据、权重与设备。

## 运行本机实验

从本目录执行。需要 Python 3.10+、PyTorch 2.5+ 和 NumPy。支持 CPU 和 CUDA。本机已用 RTX 5060 Ti、Python 3.11、PyTorch 2.7.0+cu128 执行。已有 PyTorch 的环境可直接运行命令。新环境可用 `python -m pip install -e .` 安装。

```bash
python -m robottt train --epochs 30 --output runs/robottt.pt
python -m robottt eval runs/robottt.pt --metrics runs/robottt_eval.json
python -m robottt eval runs/robottt.pt --reset-memory
python -m robottt train --no-ttt --epochs 30 --output runs/single_step.pt
python -m robottt eval runs/single_step.pt
python -m robottt stress --steps 8192
python -m robottt stress --steps 8192 --backward
```

- 闭环任务先给出两段只写快权重的条件帧，随后当前观测只给阶段与位置，目标的左右顺序取决于前缀。控制输出执行后改变下一步的本体位置。匹配、反转和去除左右线索的前缀使用相同的初始状态与去噪种子。
- `--reset-memory` 在每次动作预测前清除状态，使用原有 checkpoint。`--no-ttt` 从头训练同尺寸的当前帧模型，保留 registers。`--shared-time` 可对照序列内共享噪声等级。
- 压力测试向同一状态输入 8192 个合成上下文，检查输出有限与状态大小，并测量短、长历史下单次动作预测耗时。模型未在这段合成序列上训练。
- `--device cpu` 强制使用 CPU。训练保存模型权重、模型配置和验证日志。默认小模型设置适用于本地验证，论文的训练规模在 [REPRODUCTION.md](REPRODUCTION.md) 中列出。

## 自有数据

每条轨迹保存为一个 `.npz`。训练与验证按完整轨迹分目录。动作应与本体状态行对齐，单位或归一化在创建数据集时统一，部署时相应地反归一化。

| 字段 | 形状和含义 |
| --- | --- |
| `obs` | float32 `[F,V,C]`。每帧 V 个视觉或视觉语言 token，每个 C 维。也可提供 uint8 `rgb[F,views,height,width,3]`，由本地网格均值编码器转 token。 |
| `state` | float32 `[F,P]`，P 个本体状态通道。 |
| `actions` | float32 `[F,A]` 或 `[T,H,A]`。前者按 H 组成完整非重叠块，当前观测取块首，尾端不足 H 帧丢弃。后者直接读取对齐后的块，并要求 `obs` 和 `state` 都有 T 行。 |
| `loss_mask` | 可选 `[F]` 或 `[T]` 布尔掩码，缺省为全真。假值行仍参与快权重更新。对稠密帧动作，只有块内全部帧为真时该块才提供监督，接管边界需要精确时建议直接提供 `[T,H,A]`。 |

人视频和同任务机器人示范应按时间拼接在同一文件中。将人视频动作和缺失的本体状态填零，前缀 `loss_mask` 设为假，机器人片段设为真；`--context` 应容纳配对序列。对 DAgger，保留实际执行的机器人动作，只将可监督的人类接管段标真。批次右侧填充由独立的 `valid_mask` 禁用写入。前缀若超过 TBPTT 段边界，快权重值仍会传递，来自后段的梯度会在段边界停止。

```bash
python -m robottt train --data /path/to/train --validation-data /path/to/validation \
  --horizon 4 --context 128 --segment 16 --output runs/custom.pt
python -m robottt eval runs/custom.pt --data /path/to/validation --context 128
```

### 光模块插拔数据

转换器适配本机 insert_benchmark 的 HDF5 记录。读取三个相机、16 维 qpos 和 16 维同帧绝对关节/夹爪目标，按不同 seed 做训练和验证切分。网格 RGB 特征是本地动作头输入，通道值保持仿真记录的原单位。转换需要 `h5py`。

```bash
python -m robottt.isaac_export /home/dm/1121/insert_benchmark/benchmark_data/timed_single \
  runs/insert_data --train 12 --validation 4 --grid 4
python -m robottt train --data runs/insert_data/train --validation-data runs/insert_data/validation \
  --horizon 4 --context 64 --segment 16 --batch-size 4 --epochs 40 --output runs/insert.pt
python -m robottt eval runs/insert.pt --data runs/insert_data/validation
```

### Isaac Sim 闭环入口

在允许导入 XPolicyLab WebSocket 依赖的 PyTorch 环境中启动服务。使用导出时的 RGB 网格大小。

```bash
python -m robottt.isaac_serve runs/insert.pt \
  --benchmark /home/dm/1121/insert_benchmark --grid 4 --port 19001
```

在 insert_benchmark 目录运行，`--output` 可指向单独的结果目录。

```bash
./benchmark.sh evaluate --level timed_single --seeds 25001 --policy websocket \
  --url ws://127.0.0.1:19001 --policy-id robottt --output /tmp/robottt-isaac-eval
```

服务只绑定本机。仿真器逐帧发送观测，每次取动作块时才写入状态。使用正常的整块执行语义，关闭 receding-horizon 裁切。每个 trial 重新初始化快权重，动作范围按 Marvin 资产的关节限位与夹爪开合范围约束。

## 公开 GR00T N1.7 动作头

使用 NVIDIA [公开代码](https://github.com/NVIDIA/Isaac-GR00T/tree/708617f5716c63fc0a327391492ea2b91977ae90) 的 `Gr00tN1d7ActionHead`，保留其原生编码器、各层 attention、FFN 与动作解码器，在 attention 后插入 TTT。下例可用已安装的 NVIDIA 环境和已有基础模型，通过 `Gr00tRoboTTT(gr00t.action_head)` 接入，也可从本地 [官方 safetensors 快照](https://huggingface.co/nvidia/GR00T-N1.7-3B) 单独加载 head。后者可使用事先缓存的特征，无需在该进程中运行视觉语言骨干。

```python
from robottt.gr00t import Gr00tRoboTTT
from robottt.policy import corrupt_actions

# embodiment_id 是上游处理器使用的整数 ID。
policy = Gr00tRoboTTT.from_local_checkpoint(snapshot_path, embodiment_id=embodiment_id)
policy.set_stage("pretrain")
tau, noised, target = corrupt_actions(actions)
velocity, carry = policy(
    features, proprio, noised, tau,
    state=carry,  # 每条新轨迹传 None
    valid=valid_timesteps,
    vl_mask=vl_mask,
    image_mask=image_mask,
    action_mask=action_mask,
)
# 根据实际动作通道及上下文掩码计算 loss，再做 backward。
carry = carry.detach()  # TBPTT 段边界
```

`features[B,T,V,D]` 来自上游 backbone 输出的 `backbone_features`；另外传入 `backbone_attention_mask` 和 `image_mask`。本体状态、action horizon、动作通道和补位掩码以相应 GR00T processor 配置为准。推理用 `policy.eval().sample`，单次调用移除上述输入的 T 维；返回动作块和下一状态。`set_stage("pretrain")` 冻结原动作头，`set_stage("posttrain")` 解冻原动作头。上面的缓存特征用法仅微调动作头，全 VLM 后训练需要额外接入 VLM 训练。

公开 GR00T checkpoint 的配置为 Cosmos/Qwen3-VL、32 层 DiT，论文配置为 Eagle、16 层 DiT。接入接口按实际动作头的层数插入 TTT。[公开权重配置](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/main/config.json)
