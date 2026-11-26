# RoboTTT for XPolicyLab

The adapter runs the compact RoboTTT action head on the Isaac Sim Insert Benchmark. The supported XPolicyLab environment is `marvin_dual_gripper` with `action_type=joint`, three RGB cameras, and 16 absolute joint/gripper targets at 30 Hz. Model weights load on startup. Each environment carries its own fast weights, which reset at the start of each episode and advance once per returned action chunk.

## Install

From the RoboTTT repository, use an environment that already has a PyTorch build for the target accelerator.

```bash
bash setup_xpolicylab.sh /path/to/XPolicyLab
cd /path/to/XPolicyLab/policy/RoboTTT
```

The installer copies this directory into XPolicyLab and adds the Marvin entry to the two official robot registries if absent. Use `--no-install` with a separately prepared Python environment and put the RoboTTT repository on that environment's `PYTHONPATH` when using this option.

## Data processing and training

These six tasks use the benchmark's original HDF5 trajectories, preserving synchronized RGB, joint positions, and same-step action targets. The inputs are raw RGB arrays, and the main trainer prepares patch tokens and computes train-only normalization. The trajectories are consumed directly, so a LeRobot conversion step is unnecessary.

```bash
export ROBOTTT_DATA=/mnt/workspace/raw
bash process_data.sh InsertBenchmark cotrain marvin_dual_gripper joint 50
bash train.sh InsertBenchmark cotrain marvin_dual_gripper joint 0 0,1,2,3
```

`process_data.sh` validates the expected 50 demonstrations in each of the six tasks. The training script calls the same `robottt.ddp_train` module used by DLC, with global batch size 64 and 50,000 optimizer steps by default. Its conventional run directory is `checkpoints/InsertBenchmark-cotrain-marvin_dual_gripper-joint-0/`. Evaluation loads `latest.pt` for a directory and also accepts an absolute path to a specific checkpoint. Set `checkpoint_path` in `deploy.yml` for the shared resolver's explicit-path option.

## Evaluate

The final two arguments accept conda environment names, paths that contain `bin/python`, or `current` to use Python on the shell's PATH. Use a benchmark task name such as `timed_single` or `recover_three` for the task-conditioned training checkpoint.

```bash
EVAL_ENV_TYPE=debug DEBUG_EVAL_EPISODES=2 bash eval.sh InsertBenchmark timed_single \
  /absolute/path/to/insert_timed_single.pt marvin_dual_gripper joint 0 "" 0 \
  /path/to/python-env /path/to/python-env

EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 DEBUG_EVAL_EPISODES=2 bash eval.sh \
  InsertBenchmark timed_single /absolute/path/to/insert_timed_single.pt \
  marvin_dual_gripper joint 0 "" 0 /path/to/python-env /path/to/python-env
```

For the actual Insert Benchmark simulator, run the standard XPolicyLab server and connect the benchmark's WebSocket evaluator. `INSERT_BENCHMARK_ROOT` enables simulator-specific arm joint limits from the benchmark assets. The adapter always bounds gripper openings to `[0, 1]`.

```bash
export INSERT_BENCHMARK_ROOT=/path/to/insert_benchmark
bash setup_eval_policy_server.sh InsertBenchmark recover_three /absolute/path/to/latest.pt \
  marvin_dual_gripper joint 0 0 /path/to/python-env 19001

cd /path/to/insert_benchmark
./benchmark.sh evaluate --level recover_three --seeds 24001 --policy websocket \
  --url ws://127.0.0.1:19001 --policy-id robottt
```

Set `reset_memory_each_chunk: true` in `deploy.yml` for the same-checkpoint history ablation. Evaluation uses the full returned chunk, matching the policy's non-overlapping chunk training. Plain RGB and encoded RGB debug paths use the upstream server's color decoder. `deploy.py` and `eval.sh` are based on the upstream XPolicyLab demo adapter and covered by [the XPolicyLab license](XPolicyLab-LICENSE).

## Native GR00T N1.7 backbone

Train the two-stage native policy through the XPolicyLab entry point on a single eligible four-card host:

```bash
ROBOTTT_BACKEND=gr00t_n1_7 ROBOTTT_RECIPE=paper_two_stage ROBOTTT_TTT_VARIANT=reference \
  bash train.sh InsertBenchmark paper50k marvin_dual_gripper joint 1745 0,1,2,3
```

Select `backend: gr00t_n1_7` in `deploy.yml` or set `ROBOTTT_BACKEND=gr00t_n1_7`. Use an environment with the [official Isaac-GR00T N1.7 dependencies](https://github.com/NVIDIA/Isaac-GR00T/tree/708617f5716c63fc0a327391492ea2b91977ae90). The official base snapshot and a local Cosmos-compatible processor directory are required. For the public snapshot, the VLM is Cosmos-Reason2-2B and the action head has 32 DiT blocks. The adapter keeps the official VLM and action head and adds the 32 RoboTTT memories. It forwards the three RGB cameras and the benchmark's original language instruction through the native processor. Native `decode_action` restores the 16 absolute joint and gripper commands from normalized model output.

For a trained native checkpoint:

```bash
export ROBOTTT_BACKEND=gr00t_n1_7
export ROBOTTT_GR00T_BASE=/absolute/path/to/nvidia-gr00t-n1.7
export ROBOTTT_GR00T_PROCESSOR=/absolute/path/to/cosmos-compatible-processor
export INSERT_BENCHMARK_ROOT=/absolute/path/to/insert_benchmark

EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 DEBUG_EVAL_EPISODES=1 bash eval.sh \
  InsertBenchmark recover_three /absolute/path/to/native-run/latest.pt \
  marvin_dual_gripper joint 0 0 0 /path/to/python-env /path/to/python-env
```

A trained `latest.pt` records the native backend, optimizer step, adapter configuration, learned tensors and Marvin normalization statistics. Stage-one checkpoints restore TTT, register tokens and gates. Legacy adaptation checkpoints may also include Marvin projectors. A `model_stage=posttrain` checkpoint restores the entire trained VLM and action head as well as TTT. Optimizer shards are used for training resume. The deployment service reads the single weight file and uses its saved execution stride. The non-overlapping training recipe predicts and executes 40 actions per policy call, then passes the updated fast weights to the next observation.

To test integration with newly initialized TTT parameters, set `ROBOTTT_ALLOW_UNTRAINED_TTT=1`, set `ROBOTTT_GR00T_PROCESSOR`, and pass the official base snapshot directory as `ckpt_name`. The backend can compute Marvin normalization statistics from the 300 local benchmark demos or load the same statistics from `ROBOTTT_GR00T_STATISTICS`. This option does not load trained RoboTTT weights.

To run the actual Isaac simulator, use the same `setup_eval_policy_server.sh` and `benchmark.sh evaluate` commands as above, with the native backend environment variables and native `ckpt_name`. This releases a finite action chunk to a simulator, advances the per-environment fast state once per chunk, and resets it between episodes. Benchmark success is measured separately in the recorded report.
