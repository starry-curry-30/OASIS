# OASIS 中文运行指南

[返回仓库首页](../README.md)

OASIS 围绕床、桌子、沙发等稳定语义锚点组织场景记忆，用语言模型筛选候选位置，再用视觉语言模型检查图像。机器人导航时还会记录近期观察，用于重新寻找被移动的物体。

建议先完成 **离线数据建图 → 检查地图和锚点文件 → 语言检索**，再接 ROS 和机器人。下面的命令都在 OASIS 仓库根目录执行。

## 1. 安装环境

当前本地环境使用 Python 3.12、PyTorch 2.9.0、torchvision 0.24.0。以下 CUDA 12.8 命令要求 NVIDIA 驱动支持相应版本；其他驱动请参考 [PyTorch 官方安装命令](https://pytorch.org/get-started/previous-versions/#v290)。

```bash
git clone https://github.com/starry-curry-30/OASIS.git
cd OASIS
conda create -n oasis python=3.12 tk pip -y
conda activate oasis
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
mkdir -p 3rdparty
git clone https://github.com/apple/ml-mobileclip.git 3rdparty/mobileclip
git -C 3rdparty/mobileclip checkout 1140b8d197e4ed7d56b3a92216ded98bb1c2ac87
python -m pip install --no-deps -e 3rdparty/mobileclip
```

仓库目前为私有，克隆时需要已获授权的 GitHub 账号。MobileCLIP 使用与本地原项目一致的源码版本；`--no-deps` 用于保留前面安装的 PyTorch/torchvision 配对。

## 2. 放置模型权重

按 [README 的权重来源](../README.md#4-prepare-model-weights) 下载，放入：

```text
model/
├── yolov8l-world.pt
├── mobile_sam.pt
└── FastSAM-s.pt
```

路径在 `config/system_config.yaml` 中配置。默认 CLIP 编码器是 `MobileCLIP-S2`，预训练参数为 `datacompdr`，第一次加载需要下载权重或提前准备缓存。

检查安装和配置：

```bash
python -c "import torch, open3d, faiss, open_clip, mobileclip; print('CUDA available:', torch.cuda.is_available())"
python -m applications.runner_dataset --cfg job --resolve
```

第二条只打印配置，不加载数据和模型。正式 GPU 建图前应确认第一条输出 `CUDA available: True`。

## 3. 准备 Replica 数据

需要 RGB、深度图和相机位姿，不能只放场景 mesh。可参考 [NICE-SLAM 的 Replica 数据说明](https://github.com/cvg/nice-slam#replica)。目录格式：

```text
dataset/Replica/room0/
├── traj.txt
└── results/
    ├── frame000000.jpg
    ├── depth000000.png
    ├── frame000001.jpg
    └── depth000001.png
```

`traj.txt` 每行是一帧展平后的 4×4 camera-to-world 矩阵，图像和位姿需要逐帧对应。`scene_id: room_0` 会读取 `room0` 目录。相机内参和深度缩放在 `config/data_config/dataset/replica.yaml` 中；自采数据必须对应真实标定和深度单位。

## 4. 配置路径并建图

先修改 `config/base_config.yaml` 中对应的字段：

```yaml
dataset_name: replica
scene_id: room_0
dataset_path: ./dataset/Replica
dataset_conf_path: config/data_config/dataset/replica.yaml
representative_frames_path: ./output/representative_frames/replica_room_0
output_path: ./output/map_results
run_oasis_ablation: false
```

如果数据在其他硬盘，将 `dataset_path` 改为绝对路径。关闭消融模式后，普通建图不需要另行准备评测真值。

```bash
python -m applications.runner_dataset \
  use_rerun=false \
  use_parallel=false \
  run_oasis_ablation=false \
  keyframe_image_save.output_dir=output/keyframes/replica_room_0
```

首次检查可追加 `end=180 stride=18`，只处理少量帧；完整场景运行时去掉 `end=180`。短序列可能还不能建立稳定锚点，不适合直接评价检索效果。确认顺序模式正常后可使用 `use_parallel=true`。

输出包括：

- 地图：`output/map_results/replica_room_0/map/`。
- 锚点图像与清单：`output/representative_frames/replica_room_0/anchor_key/`。
- 关键帧：`output/keyframes/replica_room_0/`。
- 耗时统计：`output/map_results/replica_room_0/` 下的 CSV。

导出器会替换配置目录中的旧地图或关键帧。要保留不同实验，请分别设置地图、代表帧和关键帧输出目录。

## 5. 用自然语言检索

先确认锚点目录中有 `manifest.json` 和 `anchor_attached_objects.txt`。检索脚本也读取 `config/base_config.yaml`，其中的场景、地图和代表帧路径必须与建图时一致。

```bash
export OPENAI_API_KEY="<你的文本服务 API Key>"
export GEMINI_API_KEY="<你的 Gemini API Key>"
python scripts/anchor_task_runner.py \
  "Find the sealed silver can on a round table" \
  --anchor-file output/representative_frames/replica_room_0/anchor_key/anchor_attached_objects.txt \
  --manifest output/representative_frames/replica_room_0/anchor_key/manifest.json
```

文本模型用于锚点筛选，Gemini 用于视觉检查。可通过 `--model`、`--vlm-model` 指定账号能访问的模型，通过 `OPENAI_BASE_URL` 配置兼容文本接口。API 调用可能产生服务商费用；密钥通过环境变量传入，不要提交到仓库。程序不会自动加载 `.env`。

结果保存在 `output/query_results/`，包括 `predictions.json`。批量任务一行一个：

```bash
python scripts/anchor_task_runner.py --tasks-file scripts/batch_tasks.txt
```

脚本现在默认读取本仓库。若以前设置过 `OASIS_ROOT` 或兼容旧项目的 `DUALMAP_ROOT`，请确认它们没有指向其他工程。

## 6. ROS、导航与评测

离线建图不需要安装 ROS。实时输入需要安装对应 ROS 及消息包，配置 RGB、深度、位姿、camera-info、内参和外参。ROS 的 Python ABI 必须匹配：例如 Ubuntu 22.04 的 ROS 2 Humble 二进制包使用 Python 3.10，应单独建立匹配环境并重复安装依赖，不能直接混用上面的 Python 3.12 环境。

首次实时建图应关闭默认的地图预加载：

```bash
source /opt/ros/humble/setup.bash
python -m applications.runner_ros \
  ros_stream_config_path=config/data_config/ros/self_collected.yaml \
  preload_global_map=false \
  preload_layout=false \
  use_rerun=false
```

导航还需要真实机器人或仿真器、运动控制器和正确的 ROS 话题。用已有地图导航时，先设置 `preload_path`、`representative_frames_path`，再启用相应预加载选项。详细说明见 [README 的 ROS 部分](../README.md#ros-and-live-inputs)。

评测脚本需要对应数据集、标注和已生成地图；先查看实际参数：

```bash
python scripts/evaluate_anchor_sr3d.py --help
```

当前已检查语法、入口导入、命令行帮助和 Hydra 配置；尚未验证从零安装环境及论文的完整建图、导航和评测流程。
