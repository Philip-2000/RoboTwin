# SceneRecon2 标注复现指南

这份文档用于让其他人在自己的 Python/Conda 环境中复现当前的
WorldArena / RoboTwin2.0 场景还原标注流程。当前流程的核心是：

1. 用 RoboTwin/SAPIEN 渲染一个可编辑的静态场景。
2. 用 task-specific initializer 给出初始物体模型、位置、朝向和相机。
3. 人工在浏览器里微调并保存 reconstruction state。
4. 后续再把保存出的场景接到机械臂动作 replay、物理仿真和视频渲染。

当前 SceneRecon2 的人工标注状态和 initializer 状态是分开保存的。人工保存
结果是最终标注进度的依据，initializer 只是起点。

## 0. 获取代码仓库

建议从 RoboTwin 原仓库 fork 到个人仓库，并保持仓库名不变：

```bash
git clone https://github.com/Philip-2000/RoboTwin.git
cd RoboTwin
```

如果机器上已经 clone 了 RoboTwin 原仓库，可以把 remote 切到个人 fork：

```bash
git remote set-url origin https://github.com/Philip-2000/RoboTwin.git
git remote -v
```

也可以保留官方仓库作为 upstream：

```bash
git remote add upstream https://github.com/RoboTwin-Platform/RoboTwin.git
```

请确认使用的是包含 `SceneRecon2/` 修改的分支。如果不是默认分支，额外执行：

```bash
git checkout <branch-with-scenerecon2>
```

## 1. 安装 RoboTwin 环境

推荐使用 Conda。下面假设 Miniconda 位于 `~/miniconda3`，环境名为
`RoboTwin`。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
```

安装 RoboTwin 基础依赖：

```bash
pip install -r script/requirements.txt
```

安装 PyTorch3D。若 GitHub 访问正常，可以直接：

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
```

若 GitHub 很慢，建议先在网络更好的机器上下载源码或 wheel，再拷贝到目标机器
安装。本仓库内也可能已经有 `pytorch3d/` 目录，可以根据本机 CUDA/PyTorch 版本
改用本地安装方式。

安装 Curobo。官方脚本默认从 GitHub clone：

```bash
cd envs
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
pip install warp-lang==1.12.0
pip install setuptools==69.5.1
cd ../..
```

如果目标机器访问 GitHub 很慢，可以在另一台机器下载 `NVlabs/curobo`
的 `v0.7.8` 分支，然后把目录拷贝为：

```text
<RoboTwin repo>/envs/curobo
```

再执行上面的 `pip install -e .`。

RoboTwin 的安装脚本还会 patch `sapien` 和 `mplib` 的若干兼容性问题。可以直接
执行：

```bash
bash script/_install.sh
```

如果已经手动安装了依赖，至少要确认 `script/_install.sh` 中对
`sapien/wrapper/urdf_loader.py` 和 `mplib/planner.py` 的 patch 已经生效。

运行 SceneRecon2 前，建议设置 CUDA/Vulkan 环境变量。下面是当前机器上用过的
配置，其他机器按实际 CUDA 路径调整：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export TORCH_CUDA_ARCH_LIST=12.0
```

## 2. 下载和配置 WorldArena 数据

当前代码默认使用如下目录结构：

```text
~/D/WorldArena_Robotwin2.0/
  test_dataset/
    first_frame/fixed_scene_task/episode1.png
    instructions/fixed_scene_task/episode1.json
    data/fixed_scene_task/episode1.hdf5
    ...
  val_dataset/
    first_frame/fixed_scene_task/episode1.png
    instructions/fixed_scene_task/episode1.json
    data/fixed_scene_task/episode1.hdf5
    ...
```

其中标注服务当前主要使用：

- `first_frame/fixed_scene_task/episodeN.png`：目标首帧图片。
- `instructions/fixed_scene_task/episodeN.json`：语言指令和辅助信息。
- `data/fixed_scene_task/episodeN.hdf5`：未来 replay 机械臂动作时会用到。

另外还需要 WorldArena 的 task top1 匹配文件。当前默认路径是：

```text
~/C/WorldArena/yl_outputs/search_gt/
  worldarena_test_to_robotwin_clean50_ollama_task_gttrace_top10.json
  worldarena_val_to_robotwin_clean50_ollama_task_gttrace_top10.json
```

代码入口在 `SceneRecon/task_mapping.py`：

```python
default_search_gt_path("test_dataset")
default_search_gt_path("val_dataset")
```

如果数据不放在上述位置，有两种做法：

1. 推荐复现时先保持同样的目录结构。
2. 或者修改 `SceneRecon/task_mapping.py` 和
   `SceneRecon2/precompute_initializers.py` 中的默认路径。

放好数据后可以先做两个检查：

```bash
ls ~/D/WorldArena_Robotwin2.0/test_dataset/first_frame/fixed_scene_task/episode1.png
ls ~/D/WorldArena_Robotwin2.0/test_dataset/instructions/fixed_scene_task/episode1.json
ls ~/C/WorldArena/yl_outputs/search_gt/worldarena_test_to_robotwin_clean50_ollama_task_gttrace_top10.json
```

## 3. 运行标注服务和执行标注

先离线预计算 initializer。默认会对当前支持的所有 task 执行初始化：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

python -m SceneRecon2.precompute_initializers --split test_dataset --overwrite
```

只跑部分 task 时：

```bash
python -m SceneRecon2.precompute_initializers \
  --split test_dataset \
  --tasks adjust_bottle press_stapler place_empty_cup \
  --overwrite
```

initializer 输出位置：

```text
SceneRecon2/outputs/place_empty_cup_editor/initializer_states/test_dataset/
```

人工标注输出位置：

```text
SceneRecon2/outputs/place_empty_cup_editor/states/test_dataset/
```

启动浏览器标注服务：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export TORCH_CUDA_ARCH_LIST=12.0

python -m SceneRecon2.place_empty_cup_editor --host 127.0.0.1 --port 8788
```

本机浏览器打开：

```text
http://127.0.0.1:8788
```

远程机器上运行服务时，建议用 SSH 端口转发：

```bash
ssh -L 8788:127.0.0.1:8788 <user>@<host>
```

然后在本地浏览器打开 `http://127.0.0.1:8788`。

常用页面：

```text
/progress                         人工保存进度
/progress_initializer             initializer 预计算进度
/task/<task>                      某个 task 的 episode 列表
/task/<task>/<episode_number>     指定 task 和 episode
/<episode_number>                 只知道数字时，自动跳转到对应 task
```

例如：

```text
http://127.0.0.1:8788/task/adjust_bottle/273
http://127.0.0.1:8788/890
```

如果页面打不开，先检查服务是否在线：

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8788/progress
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8788/progress_initializer
```

重启服务：

```bash
pkill -f 'SceneRecon2.place_empty_cup_editor' || true
python -m SceneRecon2.place_empty_cup_editor --host 127.0.0.1 --port 8788
```

### 标注操作

快捷键的权威说明在：

```text
SceneRecon2/docs/editor_controls.md
```

当前主要操作如下：

- `1`-`9`：按对象栏编号选择物体。
- `Tab`：切换当前可编辑物体。
- `C`：切换到相机编辑。
- 方向键或 `W/A/S/D`：移动当前物体；相机模式下移动相机。
- 按住 `Z`：细调。
- 按住 `X`：粗调。
- 不按 `Z/X`：正常步长。
- `Q` / `E`：旋转当前物体 yaw；相机模式下调整 pitch。
- `,` / `.`：切换当前物体的上一/下一候选模型。
- 按住 `Space`：右侧 current render 临时显示目标首帧，用于对比。
- 松开 `Space`：恢复当前渲染。
- `Enter`：保存人工标注。
- `+`：跳到当前 task 的下一个未完成人工标注 episode。
- `-`：跳到当前 task 的上一个未完成人工标注 episode。

人工保存后，状态文件会写入：

```text
SceneRecon2/outputs/place_empty_cup_editor/states/test_dataset/episodeN.<task>.json
```

页面上 `Current Render` 后面的状态含义：

- `saved` / 上一次人工保存的：当前加载自人工保存状态。
- `initializer` / Initializer 跑出来的：当前加载自离线 initializer。
- `dirty` / 修改中：当前状态已经被手动改过，还没有保存。

## 4. 初始化场景很糟糕时，如何让 Codex 优化

不要直接让 Codex 随手改某一个人工保存文件，除非这是明确的个例修补。更推荐让
Codex 优化 task-specific initializer，这样同类 episode 会一起受益。

一次好的指令应包含：

1. task 名。
2. 具体 episode 编号。
3. 目标首帧路径或页面 URL。
4. 当前错误现象：模型错、颜色错、朝向错、悬空、相机不对、物体缺失等。
5. 希望优先改规则，只有不可规则化时才加 episode override。
6. 要求重跑 affected task 的 initializer，并抽样检查 before/after。

可以直接套用这个模板：

```text
请优化 SceneRecon2 的 <task> 初始化。样例是 episode273、episode486，
页面是 http://127.0.0.1:8788/task/<task>/273。

现象：目标图里应该是雪碧，但 initializer 选成了可乐；另外瓶子的 yaw
和参考图差约 90 度。请检查：
1. first_frame 图片；
2. instructions JSON；
3. envs/<task>.py 中 RoboTwin 原始随机逻辑；
4. SceneRecon2/initializers/simple_tasks.py；
5. SceneRecon2/initializers/model_selector.py；
6. initializer_states 和 debug 图。

优先写 task-specific 规则，不要直接修改人工 states。改完后重跑这个 task
的 initializer，给我几个 episode 的对照结果和需要人工复核的编号。
```

常见优化方向：

- 模型选择：在 `SceneRecon2/initializers/model_selector.py` 或 task initializer
  中加入颜色、形状、文字图案、候选模型 ID 规则。
- 位置初始化：从检测框或颜色分割结果反投影到桌面 `x/y`，再写入对象
  `table_xy`。
- 朝向初始化：根据细长物体的主方向、瓶身方向、订书机头尾方向等设置
  `yaw_deg` 或 `qpos`。
- 高度问题：检查该 task 的桌面高度、物体 `z`、模型本身坐标原点以及是否误把
  站立/躺倒姿态混用。
- 分类错误：修改 top1 mapping 源文件，或对明确分错的 episode 加映射修正。

优化完成后，常用重跑命令：

```bash
python -m SceneRecon2.precompute_initializers \
  --split test_dataset \
  --tasks <task> \
  --overwrite
```

然后刷新：

```text
http://127.0.0.1:8788/progress_initializer
http://127.0.0.1:8788/task/<task>
```

## 5. 未来：利用还原场景和机械臂动作做仿真渲染

这一段目前是待定流程，不是稳定入口。目标是把人工保存出的场景状态和
WorldArena 的 HDF5 机械臂动作结合起来，重新执行物理仿真并渲染视频。

计划中的输入：

```text
SceneRecon2/outputs/place_empty_cup_editor/states/test_dataset/episodeN.<task>.json
~/D/WorldArena_Robotwin2.0/test_dataset/data/fixed_scene_task/episodeN.hdf5
```

计划中的过程：

1. 读取人工 reconstruction state，创建 RoboTwin/SAPIEN 场景。
2. 根据 state 放置物体模型、桌面、相机和必要的 task-specific 元素。
3. 读取 HDF5 中的双臂关节动作序列。通常是按顺序存储的 `T x (7+7)`
   关节信号，文件本身主要提供顺序，不直接保存高层动作段语义。
4. 用正确的机器人初始姿态和动作顺序逐帧驱动机械臂。
5. 以 24 FPS 渲染视频。
6. 将渲染首帧、过程视频和 WorldArena 原始 first frame / reference video 做对比。

相关探索脚本曾经包括：

```text
script/render_joint_video.py
script/reconstruct_traj_from_hdf5.py
script/replay_robotwin_dataset.py
```

但 SceneRecon2 标注服务当前是静态、无物理的摆放编辑器。完整 replay/export
链路应在未来单独固化，避免和人工标注服务混在一起。

## 维护约定

- 修改快捷键时，同步更新 `SceneRecon2/docs/editor_controls.md`。
- 修改复现流程时，同步更新本文档。
- initializer 自动结果放在 `initializer_states/`。
- 人工保存结果放在 `states/`，人工保存结果才算 progress finished。
- 尽量把 task-specific 策略放在 `SceneRecon2/initializers/` 下，不要污染
  RoboTwin 原始 `envs/` 文件，除非确实是在修 RoboTwin 本身的问题。
