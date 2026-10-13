# OpenMMLab（MMEngine + MMCV + MMDetection）Blackwell 适配版

本目录包含三个 OpenMMLab 仓库的本地源码。它们已经过修改，可以在 **PyTorch ≥ 2.8 + CUDA ≥ 12.8** 环境中运行，这一环境是 Blackwell 架构 GPU（RTX 50 系列 / B200 / GB200 等）的最低要求。

| 目录 | 版本 | 作用 | 是否需要编译 |
| --- | --- | --- | --- |
| `mmengine/` | 0.11.0rc3 | 训练引擎（Runner、Config、Registry、Hook 等） | 否，纯 Python |
| `mmcv/` | 2.2.0 | 视觉基础库，带 C++/CUDA 自定义算子 | **是**，必须本地编译 |
| `mmdetection/` | 3.3.0 | 目标检测 / 实例分割 / 跟踪工具箱 | 否，纯 Python |

依赖关系：`mmdetection` → `mmcv` → `mmengine`，安装时必须**按 mmengine → mmcv → mmdetection 的顺序**。

> ⚠️ 三个库都要**从本目录源码安装**，不要用 `pip install mmcv` / `mim install mmcv` / `pip install mmdet` 安装官方发行版：
> - 官方 mmcv 预编译包最高只支持到 torch 2.4 / CUDA 12.x，没有 Blackwell 的算子代码；
> - 官方 mmdet 3.3.0 限制了 `mmcv<2.2.0`，与 mmcv 2.2.0 一起使用时会在 `import mmdet` 时报错；
> - 下文的兼容性修改只存在于本地源码中。

---

## 1. 环境要求

| 项目 | 要求 |
| --- | --- |
| GPU | NVIDIA Blackwell（计算能力 10.0 / 12.0），也兼容更早的架构 |
| NVIDIA 驱动 | 支持 CUDA 12.8 的驱动（Linux ≥ 570，Windows ≥ 572） |
| CUDA Toolkit（nvcc） | **≥ 12.8**，并且与 PyTorch 自带的 CUDA 版本一致 |
| PyTorch | **≥ 2.8**（cu128 或更高版本的 wheel） |
| Python | 3.9 – 3.12（以 PyTorch 官方支持的版本为准） |
| 编译器 | Linux：GCC 9 – 13；Windows：Visual Studio 2022（含"使用 C++ 的桌面开发"） |

> 注意：`nvidia-smi` 右上角显示的 "CUDA Version" 是**驱动最高支持的 CUDA 版本**，并不代表已经安装了 CUDA Toolkit。编译 mmcv 需要 `nvcc`，请用 `nvcc -V` 确认。

---

## 2. 安装步骤

下面以 Linux 为例，Windows 的差异见[第 2.6 节](#26-windows-额外说明)。

### 2.1 创建 Python 环境

```bash
conda create -n openmmlab python=3.10 -y
conda activate openmmlab
```

### 2.2 安装 PyTorch（cu128）

```bash
pip install torch==2.8.* torchvision --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_capability())"
```

预期输出类似 `2.8.0+cu128 12.8 True` 和 `(12, 0)`（RTX 50 系列）或 `(10, 0)`（B200）。

如果使用更新的 PyTorch（2.9+）或 CUDA（cu129 / cu130），把版本号和 `--index-url` 换成对应值即可，注意事项见[第 5 节](#5-使用更新的-pytorch--cuda)。

### 2.3 安装 MMEngine

```bash
cd /path/to/openmmlab        # 本 README 所在目录
pip install -e ./mmengine
```

### 2.4 编译安装 MMCV（最关键的一步）

#### 2.4.1 准备编译依赖

```bash
pip install -U ninja psutil packaging wheel "setuptools<81"
```

> 为什么要用 `setuptools<81`：`mmcv/setup.py` 仍然使用 `pkg_resources`，而较新的 setuptools 已弃用并计划移除该模块，可能导致 `ModuleNotFoundError: No module named 'pkg_resources'`。

#### 2.4.2 确认 CUDA Toolkit

```bash
nvcc -V                     # 必须 ≥ 12.8，且主版本与 torch.version.cuda 一致
echo $CUDA_HOME             # 未设置时手动指定，例如：
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 2.4.3 指定目标 GPU 架构

```bash
# RTX 5090 / 5080 / 5070 等消费级 Blackwell
export TORCH_CUDA_ARCH_LIST="12.0"
# B200 / GB200 等数据中心 Blackwell
# export TORCH_CUDA_ARCH_LIST="10.0"
# 同时支持两类，或再加上 Ada(8.9) / Hopper(9.0)
# export TORCH_CUDA_ARCH_LIST="8.9;9.0;10.0;12.0"
```

- 架构列表写得越多，编译越慢。只写自己的卡最快。
- 不设置时，PyTorch 会按当前机器上检测到的 GPU 编译；在没有 GPU 的机器上编译（如 Docker build）时**必须**设置，并同时设置 `FORCE_CUDA=1`。

#### 2.4.4 编译并安装

```bash
cd mmcv
pip install -r requirements/optional.txt     # 可选
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e . -v --no-build-isolation
cd ..
```

- **`--no-build-isolation` 必须加上**，否则 pip 会在隔离环境里另装一个 torch，导致编译出的算子和实际使用的 torch 不匹配。
- 编译耗时通常为 10–40 分钟，取决于 CPU 核数和架构列表。默认并行数是 `CPU 核数 - 1`。内存不足时可以用 `export MAX_JOBS=4` 降低并行度。
- 加上 `-v` 才能看到编译日志，出错时把日志里第一个 `error:` 附近的内容拿来排查。

#### 2.4.5 验证 MMCV

```bash
python -c "import mmcv, mmcv.ops; print(mmcv.__version__)"
python .dev_scripts/check_installation.py 2>/dev/null || python - <<'EOF'
import torch
from mmcv.ops import nms, roi_align
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32, device='cuda')
scores = torch.tensor([0.9, 0.8], device='cuda')
print('nms ok:', nms(boxes, scores, 0.5)[1])
feat = torch.rand(1, 3, 16, 16, device='cuda')
rois = torch.tensor([[0, 0, 0, 8, 8]], dtype=torch.float32, device='cuda')
print('roi_align ok:', roi_align(feat, rois, (2, 2)).shape)
EOF
```

正常时会输出 `2.2.0`，并且 CUDA 算子运行不报错。如果出现 `no kernel image is available for execution on the device`，说明 `TORCH_CUDA_ARCH_LIST` 里没有包含当前 GPU 的架构，需要重新编译（见[第 4 节](#4-常见问题)）。

### 2.5 安装 MMDetection

```bash
pip install -e ./mmdetection
# 可选依赖
# pip install -r mmdetection/requirements/optional.txt
# pip install -r mmdetection/requirements/multimodal.txt   # GLIP / Grounding DINO 等
# pip install -r mmdetection/requirements/tracking.txt     # 多目标跟踪
```

验证：

```bash
python -c "import mmengine, mmcv, mmdet; print(mmengine.__version__, mmcv.__version__, mmdet.__version__)"
```

预期输出：`0.11.0rc3 2.2.0 3.3.0`

推理示例：

```bash
cd mmdetection
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest ./checkpoints   # 需要 pip install openmim
python demo/image_demo.py demo/demo.jpg configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py \
    --weights checkpoints/rtmdet_tiny_8xb32-300e_coco_*.pth --device cuda
# 结果默认保存在 outputs/vis/
```

### 2.6 Windows 额外说明

1. 安装 **Visual Studio 2022**，勾选"使用 C++ 的桌面开发"。
2. 安装与 PyTorch 对应的 **CUDA Toolkit 12.8+**，安装后重开终端。
3. 在 **"x64 Native Tools Command Prompt for VS 2022"** 中激活 conda 环境后编译 mmcv：

   ```bat
   set DISTUTILS_USE_SDK=1
   set MMCV_WITH_OPS=1
   set FORCE_CUDA=1
   set TORCH_CUDA_ARCH_LIST=12.0
   cd mmcv
   pip install -e . -v --no-build-isolation
   ```

4. 源码路径最好**不含中文和空格**，否则 nvcc 可能报路径错误。
5. Windows 不支持 NCCL，多卡分布式训练请在 Linux 下进行。

---

## 3. 相对官方版本的修改

各子目录基于以下上游 commit（合并到本仓库时去掉了原 git 历史，可以拿这些 commit 和官方仓库对比差异）：

| 目录 | 上游仓库 | 基准 commit |
| --- | --- | --- |
| `mmengine/` | https://github.com/open-mmlab/mmengine | `76bff21`（之后另有 2 个本地 commit，见下表最后一行） |
| `mmcv/` | https://github.com/open-mmlab/mmcv | `d2111bda` |
| `mmdetection/` | https://github.com/open-mmlab/mmdetection | `cfd5d3a9` |

### mmdetection

| 文件 | 修改 | 原因 |
| --- | --- | --- |
| `mmdet/__init__.py`、`requirements/mminstall.txt`、`requirements/readthedocs.txt` | mmcv 版本上限 `<2.2.0` → `<2.3.0` | 原上限导致 `import mmdet` 时报错 |
| `mmdet/models/backbones/csp_darknet.py`、`mmdet/models/layers/se_layer.py`、`mmdet/models/task_modules/assigners/sim_ota_assigner.py`、`projects/SparseInst/sparseinst/loss.py` | `torch.cuda.amp.autocast` → `torch.autocast(device_type='cuda')` | 旧接口自 PyTorch 2.4 起弃用 |
| `mmdet/datasets/transforms/text_transformers.py`、`mmdet/evaluation/functional/ytviseval.py` | `np.long` / `np.float` → `np.int64` / `np.float64` | numpy ≥ 1.24 / 2.x 已移除这些别名 |
| `tools/model_converters/{publish_model,upgrade_model_version,upgrade_ssd_version}.py` | `torch.load(..., weights_only=False)` | PyTorch ≥ 2.6 默认 `weights_only=True`，无法加载带 `meta` 的 checkpoint |

### mmcv

| 文件 | 修改 | 原因 |
| --- | --- | --- |
| `mmcv/ops/csrc/common/pytorch_cuda_helper.hpp`、`mmcv/ops/csrc/pytorch/cuda/ms_deform_attn_cuda.cu` | 优先引用 `ATen/cuda/Atomic.cuh`，找不到时才回退到 `THC/THCAtomics.cuh` | THC 头文件已被 PyTorch 弃用 |
| `mmcv/ops/pixel_group.py` | `np.float` → `np.float64` | numpy 2.x 兼容 |

### mmengine

| 文件 | 修改 | 原因 |
| --- | --- | --- |
| `mmengine/optim/optimizer/amp_optimizer_wrapper.py` | 优先使用 `torch.amp.GradScaler` | `torch.cuda.amp.GradScaler` 已弃用 |
| `mmengine/runner/amp.py` | 优先使用 `torch.get_autocast_dtype('cuda')` | `torch.get_autocast_gpu_dtype` 已弃用 |
| `tests/test_optim/test_optimizer/test_optimizer_wrapper.py` | 同步修改 GradScaler 的导入 | 测试兼容 |
| `setup.py`、`mmengine/utils/package_utils.py`（此前已提交到本地） | `pkg_resources` → `importlib.metadata`；lite 包改名为 `mmengine_lite` | 新版 setuptools / Python 3.12 兼容 |

所有修改都保留了对旧版 PyTorch 的兼容。

---

## 4. 常见问题

**Q1：`import mmdet` 报 `AssertionError: MMCV==2.2.0 is used but incompatible`**
说明装的是官方 mmdet，而不是本地源码。执行 `pip uninstall mmdet -y && pip install -e ./mmdetection`。

**Q2：`ModuleNotFoundError: No module named 'mmcv._ext'`**
mmcv 的算子没有编译成功，或者装成了 `mmcv-lite`。检查编译日志，确认设置了 `MMCV_WITH_OPS=1`，然后重新编译（见 Q4）。

**Q3：`RuntimeError: CUDA error: no kernel image is available for execution on the device`**
编译时没有包含当前 GPU 的架构。先用 `python -c "import torch; print(torch.cuda.get_device_capability())"` 查到架构，再把对应值写进 `TORCH_CUDA_ARCH_LIST`（如 `(12, 0)` → `"12.0"`），然后重新编译。

**Q4：怎样彻底重新编译 mmcv？**

```bash
cd mmcv
pip uninstall mmcv mmcv-lite -y
rm -rf build/ mmcv/_ext*.so mmcv/_ext*.pyd *.egg-info
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e . -v --no-build-isolation
```

**更换 PyTorch 版本之后必须重新编译 mmcv**，否则会出现 `undefined symbol` 或 `DLL load failed` 错误。

**Q5：`The detected CUDA version (x.x) mismatches the version that was used to compile PyTorch (12.8)`**
系统 nvcc 的版本和 torch 的 CUDA 版本不一致。用 `CUDA_HOME` 指向匹配版本的 CUDA Toolkit，或者重装对应 cu 版本的 torch。

**Q6：`No module named 'pkg_resources'`**
执行 `pip install "setuptools<81"` 后重新编译。

**Q7：编译过程被 kill，或出现 `c++: fatal error: Killed signal`**
内存不足。设置 `export MAX_JOBS=2`（或更小）后重新编译。

**Q8：加载 checkpoint 时报 `Weights only load failed`**
PyTorch ≥ 2.6 默认只允许加载纯权重。mmengine 的 `load_checkpoint` / `Runner` 已经处理了这个问题；如果在自己的脚本里直接调用 `torch.load`，请加上 `weights_only=False`（只对可信来源的文件这样做）。

**Q9：numpy 相关报错（`np.float` / `np.int` / `np.bool` has no attribute）**
本仓库已修复已知位置。如果第三方依赖（如旧版 `pycocotools`、`imagecorruptions`）报错，请升级这些依赖到支持 numpy 2 的版本，或安装 `numpy<2`。

---

## 5. 使用更新的 PyTorch / CUDA

- **PyTorch 2.9+**：代码中的版本判断都只有下限，没有上限，正常情况下不需要改代码，但必须**为新的 torch 重新编译 mmcv**。新版 torch 如果移除了更多旧接口，需要按报错逐处修复。
- **CUDA 13.0+**：
  - CUDA 13 不再支持 Maxwell / Pascal / Volta（计算能力 < 7.5），`TORCH_CUDA_ARCH_LIST` 中不要包含 `7.0` 及以下的值；
  - `mmcv/ops/csrc/pytorch/cuda/filtered_lrelu.cu` 使用了在 CUDA 13 中弃用的 `double4` / `longlong4` 类型，预计只会产生警告。
- 不论哪种升级，nvcc 版本都必须与 torch 的 CUDA 版本一致。

---

## 6. 开发与测试

```bash
# 单元测试（在各子仓库根目录下运行）
cd mmengine    && pytest tests/test_runner
cd mmcv        && pytest tests/test_ops/test_nms.py
cd mmdetection && pytest tests/test_models/test_detectors/test_single_stage.py

# 代码风格检查
pip install pre-commit
cd <子仓库> && pre-commit run --all-files
```

### 训练 / 测试示例

```bash
cd mmdetection
# 单卡训练（Blackwell 上推荐开启 AMP）
python tools/train.py configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py --amp
# 多卡训练
bash tools/dist_train.sh configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py 8 --amp
# 测试
python tools/test.py CONFIG CHECKPOINT
```

数据集默认放在 `mmdetection/data/` 下（例如 `data/coco/`），具体路径见各数据集配置文件中的 `data_root`。

各库的原始文档见 `mmengine/README.md`、`mmcv/README.md` 和 `mmdetection/README.md`。
