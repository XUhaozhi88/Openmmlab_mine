# 本次实验代码同步说明

本目录保存 2026-10-13 实际 DINO 实验使用的兼容修复和确定性模块。
来源仓库：https://github.com/XUhaozhi88/Openmmlab_mine
固定提交：efcd71be2e87e51578431bcdf43ef43f565d5504。
服务器已验证：PyTorch 2.8.0+cu128、Python 3.13.5、MMEngine 0.11.0rc3、MMCV 2.2.0、MMDetection 3.3.0，两张 RTX 5090。

## 已同步的修改

仓库内 mmengine/setup.py、mmcv/setup.py、mmdetection/setup.py 的 get_version() 使用显式 namespace 执行版本文件。
这修复 Python 3.13 下 exec() 与函数 locals() 的作用域问题，仅涉及安装元数据读取。
保留原本的 Windows CRLF 换行；统一为 LF 后，文件 SHA256 与服务器已审计的部署文件完全一致。
原始文件逐字节备份位于 original_before_update/，完整差异见 python313_metadata.patch。

deterministic_attention.py 是实际实验采用的确定性 MSDA 反向计算模块，原文哈希与服务器记录一致。
它保留 MMCV 原 CUDA forward，运行时将 MSDA backward 替换为逐 level 的双线性 gather 重算，避免原 CUDA backward 的原子累加差异。
模块适用于一阶训练；不支持高阶梯度。导入本模块不会自动改变算子，需显式调用 install()。

## 在自己的训练入口启用

在启动 Python 前，PowerShell 中设置：

```powershell
$env:PYTHONHASHSEED='20261008'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONPATH='E:\Files\Codes\Python\openmmlab'
```

在训练入口构建模型和开始训练前执行：

```python
from reproducibility_20261013.deterministic_attention import configure, install

configure(20261008)
install()
```

configure() 还会设置 Python、NumPy、Torch/CUDA 随机种子，启用严格确定性算子，关闭 TF32、cuDNN benchmark 和非确定性高效注意力，并固定 CPU 线程数。
仅复制这些代码不会自动复现全部训练结果；还需使用相同数据、模型配置、初始权重、采样顺序、精度与软件环境。

## 编译与核验

未复制服务器的 Linux .so、虚拟环境、数据或权重。Windows 使用时，MMCV CUDA 算子需按原仓库 README 在本机匹配的 PyTorch/CUDA 环境中编译。
本次仅回传并核验源码，没有在此 Windows 机器上安装依赖或执行 GPU 训练。
environment.json 和 source_verified.json 是服务器的核验记录，其中绝对路径指向服务器。
local_return_manifest.json 记录本地备份与修改后文件的校验值。
