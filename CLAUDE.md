# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

This directory is a single git repo that vendors three OpenMMLab projects. They were originally separate checkouts; their `.git` dirs were removed and backed up to `../openmmlab_git_backup/`. Together they form a dependency stack:

```
mmengine/      (0.11.0rc3)  training engine: Runner, Registry, Config, hooks, optim, dist, fileio
  ↑
mmcv/          (2.2.0)      CV primitives: image/video IO, transforms, cnn building blocks, compiled C++/CUDA ops
  ↑
mmdetection/   (3.3.0)      detection/segmentation/tracking toolbox built on both (package name `mmdet`)
```

The upstream base commits are listed in README.md §3. mmengine also carries 2 local changes on top of upstream: `setup.py`/`package_utils.py` switched from `pkg_resources` to `importlib.metadata`, and the lite package was renamed to `mmengine_lite`. Each subdir keeps its own `.gitignore`, and mmcv's ignores `data/`. Upstream-tracked files under `tests/data/` were force-added, so new test data there needs `git add -f`.

**Target environment:** these checkouts have been locally patched to run on PyTorch ≥ 2.8 + CUDA ≥ 12.8 (needed for Blackwell GPUs, sm_100/sm_120). The patches (see README.md §3):
- `mmdet/__init__.py` and `requirements/mminstall.txt` now accept `mmcv < 2.3.0`, so mmcv 2.2.0 imports.
- mmcv CUDA sources include `ATen/cuda/Atomic.cuh` instead of the deprecated THC header.
- Deprecated `torch.cuda.amp.*` / `torch.get_autocast_gpu_dtype` calls are replaced.
- Removed numpy aliases (`np.float`, `np.long`) are replaced.

Keep new code compatible with PyTorch 2.6+: `torch.load` defaults to `weights_only=True` there, so loading full OpenMMLab checkpoints (which carry a non-tensor `meta`) must pass `weights_only=False`. Also use `torch.autocast(device_type='cuda', ...)` and `torch.amp.GradScaler` rather than the `torch.cuda.amp` equivalents.

## Install (editable, in dependency order)

```bash
pip install -e mmengine            # MMENGINE_LITE=1 builds the slimmer `mmengine_lite`
MMCV_WITH_OPS=1 pip install -e mmcv -v   # MMCV_WITH_OPS=0 → pure-python `mmcv-lite`, no compiled ops
pip install -e mmdetection
```

For Blackwell, build mmcv against a CUDA 12.8 toolkit that matches torch's CUDA version. Set `TORCH_CUDA_ARCH_LIST="12.0"` (RTX 50 series) or `"10.0"` (B200), or `"10.0;12.0"`, especially when building on a machine without the target GPU. Use `pip install -e mmcv -v --no-build-isolation` so the build sees the installed torch.

mmcv build env vars (see `mmcv/setup.py`): `FORCE_CUDA=1` builds CUDA ops when no GPU is visible, `MMCV_CUDA_ARGS` adds extra nvcc flags, and `FORCE_MLU` / `FORCE_NPU` select other backends. On macOS without CUDA you get CPU (and experimental MPS) ops only.

## Tests

Each repo uses pytest from its own root:

```bash
cd mmengine && pytest tests/test_config/test_config.py::TestConfig::test_fromfile
cd mmcv && pytest tests/test_ops/test_nms.py
cd mmdetection && pytest tests/test_models/test_detectors/test_single_stage.py -k retinanet
```

- mmdetection's `pytest.ini` adds `--xdoctest`, so docstring examples in `mmdet/` run as tests when you collect those modules.
- Tests that need CUDA, or a specific device, are skipped when that hardware isn't available.

## Lint / style

All three repos use the same pre-commit stack: flake8, isort (line length 79), yapf (pep8-based), double-quote→single-quote fixer, docformatter, codespell, mdformat, and the OpenMMLab copyright-header check. mmengine and mmcv also run mypy.

```bash
cd <repo> && pre-commit run --all-files     # or: pre-commit run --files path/to/file.py
```

New `.py` files need the `# Copyright (c) OpenMMLab. All rights reserved.` header. Use single-quoted strings.

## Architecture (big picture)

**Registry + Config drive everything.** Components are built from config dicts like `dict(type='RetinaNet', ...)` through registries (`mmengine/registry`). Each downstream library defines child registries whose parent is mmengine's root registry. For example, `mmdet/registry.py` creates `MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mmdet.models'])`. As a result:
- A new class must be decorated with `@MODELS.register_module()` (or the matching registry), **and** its module must be importable from the `locations` package, which usually means exporting it from that package's `__init__.py`.
- Scope matters. `default_scope='mmdet'` in a config makes lookups resolve in mmdet's registries first and then fall back to the parent.

**Config system** (`mmengine/config`): Python config files inherit through `_base_ = [...]`, and child keys merge into parent dicts (`_delete_=True` replaces a dict instead). mmdetection has two config styles:
- `mmdetection/configs/<algo>/*.py`: the classic `_base_` string-inheritance style, built from `configs/_base_/{models,datasets,schedules,default_runtime.py}`. This is the primary set.
- `mmdetection/mmdet/configs/`: the newer "pure Python" style using `read_base()` and imported class objects (`type=RetinaNet`). Only a subset of algorithms exist here.

**Runner** (`mmengine/runner`): `Runner.from_cfg(cfg)` builds the model, data loaders, optimizer wrapper, param schedulers, evaluator, hooks, and loops (`EpochBasedTrainLoop`/`IterBasedTrainLoop`, `ValLoop`, `TestLoop`). Training behavior such as checkpointing, logging, and EMA is added through hooks rather than by editing the loop. `mmdetection/tools/train.py` and `tools/test.py` are thin wrappers around this.

**Data flow in mmdet:** the dataset (`mmdet/datasets`) runs a transform pipeline (`mmdet/datasets/transforms`, built on `mmcv.transforms.BaseTransform`) that ends in `PackDetInputs`. This produces `inputs` plus `DetDataSample`s (`mmdet/structures`, built on mmengine's `BaseDataElement`/`InstanceData`). Next, the model's `data_preprocessor` (normalization, padding, batch augmentation) runs on device. Then `model(inputs, data_samples, mode='loss'|'predict'|'tensor')` runs. Detectors in `mmdet/models/detectors` combine backbone → neck → heads (`dense_heads` for one-stage/DETR-style models, `roi_heads` for two-stage). Assigners, samplers, and coders live in `models/task_modules` under the `TASK_UTILS` registry.

**mmcv ops:** the Python wrappers in `mmcv/mmcv/ops/*.py` call `ext_loader` into a single compiled extension. The C++ entry points are in `mmcv/ops/csrc/pytorch/<op>.cpp`, which dispatch by device via `pytorch_device_registry.hpp` to `pytorch/{cpu,cuda,mps,mlu,npu,musa}/` implementations. Shared kernels live in `csrc/common/`. A new op must be added to `pybind.cpp` and given a backend implementation. See `mmcv/mmcv/ops/csrc/README.md` for the full guide.

**mmdetection extras:** `projects/` holds self-contained research projects that live outside the core package and are registered via `custom_imports`. `tools/dataset_converters` and `tools/analysis_tools` hold standalone scripts. Model zoo metadata is in each `configs/<algo>/metafile.yml` and `README.md`, and the pre-commit `check-algo-readme` hook validates those READMEs.

## Common mmdetection commands

```bash
cd mmdetection
python tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py [--work-dir DIR] [--cfg-options key=val ...] [--amp] [--resume]
python tools/test.py CONFIG CHECKPOINT [--show-dir DIR]
bash tools/dist_train.sh CONFIG NUM_GPUS
python tools/misc/print_config.py CONFIG      # view fully-merged config
python tools/analysis_tools/browse_dataset.py CONFIG   # visualize the pipeline output
```

Datasets are expected under `mmdetection/data/` (for example `data/coco/`), as set by `data_root` in `configs/_base_/datasets/*.py`.
