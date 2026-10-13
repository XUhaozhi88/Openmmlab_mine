#!/bin/bash
# Compatibility test script for OpenMMLab on PyTorch>=2.8 + CUDA>=12.8

set -e

echo "=== System Information ==="
uname -a
echo ""

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo ""

echo "=== CUDA Toolkit Version ==="
nvcc --version || echo "nvcc not found in PATH"
echo ""

echo "=== Python Version ==="
python3 --version
which python3
echo ""

echo "=== PyTorch Installation Check ==="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1 || echo "PyTorch not installed or import failed"
echo ""

echo "=== Check mmengine ==="
python3 -c "import mmengine; print(f'mmengine version: {mmengine.__version__}')" 2>&1 || echo "mmengine not installed"
echo ""

echo "=== Check mmcv ==="
python3 -c "import mmcv; print(f'mmcv version: {mmcv.__version__}'); from mmcv.ops import get_compiling_cuda_version, get_compiler_version; print(f'MMCV CUDA: {get_compiling_cuda_version()}'); print(f'MMCV Compiler: {get_compiler_version()}')" 2>&1 || echo "mmcv not installed or ops unavailable"
echo ""

echo "=== Check mmdet ==="
python3 -c "import mmdet; print(f'mmdet version: {mmdet.__version__}')" 2>&1 || echo "mmdet not installed"
echo ""

echo "=== CUDA Compute Capability Check ==="
python3 << 'EOF' 2>&1 || echo "Failed to check compute capability"
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {cap[0]}.{cap[1]}")
    arch_list = torch.cuda.get_arch_list()
    print(f"PyTorch built for architectures: {arch_list}")
    if f"sm_{cap[0]}{cap[1]}" in arch_list or f"compute_{cap[0]}{cap[1]}" in arch_list:
        print("✓ GPU architecture is supported by PyTorch")
    else:
        print("✗ WARNING: GPU architecture may not be fully supported")
else:
    print("CUDA not available")
EOF
echo ""

echo "=== Test mmcv ops (basic NMS) ==="
python3 << 'EOF' 2>&1 || echo "mmcv ops test failed"
import torch
from mmcv.ops import nms
boxes = torch.tensor([[0., 0., 10., 10.],
                      [1., 1., 11., 11.],
                      [20., 20., 30., 30.]], device='cuda')
scores = torch.tensor([0.9, 0.8, 0.7], device='cuda')
inds = nms(boxes, scores, iou_threshold=0.5)
print(f"NMS test passed. Output indices: {inds}")
EOF
echo ""

echo "=== Test mmdet model import ==="
python3 << 'EOF' 2>&1 || echo "mmdet model import failed"
from mmdet.models import build_detector
from mmengine import Config
print("✓ mmdet model imports successful")
EOF
echo ""

echo "=== All checks completed ==="
