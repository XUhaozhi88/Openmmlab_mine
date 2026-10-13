"""First-order deterministic MSDA backward; preserve MMCV CUDA forward.

Recompute one level at a time with bilinear gather. Strict PyTorch mode selects
deterministic gather backward, avoiding MMCV CUDA atomic gradient accumulation.
Higher-order gradients are deliberately unsupported. No model weights change.
"""
import os
import random
import numpy as np
import torch
from torch.autograd.function import once_differentiable
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

ORIGINAL_APPLY = MultiScaleDeformableAttnFunction.apply
COUNTERS = {'forward': 0, 'backward': 0}


def configure(seed):
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert os.environ.get('PYTHONHASHSEED') == str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)


def level_output(value, location, weight, h, w):
    b, _, heads, channels = value.shape
    queries, points = location.shape[1], location.shape[3]
    features = value.permute(0, 2, 1, 3).reshape(b*heads, h*w, channels)
    xy = location.permute(0, 2, 1, 3, 4).reshape(b*heads, queries*points, 2)
    x, y = xy[..., 0]*w-.5, xy[..., 1]*h-.5
    x0, y0 = x.floor(), y.floor()
    dx, dy = x-x0, y-y0
    sampled = None
    for xx, yy, coefficient in [
            (x0, y0, (1-dx)*(1-dy)), (x0+1, y0, dx*(1-dy)),
            (x0, y0+1, (1-dx)*dy), (x0+1, y0+1, dx*dy)]:
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        index = (yy.clamp(0, h-1).long()*w+xx.clamp(0, w-1).long())
        row_index = index+torch.arange(b*heads, device=index.device)[:, None]*(h*w)
        corner = features.reshape(b*heads*h*w, channels).index_select(0, row_index.reshape(-1)).reshape(b*heads, queries*points, channels)
        contribution = corner*(coefficient*valid)[..., None]
        sampled = contribution if sampled is None else sampled+contribution
    sampled = sampled.reshape(b, heads, queries, points, channels).permute(0, 2, 1, 3, 4)
    return (sampled*weight[..., None]).sum(dim=3).reshape(b, queries, heads*channels)


class DeterministicMSDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, shapes, starts, locations, weights, step):
        COUNTERS['forward'] += 1
        ctx.save_for_backward(value, shapes, starts, locations, weights)
        # Function.forward runs without an autograd graph. The original CUDA
        # backward is never part of the train graph, while forward is identical.
        return ORIGINAL_APPLY(value, shapes, starts, locations, weights, step)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        COUNTERS['backward'] += 1
        assert torch.are_deterministic_algorithms_enabled()
        value, shapes, starts, locations, weights = ctx.saved_tensors
        gv = torch.zeros_like(value)
        gl = torch.zeros_like(locations)
        ga = torch.zeros_like(weights)
        with torch.enable_grad():
            for level, ((h, w), start) in enumerate(zip(shapes.tolist(), starts.tolist())):
                v = value[:, start:start+h*w].detach().requires_grad_(True)
                loc = locations[:, :, :, level].detach().requires_grad_(True)
                attn = weights[:, :, :, level].detach().requires_grad_(True)
                output = level_output(v, loc, attn, h, w)
                grads = torch.autograd.grad(output, (v, loc, attn), gradient,
                                            retain_graph=False, create_graph=False)
                gv[:, start:start+h*w] = grads[0]
                gl[:, :, :, level] = grads[1]
                ga[:, :, :, level] = grads[2]
                del output, grads, v, loc, attn
        return gv, None, None, gl, ga, None


def install():
    MultiScaleDeformableAttnFunction.apply = staticmethod(DeterministicMSDA.apply)


def settings():
    return {'seed_environment': os.environ['PYTHONHASHSEED'],
            'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'CPU_threads': torch.get_num_threads(), 'AMP': False,
            'SDPA_backend': 'math only',
            'attention': 'unchanged MMCV CUDA forward + deterministic per-level bilinear-gather backward'}
