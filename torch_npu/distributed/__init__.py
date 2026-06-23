__all__ = [
    "is_hccl_available", "reinit_process_group", "reduce_scatter_tensor_uneven", "all_gather_into_tensor_uneven"
]

import os

from torch.distributed import _make_nccl_premul_sum as _make_hccl_premul_sum

import torch_npu
from torch_npu.utils._error_code import ErrCode, dist_error

_ENABLE_EAGER_DISTRIBUTED_INIT = os.getenv("TORCH_NPU_ENABLE_EAGER_DISTRIBUTED_INIT", "0") == "1"


def is_available():
    """
    Returns ``True`` if the distributed package is available. Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    return hasattr(torch_npu._C, "_c10d_npu_init")


def _raise_distributed_not_initialized(*args, **kwargs):
    raise RuntimeError(
        "torch_npu.distributed eager initialization is disabled. Set "
        "TORCH_NPU_ENABLE_EAGER_DISTRIBUTED_INIT=1 before importing torch_npu "
        "to enable torch_npu distributed APIs." + dist_error(ErrCode.NOT_SUPPORT)
    )


if _ENABLE_EAGER_DISTRIBUTED_INIT:
    if is_available() and not torch_npu._C._c10d_npu_init():
        raise RuntimeError("Failed to initialize torch_npu.distributed" + dist_error(ErrCode.INTERNAL))


    from torch_npu._C._distributed_c10d import (
        ParallelStore,
        _verify_params_across_processes,
        _is_support_hccl_comm_name,
    )

    from torch_npu.distributed import rendezvous, tensor, nn
    from .distributed_c10d import is_hccl_available, reinit_process_group, _reduce_scatter_tensor_uneven as reduce_scatter_tensor_uneven, _all_gather_into_tensor_uneven as all_gather_into_tensor_uneven

    rendezvous._rendezvous_init()
else:
    ParallelStore = None
    _verify_params_across_processes = _raise_distributed_not_initialized
    _is_support_hccl_comm_name = _raise_distributed_not_initialized

    def is_hccl_available():
        return False

    reinit_process_group = _raise_distributed_not_initialized
    reduce_scatter_tensor_uneven = _raise_distributed_not_initialized
    all_gather_into_tensor_uneven = _raise_distributed_not_initialized
