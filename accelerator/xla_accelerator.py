# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import functools
import torch
from .abstract_accelerator import DeepSpeedAccelerator

try:
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend
    import torch_xla.runtime as xr

    from tpu_info import device as tpu_device
    from tpu_info import metrics
    
    XLA_AVAILABLE = True
except ImportError as e:
    XLA_AVAILABLE = False
    print(f"XLA is not available: {e}")
    pass


# accelerator for XLA Devices (TPUs/GPUs)
class XLA_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'xla'
        self._compile_backend = "inductor"
        self._communication_backend_name = 'xla'
        self.max_mem = 0
        self.chip_type, self.count = tpu_device.get_local_chips()
        if not self.chip_type or not self.count:
            raise RuntimeError("No TPU devices found.")


    def is_synchronized_device(self):
        return True

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return 'xla'
        return 'xla:{}'.format(device_index)

    def device(self, device_index=None):
        return xla.device(device_index)

    def set_device(self, device_index):
        return

    def current_device(self):
        return os.environ.get('LOCAL_RANK', 0)

    def current_device_name(self):
        return 'xla'

    def device_count(self):
        return xla.device_count()

    def synchronize(self, device_index=None):
        return xla.sync()

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return xm.set_rng_state(new_state)
        
        return xm.set_rng_state(new_state, self.device(device_index))

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return xm.get_rng_state()
        
        return xm.get_rng_state(self.device(device_index))

    def manual_seed(self, seed):
        return xla.manual_seed(seed)

    def manual_seed_all(self, seed):
        for i in range(self.device_count()):
            self.set_rng_state(seed, i)

    def initial_seed(self):
        return xla.get_rng_seed()

    def default_generator(self, device_index):
        return 

    # Streams/Events
    @property
    def Stream(self):
        return None

    def stream(self, stream):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def current_stream(self, device_index=None):
        return None

    def default_stream(self, device_index=None):
        return None

    @property
    def Event(self):
        return None

    # Memory management
    def empty_cache(self):
        return

    def get_rss(self, device_index=0):
        device_usage = metrics.get_chip_usage(self.chip_type)[device_index]
        mem = device_usage.memory_usage
        if mem > self.max_mem:
            self.max_mem = mem
        return mem

    def reset_rss(self, device_index=0):
        device_usage = metrics.get_chip_usage(self.chip_type)[device_index]
        mem = device_usage.memory_usage
        self.max_mem = mem
        return mem

    def memory_allocated(self, device_index=0):
        return self.get_rss(device_index)

    def max_memory_allocated(self, device_index=0):
        self.get_rss(device_index)
        return self.max_mem

    def reset_max_memory_allocated(self, device_index=0):
        self.reset_rss(device_index)
        return

    def memory_cached(self, device_index=0):
        return self.get_rss(device_index)

    def max_memory_cached(self, device_index=0):
        self.get_rss(device_index)
        return self.max_mem

    def reset_max_memory_cached(self, device_index=0):
        self.reset_rss(device_index)
        return

    def memory_stats(self, device_index=0):
        mem = self.get_rss(device_index)
        mem_stat = {}
        mem_stat['allocated_bytes.all.current'] = mem
        mem_stat['allocated_bytes.all.peak'] = self.max_mem
        return mem_stat

    def reset_peak_memory_stats(self, device_index=0):
        self.reset_rss(device_index)
        return

    def memory_reserved(self, device_index=0):
        return self.get_rss(device_index)

    def max_memory_reserved(self, device_index=0):
        self.get_rss(device_index)
        return self.max_mem

    def total_memory(self, device_index=0):
        device_usage = metrics.get_chip_usage(self.chip_type)[device_index]
        return device_usage.total_memory

    def available_memory(self, device_index=0):
        total = self.total_memory(device_index)
        used = self.memory_allocated(device_index)
        return total - used

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return True

    def supported_dtypes(self):
        return [torch.float, torch.half, torch.bfloat16]
    
    # Misc
    def amp(self):
        return xla.amp.autocast

    def is_available(self):
        return XLA_AVAILABLE

    def range_push(self, msg):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_push(msg)
        return

    def range_pop(self):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_pop()
        return

    def lazy_call(self, callback):
        return xm.add_step_closure(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False


    # Graph operations
    def create_graph(self):
        return None

    def capture_to_graph(self, graph, pool=None, stream=None):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def replay_graph(self, graph):
        return

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        return functools.partial(torch.tensor, dtype=torch.bfloat16, device=xla.device())

    @property
    def ByteTensor(self):
        return functools.partial(torch.tensor, dtype=torch.uint8, device=xla.device())

    @property
    def DoubleTensor(self):
        return functools.partial(torch.tensor, dtype=torch.double, device=xla.device())

    @property
    def FloatTensor(self):
        return functools.partial(torch.tensor, dtype=torch.float, device=xla.device())

    @property
    def HalfTensor(self):
        return functools.partial(torch.tensor, dtype=torch.half, device=xla.device())

    @property
    def IntTensor(self):
        return functools.partial(torch.tensor, dtype=torch.int, device=xla.device())

    @property
    def LongTensor(self):
        return functools.partial(torch.tensor, dtype=torch.long, device=xla.device())
    
    def pin_memory(self, tensor, align_bytes=1):
        return tensor

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.cpu"
        except ImportError:
            return "deepspeed.ops.op_builder.cpu"

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('xla:'):
            return True
        else:
            return False

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, op_name):
        builder_class = self.get_op_builder(op_name)
        if builder_class is not None:
            return builder_class()
        return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            from op_builder.cpu import AsyncIOBuilder, CCLCommBuilder, ShareMemCommBuilder, FusedAdamBuilder, CPUAdamBuilder, NotImplementedBuilder
        except ImportError:
            from deepspeed.ops.op_builder.cpu import AsyncIOBuilder, CCLCommBuilder, ShareMemCommBuilder, FusedAdamBuilder, CPUAdamBuilder, NotImplementedBuilder

        if class_name == "CCLCommBuilder":
            return CCLCommBuilder
        elif class_name == "ShareMemCommBuilder":
            return ShareMemCommBuilder
        elif class_name == "FusedAdamBuilder":
            return FusedAdamBuilder
        elif class_name == "CPUAdamBuilder":
            return CPUAdamBuilder
        elif class_name == "AsyncIOBuilder":
            return AsyncIOBuilder
        else:
            # return a NotImplementedBuilder to avoid get NoneType[Name] in unit tests
            return NotImplementedBuilder

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return []

    # TODO: cpu's visible envs is confirmed, keep as CUDA_VISIBLE_DEVICES
    def visible_devices_envs(self):
        return ['CUDA_VISIBLE_DEVICES']

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))

    def get_compile_backend(self):
        return self._compile_backend

    def set_compile_backend(self, backend):
        supported_backends = torch._dynamo.list_backends(exclude_tags=())
        if backend in supported_backends:
            self._compile_backend = backend
        else:
            raise ValueError(
                f"{backend} not supported by {self.device_name()}. Supported Backends are {supported_backends}")


